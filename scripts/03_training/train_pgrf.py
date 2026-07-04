from pathlib import Path
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
OUT = BASE / os.environ.get("RUN_NAME", "pgrf_train_v1")
OUT.mkdir(parents=True, exist_ok=True)

CACHES = {
    "random": Path(os.environ["RANDOM_CACHE"]),
    "temporal": Path(os.environ["TEMPORAL_CACHE"]),
}

LABELS = ["in_hospital_mortality", "mortality_30d", "mortality_1y"]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,123,1024").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "60"))
BS = int(os.environ.get("BS", "512"))
LR = float(os.environ.get("LR", "8e-4"))
KL_WEIGHT = float(os.environ.get("KL_WEIGHT", "0.02"))
GRU_RESIDUAL_SCALE = float(os.environ.get("GRU_RESIDUAL_SCALE", "1.0"))
BETA_L1_WEIGHT = float(os.environ.get("BETA_L1_WEIGHT", "0.002"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ece_score(y, p, bins=15):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = ((p >= lo) & (p < hi)) if hi < 1 else ((p >= lo) & (p <= hi))
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def metrics(Y, P):
    rows = []
    for j, lab in enumerate(LABELS):
        y = Y[:, j]
        p = np.clip(P[:, j], 1e-6, 1 - 1e-6)
        rows.append({
            "label": lab,
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
            "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
            "brier": float(brier_score_loss(y, p)),
            "nll": float(log_loss(y, p, labels=[0, 1])),
            "ece": ece_score(y, p),
            "prevalence": float(y.mean()),
            "n": int(len(y)),
            "positives": int(y.sum()),
        })
    return {
        "macro_auroc": float(np.nanmean([r["auroc"] for r in rows])),
        "macro_auprc": float(np.nanmean([r["auprc"] for r in rows])),
        "mean_brier": float(np.nanmean([r["brier"] for r in rows])),
        "mean_nll": float(np.nanmean([r["nll"] for r in rows])),
        "mean_ece": float(np.nanmean([r["ece"] for r in rows])),
        "per_label": rows,
    }


def make_age_features(Dseq):
    age0 = np.log1p(np.clip(Dseq, 0, 3650)) / np.log1p(3650)
    age1 = np.exp(-np.clip(Dseq, 0, 3650) / 30.0)
    return np.stack([age0, age1], axis=-1).astype("float32")


def build_prior(Dseq, Mask):
    w = np.exp(-np.clip(Dseq, 0, 3650) / 30.0).astype("float32")
    w[~Mask] = 0.0
    s = w.sum(axis=1, keepdims=True)
    return (w / np.maximum(s, 1e-8)).astype("float32")


def standardize_seq(Xseq, Mask, tr):
    scaler = StandardScaler().fit(Xseq[tr][Mask[tr]].reshape(-1, Xseq.shape[-1]))
    Xn = Xseq.copy()
    flat = Xn.reshape(-1, Xn.shape[-1])
    flat[:] = scaler.transform(flat).astype("float32")
    return flat.reshape(Xseq.shape).astype("float32")


class PriorGuidedGRUGate(nn.Module):
    """Prior-residual attention plus a bounded recurrent residual branch.

    The base branch is the current prior-residual gate. A GRU branch can add only a
    bounded residual logit, controlled by a learned scalar beta. This keeps the
    paper's mainline: recurrent capacity is regularized by the temporal prior and
    last-ECG anchoring rather than replacing them.
    """

    def __init__(self, d, hidden=128, k=3, dropout=0.15, gru_residual_scale=1.0):
        super().__init__()
        self.gru_residual_scale = gru_residual_scale
        self.token = nn.Sequential(
            nn.Linear(d + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.attn_residual = nn.Linear(hidden, 1)
        nn.init.zeros_(self.attn_residual.weight)
        nn.init.zeros_(self.attn_residual.bias)

        self.gru = nn.GRU(d + 2, hidden, batch_first=True)

        base_dim = d * 5 + 5
        self.base_head = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )
        self.gru_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden + 5, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Start close to the base prior-residual model. The GRU residual must earn
        # its way in during validation-based early stopping.
        nn.init.zeros_(self.gru_head[-1].weight)
        nn.init.zeros_(self.gru_head[-1].bias)
        nn.init.zeros_(self.beta_head[-1].weight)
        nn.init.constant_(self.beta_head[-1].bias, -1.5)

    def forward(self, x, age, mask, prior):
        htok = self.token(torch.cat([x, age], dim=-1))
        attn_resid = 0.75 * torch.tanh(self.attn_residual(htok).squeeze(-1))
        attn_logits = (prior.clamp_min(1e-8).log() + attn_resid).masked_fill(~mask, -1e9)
        alpha = torch.softmax(attn_logits, dim=1)

        pooled = torch.sum(x * alpha.unsqueeze(-1), dim=1)
        prior_pooled = torch.sum(x * prior.unsqueeze(-1), dim=1)
        lengths = mask.sum(dim=1).clamp_min(1)
        last = x[torch.arange(x.shape[0], device=x.device), (lengths - 1).long()]
        delta_gate = last - pooled
        delta_prior = last - prior_pooled
        alpha_entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        prior_entropy = -(prior.clamp_min(1e-8) * prior.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        max_alpha = alpha.max(dim=1, keepdim=True).values
        max_prior = prior.max(dim=1, keepdim=True).values
        len_feat = torch.log1p(lengths.float()).unsqueeze(1)

        base_feat = torch.cat([
            last, prior_pooled, pooled, delta_prior, delta_gate,
            alpha_entropy, prior_entropy, max_alpha, max_prior, len_feat,
        ], dim=1)
        base_logits = self.base_head(base_feat)

        xa = torch.cat([x, age], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            xa, lengths.long().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hn = self.gru(packed)
        hgru = hn[-1]
        gate_stats = torch.cat([alpha_entropy, prior_entropy, max_alpha, max_prior, len_feat], dim=1)
        beta = torch.sigmoid(self.beta_head(torch.cat([hgru, gate_stats], dim=1)))
        gru_delta = self.gru_residual_scale * beta * torch.tanh(self.gru_head(hgru))
        logits = base_logits + gru_delta
        return logits, alpha, beta


def train_one(seed, Xn, Age, Prior, Mask, Y, split_arr):
    set_seed(seed)
    tr = split_arr == "train"
    va = split_arr == "val"
    te = split_arr == "test"
    model = PriorGuidedGRUGate(d=Xn.shape[-1], gru_residual_scale=GRU_RESIDUAL_SCALE).to(DEVICE)
    pos = Y[tr].sum(0)
    neg = tr.sum() - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 80), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    train_idx = np.where(tr)[0]

    vx = torch.tensor(Xn[va], dtype=torch.float32, device=DEVICE)
    vaa = torch.tensor(Age[va], dtype=torch.float32, device=DEVICE)
    vm = torch.tensor(Mask[va], dtype=torch.bool, device=DEVICE)
    vp = torch.tensor(Prior[va], dtype=torch.float32, device=DEVICE)

    best, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(EPOCHS):
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(train_idx)
        model.train()
        losses, betas = [], []
        for st in range(0, len(train_idx), BS):
            j = train_idx[st:st + BS]
            x = torch.tensor(Xn[j], dtype=torch.float32, device=DEVICE)
            a = torch.tensor(Age[j], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(Mask[j], dtype=torch.bool, device=DEVICE)
            p0 = torch.tensor(Prior[j], dtype=torch.float32, device=DEVICE)
            y = torch.tensor(Y[j], dtype=torch.float32, device=DEVICE)
            logits, alpha, beta = model(x, a, m, p0)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            kl = (
                alpha.clamp_min(1e-8)
                * (alpha.clamp_min(1e-8).log() - p0.clamp_min(1e-8).log())
            ).masked_fill(~m, 0.0).sum(dim=1).mean()
            loss = bce + KL_WEIGHT * kl + BETA_L1_WEIGHT * beta.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            betas.append(float(beta.detach().mean().cpu()))

        if ep % 3 == 0 or ep == EPOCHS - 1:
            model.eval()
            outs = []
            with torch.no_grad():
                for st in range(0, len(vx), 2048):
                    logits, _, _ = model(vx[st:st + 2048], vaa[st:st + 2048], vm[st:st + 2048], vp[st:st + 2048])
                    outs.append(torch.sigmoid(logits).cpu().numpy())
            pv = np.concatenate(outs, axis=0)
            score = metrics(Y[va], pv)["macro_auprc"]
            hist.append({
                "seed": seed,
                "epoch": ep,
                "train_loss": float(np.mean(losses)),
                "train_beta": float(np.mean(betas)),
                "val_macro_auprc": float(score),
            })
            if score > best + 1e-5:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()
    tx = torch.tensor(Xn[te], dtype=torch.float32, device=DEVICE)
    ta = torch.tensor(Age[te], dtype=torch.float32, device=DEVICE)
    tm = torch.tensor(Mask[te], dtype=torch.bool, device=DEVICE)
    tp = torch.tensor(Prior[te], dtype=torch.float32, device=DEVICE)
    probs, alphas, betas = [], [], []
    with torch.no_grad():
        for st in range(0, len(tx), 2048):
            logits, alpha, beta = model(tx[st:st + 2048], ta[st:st + 2048], tm[st:st + 2048], tp[st:st + 2048])
            probs.append(torch.sigmoid(logits).cpu().numpy())
            alphas.append(alpha.cpu().numpy())
            betas.append(beta.cpu().numpy())
    return np.concatenate(probs, axis=0), np.concatenate(alphas, axis=0), np.concatenate(betas, axis=0), hist


def run_cache(dataset_name, cache_path):
    print(f"[LOAD] {dataset_name} {cache_path}", flush=True)
    dat = np.load(cache_path, allow_pickle=True)
    Xseq = dat["Xseq"].astype("float32")
    Dseq = dat["Dseq"].astype("float32")
    Mask = dat["mask"].astype(bool)
    Y = dat["Y"].astype("float32")
    split_arr = dat["split"].astype(str)
    tr = split_arr == "train"
    te = split_arr == "test"
    Xn = standardize_seq(Xseq, Mask, tr)
    Age = make_age_features(Dseq)
    Prior = build_prior(Dseq, Mask)
    print("[DATA]", dataset_name, Xseq.shape, {s: int((split_arr == s).sum()) for s in ["train", "val", "test"]}, flush=True)

    rows, per_rows, hist_rows, stat_rows = [], [], [], []
    for seed in SEEDS:
        print("[TRAIN] prior_guided_gru_gate", dataset_name, seed, flush=True)
        P, alpha, beta, hist = train_one(seed, Xn, Age, Prior, Mask, Y, split_arr)
        hist_rows.extend([{"dataset": dataset_name, **h} for h in hist])
        m = metrics(Y[te], P)
        rows.append({
            "dataset": dataset_name,
            "method": "prior_guided_gru_gate",
            "seed": seed,
            **{k: v for k, v in m.items() if k != "per_label"},
        })
        for r in m["per_label"]:
            per_rows.append({"dataset": dataset_name, "method": "prior_guided_gru_gate", "seed": seed, **r})
        lengths = Mask[te].sum(axis=1).clip(1)
        last_alpha = np.array([alpha[i, int(lengths[i]) - 1] for i in range(len(alpha))], dtype=np.float32)
        prior_te = Prior[te]
        prior_last = np.array([prior_te[i, int(lengths[i]) - 1] for i in range(len(alpha))], dtype=np.float32)
        stat_rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "mean_beta": float(beta.mean()),
            "mean_max_alpha": float(alpha.max(axis=1).mean()),
            "mean_alpha_last_position": float(last_alpha.mean()),
            "mean_prior_last_position": float(prior_last.mean()),
            "mean_abs_alpha_minus_prior": float(np.abs(alpha - prior_te).mean()),
        })
        np.savez_compressed(
            OUT / f"{dataset_name}_prior_guided_gru_gate_seed{seed}_test_probs.npz",
            y_true=Y[te], y_prob=P, alpha=alpha, beta=beta, prior=prior_te, mask=Mask[te], Dseq=Dseq[te],
        )
    return rows, per_rows, hist_rows, stat_rows


def main():
    started = time.time()
    print("[BOOT]", DEVICE, OUT, "seeds", SEEDS, "kl", KL_WEIGHT, "gru_scale", GRU_RESIDUAL_SCALE, "beta_l1", BETA_L1_WEIGHT, flush=True)
    all_rows, all_per, all_hist, all_stats = [], [], [], []
    for name, cache_path in CACHES.items():
        rows, per, hist, stats = run_cache(name, cache_path)
        all_rows.extend(rows)
        all_per.extend(per)
        all_hist.extend(hist)
        all_stats.extend(stats)

    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUT / "prior_guided_gru_gate_raw_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_per).to_csv(OUT / "prior_guided_gru_gate_per_label_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_hist).to_csv(OUT / "prior_guided_gru_gate_training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_stats).to_csv(OUT / "prior_guided_gru_gate_stats.csv", index=False, encoding="utf-8-sig")
    metric_cols = ["macro_auroc", "macro_auprc", "mean_brier", "mean_nll", "mean_ece"]
    summary = raw.groupby(["dataset", "method"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(OUT / "prior_guided_gru_gate_mean_std.csv", index=False, encoding="utf-8-sig")

    payload = {
        "out_dir": str(OUT),
        "caches": {k: str(v) for k, v in CACHES.items()},
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "batch_size": BS,
        "kl_weight": KL_WEIGHT,
        "gru_residual_scale": GRU_RESIDUAL_SCALE,
        "beta_l1_weight": BETA_L1_WEIGHT,
        "device": str(DEVICE),
        "elapsed_sec": float(time.time() - started),
    }
    with open(OUT / "prior_guided_gru_gate_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# PGRF Prior-guided GRU Residual Gate",
        "",
        "This variant keeps the prior-residual gate as the base predictor and adds a bounded GRU residual logit branch.",
        "",
        f"Seeds: {SEEDS}; KL={KL_WEIGHT}; GRU residual scale={GRU_RESIDUAL_SCALE}; beta L1={BETA_L1_WEIGHT}",
        "",
        "| Dataset | Method | Macro-AUROC | Macro-AUPRC | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['method']} | "
            f"{r['macro_auroc_mean']:.4f}卤{r['macro_auroc_std']:.4f} | "
            f"{r['macro_auprc_mean']:.4f}卤{r['macro_auprc_std']:.4f} | "
            f"{r['mean_brier_mean']:.4f}卤{r['mean_brier_std']:.4f} | "
            f"{r['mean_ece_mean']:.4f}卤{r['mean_ece_std']:.4f} |"
        )
    (OUT / "PGRF_Prior_Guided_GRU_Gate_Report.md").write_text("\n".join(lines), encoding="utf-8")
    print(raw.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(all_stats).to_string(index=False), flush=True)
    print("[DONE]", OUT, "elapsed", time.time() - started, flush=True)


if __name__ == "__main__":
    main()


