from pathlib import Path
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
CACHE = Path(os.environ.get(
    "CACHE",
    BASE / "pgrf_backbone_sequence_cache_v1" / "stmem" / "pgrf_stmem_sequence_cache_temporal.npz",
))
BACKBONE = os.environ.get("BACKBONE", "ST-MEM")
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"pgrf_ablation_{BACKBONE.lower().replace('-', '_').replace(' ', '_')}_temporal_v1",
)
OUT = BASE / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["in_hospital_mortality", "mortality_30d", "mortality_1y"]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,123,1024").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "60"))
BS = int(os.environ.get("BS", "512"))
LR = float(os.environ.get("LR", "8e-4"))
KL_WEIGHT = float(os.environ.get("KL_WEIGHT", "0.02"))
BETA_L1_WEIGHT = float(os.environ.get("BETA_L1_WEIGHT", "0.002"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS = [
    "gru_only",
    "pgrf_no_time_decay_prior",
    "pgrf_no_last_ecg_anchor",
    "pgrf_no_recurrent_branch",
    "pgrf_no_residual_gate",
    "full_pgrf",
]

PRETTY = {
    "gru_only": "GRU only",
    "pgrf_no_time_decay_prior": "PGRF w/o time-decay prior",
    "pgrf_no_last_ecg_anchor": "PGRF w/o last-ECG anchor",
    "pgrf_no_recurrent_branch": "PGRF w/o recurrent branch",
    "pgrf_no_residual_gate": "PGRF w/o residual gate",
    "full_pgrf": "Full PGRF",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def roc_auc_score_np(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and p[order[j]] == p[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_score_np(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / (np.arange(len(ys)) + 1)
    return float((precision * ys).sum() / n_pos)


def ece_score(y, p, bins=15):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ((p >= lo) & (p < hi)) if hi < 1 else ((p >= lo) & (p <= hi))
        if m.any():
            ece += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def compute_metrics(Y, P):
    per = []
    for j, lab in enumerate(LABELS):
        y = Y[:, j]
        p = np.clip(P[:, j], 1e-6, 1 - 1e-6)
        per.append({
            "label": lab,
            "auroc": roc_auc_score_np(y, p),
            "auprc": average_precision_score_np(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "ece": ece_score(y, p),
            "prevalence": float(y.mean()),
        })
    return {
        "macro_auroc": float(np.nanmean([r["auroc"] for r in per])),
        "macro_auprc": float(np.nanmean([r["auprc"] for r in per])),
        "mean_brier": float(np.nanmean([r["brier"] for r in per])),
        "mean_nll": float(np.nanmean([r["nll"] for r in per])),
        "mean_ece": float(np.nanmean([r["ece"] for r in per])),
        "per_label": per,
    }


def make_age_features(Dseq):
    age0 = np.log1p(np.clip(Dseq, 0, 3650)) / np.log1p(3650)
    age1 = np.exp(-np.clip(Dseq, 0, 3650) / 30.0)
    return np.stack([age0, age1], axis=-1).astype("float32")


def build_prior(Dseq, Mask, mode="time_decay"):
    if mode == "uniform":
        w = Mask.astype("float32")
    else:
        w = np.exp(-np.clip(Dseq, 0, 3650) / 30.0).astype("float32")
        w[~Mask] = 0.0
    s = w.sum(axis=1, keepdims=True)
    return (w / np.maximum(s, 1e-8)).astype("float32")


def standardize_seq_np(Xseq, Mask, tr):
    flat = Xseq[tr][Mask[tr]].reshape(-1, Xseq.shape[-1])
    mu = flat.mean(axis=0, keepdims=True)
    sd = flat.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((Xseq - mu) / sd).astype("float32")


class PGRFAblation(nn.Module):
    def __init__(self, d, variant, hidden=128, k=3, dropout=0.15):
        super().__init__()
        self.variant = variant
        self.use_anchor = variant != "pgrf_no_last_ecg_anchor"
        self.use_recurrent = variant not in ["pgrf_no_recurrent_branch"]
        self.use_attn_residual = variant != "pgrf_no_residual_gate"
        self.gru_only = variant == "gru_only"

        if not self.gru_only:
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

            # last, prior_pooled, pooled, delta_prior, delta_gate = d*5.
            # If anchor is removed, replace last and deltas with zeros but keep
            # dimensionality fixed to isolate the anchoring information.
            base_dim = d * 5 + 5
            self.base_head = nn.Sequential(
                nn.Linear(base_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, k),
            )

        self.gru = nn.GRU(d + 2, hidden, batch_first=True)
        self.gru_head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, k))
        # Endpoint-wise residual gate. This matches the main PGRF implementation:
        # beta has shape [batch, K], not a single sample-level scalar.
        self.beta_head = nn.Sequential(nn.Linear(hidden + 5, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, k))
        if not self.gru_only:
            nn.init.zeros_(self.gru_head[-1].weight)
            nn.init.zeros_(self.gru_head[-1].bias)
            nn.init.zeros_(self.beta_head[-1].weight)
            nn.init.constant_(self.beta_head[-1].bias, -1.5)

    def forward(self, x, age, mask, prior):
        lengths = mask.sum(dim=1).clamp_min(1)
        xa = torch.cat([x, age], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(xa, lengths.long().cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.gru(packed)
        hgru = hn[-1]

        if self.gru_only:
            beta = torch.ones((x.shape[0], self.gru_head[-1].out_features), device=x.device)
            alpha = prior
            return self.gru_head(hgru), alpha, beta

        htok = self.token(xa)
        if self.use_attn_residual:
            attn_resid = 0.75 * torch.tanh(self.attn_residual(htok).squeeze(-1))
        else:
            attn_resid = torch.zeros_like(prior)
        attn_logits = (prior.clamp_min(1e-8).log() + attn_resid).masked_fill(~mask, -1e9)
        alpha = torch.softmax(attn_logits, dim=1)

        pooled = torch.sum(x * alpha.unsqueeze(-1), dim=1)
        prior_pooled = torch.sum(x * prior.unsqueeze(-1), dim=1)
        last = x[torch.arange(x.shape[0], device=x.device), (lengths - 1).long()]
        if not self.use_anchor:
            last_for_feat = torch.zeros_like(last)
            delta_prior = torch.zeros_like(last)
            delta_gate = torch.zeros_like(last)
        else:
            last_for_feat = last
            delta_prior = last - prior_pooled
            delta_gate = last - pooled

        alpha_entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        prior_entropy = -(prior.clamp_min(1e-8) * prior.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        max_alpha = alpha.max(dim=1, keepdim=True).values
        max_prior = prior.max(dim=1, keepdim=True).values
        len_feat = torch.log1p(lengths.float()).unsqueeze(1)
        gate_stats = torch.cat([alpha_entropy, prior_entropy, max_alpha, max_prior, len_feat], dim=1)

        base_feat = torch.cat([
            last_for_feat, prior_pooled, pooled, delta_prior, delta_gate,
            alpha_entropy, prior_entropy, max_alpha, max_prior, len_feat,
        ], dim=1)
        base_logits = self.base_head(base_feat)

        if not self.use_recurrent:
            beta = torch.zeros((x.shape[0], base_logits.shape[1]), device=x.device)
            return base_logits, alpha, beta

        beta = torch.sigmoid(self.beta_head(torch.cat([hgru, gate_stats], dim=1)))
        logits = base_logits + beta * torch.tanh(self.gru_head(hgru))
        return logits, alpha, beta


def batch_predict_logits(model, X, A, M, P, bs=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for st in range(0, len(X), bs):
            logits, _, _ = model(
                torch.tensor(X[st:st+bs], dtype=torch.float32, device=DEVICE),
                torch.tensor(A[st:st+bs], dtype=torch.float32, device=DEVICE),
                torch.tensor(M[st:st+bs], dtype=torch.bool, device=DEVICE),
                torch.tensor(P[st:st+bs], dtype=torch.float32, device=DEVICE),
            )
            outs.append(logits.cpu().numpy())
    return np.concatenate(outs, axis=0)


def fit_platt(val_logits, Yv, test_logits):
    out = []
    xv = torch.tensor(val_logits, dtype=torch.float32, device=DEVICE)
    xt = torch.tensor(test_logits, dtype=torch.float32, device=DEVICE)
    for j in range(Yv.shape[1]):
        y = torch.tensor(Yv[:, j:j+1], dtype=torch.float32, device=DEVICE)
        model = nn.Linear(1, 1)
        model.to(DEVICE)
        opt = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=80)
        col = xv[:, j:j+1]
        pos = float(Yv[:, j].sum())
        neg = float(len(Yv) - pos)
        pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(col), y, pos_weight=pw)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            out.append(torch.sigmoid(model(xt[:, j:j+1])).cpu().numpy().reshape(-1))
    return np.stack(out, axis=1).astype("float32")


def train_one_variant(seed, variant, Xn, Age, Prior, Mask, Y, split_arr):
    set_seed(seed)
    tr = split_arr == "train"
    va = split_arr == "val"
    te = split_arr == "test"
    model = PGRFAblation(d=Xn.shape[-1], variant=variant).to(DEVICE)
    pos = Y[tr].sum(0)
    neg = tr.sum() - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 80), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    train_idx = np.where(tr)[0]
    best, best_state, wait, hist = -1.0, None, 0, []

    for ep in range(EPOCHS):
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(train_idx)
        model.train()
        losses, betas = [], []
        for st in range(0, len(train_idx), BS):
            idx = train_idx[st:st + BS]
            x = torch.tensor(Xn[idx], dtype=torch.float32, device=DEVICE)
            a = torch.tensor(Age[idx], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(Mask[idx], dtype=torch.bool, device=DEVICE)
            p0 = torch.tensor(Prior[idx], dtype=torch.float32, device=DEVICE)
            y = torch.tensor(Y[idx], dtype=torch.float32, device=DEVICE)
            logits, alpha, beta = model(x, a, m, p0)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            if variant not in ["gru_only", "pgrf_no_time_decay_prior"]:
                kl = (
                    alpha.clamp_min(1e-8)
                    * (alpha.clamp_min(1e-8).log() - p0.clamp_min(1e-8).log())
                ).masked_fill(~m, 0.0).sum(dim=1).mean()
                loss = loss + KL_WEIGHT * kl
            if variant not in ["gru_only", "pgrf_no_recurrent_branch"]:
                loss = loss + BETA_L1_WEIGHT * beta.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            betas.append(float(beta.detach().mean().cpu()))

        if ep % 3 == 0 or ep == EPOCHS - 1:
            val_logits = batch_predict_logits(model, Xn[va], Age[va], Mask[va], Prior[va])
            val_probs = 1 / (1 + np.exp(-val_logits))
            score = compute_metrics(Y[va], val_probs)["macro_auprc"]
            hist.append({"seed": seed, "variant": variant, "epoch": ep, "loss": float(np.mean(losses)), "beta": float(np.mean(betas)), "val_macro_auprc": score})
            if score > best + 1e-5:
                best, wait = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= 8:
                break

    model.load_state_dict(best_state)
    val_logits = batch_predict_logits(model, Xn[va], Age[va], Mask[va], Prior[va])
    test_logits = batch_predict_logits(model, Xn[te], Age[te], Mask[te], Prior[te])
    raw_probs = 1 / (1 + np.exp(-test_logits))
    platt_probs = fit_platt(val_logits, Y[va], test_logits)
    return raw_probs.astype("float32"), platt_probs.astype("float32"), hist


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["variant"], r["calibration"])
        groups.setdefault(key, []).append(r)
    out = []
    for (variant, cal), rs in groups.items():
        item = {"variant": variant, "variant_pretty": PRETTY[variant], "calibration": cal, "n_seeds": len(rs)}
        for m in ["macro_auroc", "macro_auprc", "mean_brier", "mean_nll", "mean_ece"]:
            vals = np.array([float(r[m]) for r in rs], dtype=float)
            item[m + "_mean"] = float(vals.mean())
            item[m + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    order = {v: i for i, v in enumerate(VARIANTS)}
    out.sort(key=lambda r: (order[r["variant"]], r["calibration"]))
    return out


def main():
    started = time.time()
    dat = np.load(CACHE, allow_pickle=True)
    Xseq = dat["Xseq"].astype("float32")
    Dseq = dat["Dseq"].astype("float32")
    Mask = dat["mask"].astype(bool)
    Y = dat["Y"].astype("float32")
    split_arr = dat["split"].astype(str)
    tr = split_arr == "train"
    te = split_arr == "test"
    Xn = standardize_seq_np(Xseq, Mask, tr)
    Age = make_age_features(Dseq)
    print("[BOOT]", DEVICE, CACHE, OUT, "shape", Xseq.shape, {s: int((split_arr == s).sum()) for s in ["train", "val", "test"]}, flush=True)

    rows, per_rows, hist_rows = [], [], []
    for variant in VARIANTS:
        prior_mode = "uniform" if variant == "pgrf_no_time_decay_prior" else "time_decay"
        Prior = build_prior(Dseq, Mask, mode=prior_mode)
        for seed in SEEDS:
            print("[TRAIN]", variant, "seed", seed, flush=True)
            raw_p, platt_p, hist = train_one_variant(seed, variant, Xn, Age, Prior, Mask, Y, split_arr)
            hist_rows.extend(hist)
            for cal, P in [("raw", raw_p), ("platt", platt_p)]:
                m = compute_metrics(Y[te], P)
                rows.append({
                    "backbone": BACKBONE,
                    "split": "temporal",
                    "variant": variant,
                    "variant_pretty": PRETTY[variant],
                    "calibration": cal,
                    "seed": seed,
                    **{k: v for k, v in m.items() if k != "per_label"},
                })
                for r in m["per_label"]:
                    per_rows.append({
                        "backbone": BACKBONE,
                        "split": "temporal",
                        "variant": variant,
                        "variant_pretty": PRETTY[variant],
                        "calibration": cal,
                        "seed": seed,
                        **r,
                    })
            np.savez_compressed(OUT / f"{variant}_seed{seed}_test_probs.npz", y_true=Y[te], raw_prob=raw_p, platt_prob=platt_p)

    summary = summarize(rows)
    write_csv(OUT / "pgrf_ablation_raw_metrics.csv", rows, [
        "backbone", "split", "variant", "variant_pretty", "calibration", "seed",
        "macro_auroc", "macro_auprc", "mean_brier", "mean_nll", "mean_ece",
    ])
    write_csv(OUT / "pgrf_ablation_per_label_metrics.csv", per_rows, [
        "backbone", "split", "variant", "variant_pretty", "calibration", "seed",
        "label", "auroc", "auprc", "brier", "nll", "ece", "prevalence",
    ])
    write_csv(OUT / "pgrf_ablation_training_history.csv", hist_rows, ["seed", "variant", "epoch", "loss", "beta", "val_macro_auprc"])
    write_csv(OUT / "pgrf_ablation_mean_std.csv", summary, [
        "variant", "variant_pretty", "calibration", "n_seeds",
        "macro_auroc_mean", "macro_auroc_std", "macro_auprc_mean", "macro_auprc_std",
        "mean_brier_mean", "mean_brier_std", "mean_nll_mean", "mean_nll_std",
        "mean_ece_mean", "mean_ece_std",
    ])

    lines = [
        f"# {BACKBONE} Temporal PGRF Module Ablation",
        "",
        "| Variant | Calibration | AUROC | AUPRC | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['variant_pretty']} | {r['calibration']} | "
            f"{r['macro_auroc_mean']:.4f} 卤 {r['macro_auroc_std']:.4f} | "
            f"{r['macro_auprc_mean']:.4f} 卤 {r['macro_auprc_std']:.4f} | "
            f"{r['mean_brier_mean']:.4f} 卤 {r['mean_brier_std']:.4f} | "
            f"{r['mean_ece_mean']:.4f} 卤 {r['mean_ece_std']:.4f} |"
        )
    (OUT / "PGRF_PGRF_Ablation_Report.md").write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "pgrf_ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": BACKBONE,
                "cache": str(CACHE),
                "out": str(OUT),
                "seeds": SEEDS,
                "epochs": EPOCHS,
                "beta_gate": "endpoint-wise vector",
                "elapsed_sec": time.time() - started,
            },
            f,
            indent=2,
        )
    print("\n".join(lines), flush=True)
    print("[DONE]", OUT, "elapsed", time.time() - started, flush=True)


if __name__ == "__main__":
    main()


