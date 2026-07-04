#!/usr/bin/env python3
"""
PGRF count>=2 supplemental pooling baselines without pandas/sklearn.

Baselines:
  - mean_pool: simple average of all valid ECG embeddings.
  - time_decay_pool: fixed exponential time-decay weighted pooling.
  - attention_pool: learned attention pooling without time-decay prior.

The script mirrors the PGRF/PGRF training setting as closely as possible:
same cache format, same seeds, weighted BCE, validation macro-AUPRC early
stopping, and per-label Platt scaling using validation logits.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
OUT = BASE / os.environ.get("RUN_NAME", "pgrf_longitudinal_count_ge2_v1/pooling_baselines_v1")
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_PGRF_CACHE = BASE / "pgrf_longitudinal_count_ge2_v1" / "caches"
DEFAULT_CACHE_SPEC = ";".join([
    f"ST-MEM,temporal,{DEFAULT_PGRF_CACHE / 'st_mem_temporal_count_ge2.npz'}",
    f"ECG-FM,temporal,{DEFAULT_PGRF_CACHE / 'ecg_fm_temporal_count_ge2.npz'}",
    f"CLEAR-HUG,temporal,{DEFAULT_PGRF_CACHE / 'clear_hug_temporal_count_ge2.npz'}",
    f"MERL,temporal,{DEFAULT_PGRF_CACHE / 'merl_temporal_count_ge2.npz'}",
    f"MELP,temporal,{DEFAULT_PGRF_CACHE / 'melp_temporal_count_ge2.npz'}",
])

CACHES = {}
for item in os.environ.get("CACHE_SPEC", DEFAULT_CACHE_SPEC).split(";"):
    if not item.strip():
        continue
    b, s, p = item.split(",", 2)
    CACHES[(b, s)] = Path(p)

LABELS = ["in_hospital_mortality", "mortality_30d", "mortality_1y"]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,123,1024").split(",") if x.strip()]
METHODS = [x.strip() for x in os.environ.get("METHODS", "mean_pool,time_decay_pool,attention_pool").split(",") if x.strip()]
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BS = int(os.environ.get("BS", "512"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-3"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "3"))
PATIENCE = int(os.environ.get("PATIENCE", "8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def binary_auroc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    npos = int(y.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and p[order[j]] == p[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg))


def binary_auprc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    npos = int(y.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / (np.arange(len(y)) + 1.0)
    return float(precision[ys == 1].sum() / npos)


def ece_score(y, p, bins=15):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def metrics(Y, P):
    rows = []
    P = np.clip(np.asarray(P), 1e-6, 1 - 1e-6)
    for j, lab in enumerate(LABELS):
        y = Y[:, j].astype(int)
        p = P[:, j]
        rows.append({
            "label": lab,
            "auroc": binary_auroc(y, p),
            "auprc": binary_auprc(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
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


def standardize_inplace(X, Mask, train_mask, chunk=2048):
    d = X.shape[-1]
    total = np.zeros(d, dtype=np.float64)
    total2 = np.zeros(d, dtype=np.float64)
    count = 0
    idx = np.where(train_mask)[0]
    for st in range(0, len(idx), chunk):
        j = idx[st:st + chunk]
        vals = X[j][Mask[j]]
        if vals.size:
            total += vals.sum(axis=0, dtype=np.float64)
            total2 += (vals.astype(np.float64) ** 2).sum(axis=0)
            count += vals.shape[0]
    mean = total / max(count, 1)
    var = np.maximum(total2 / max(count, 1) - mean ** 2, 1e-6)
    std = np.sqrt(var)
    mean32 = mean.astype("float32")
    std32 = std.astype("float32")
    for st in range(0, X.shape[0], chunk):
        X[st:st + chunk] = (X[st:st + chunk] - mean32) / std32
    return mean32, std32


class MeanPool(nn.Module):
    def __init__(self, d, hidden=128, k=3, dropout=0.15):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )

    def forward(self, x, age, mask, prior):
        denom = mask.sum(1).clamp_min(1).float().unsqueeze(1)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / denom
        return self.head(pooled)


class LastECG(nn.Module):
    def __init__(self, d, hidden=128, k=3, dropout=0.15):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )

    def forward(self, x, age, mask, prior):
        lengths = mask.sum(1).clamp_min(1)
        last = x[torch.arange(x.shape[0], device=x.device), (lengths - 1).long()]
        return self.head(last)


class TimeDecayPool(nn.Module):
    def __init__(self, d, hidden=128, k=3, dropout=0.15):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )

    def forward(self, x, age, mask, prior):
        pooled = (x * prior.unsqueeze(-1)).sum(1)
        return self.head(pooled)


class AttentionPool(nn.Module):
    def __init__(self, d, hidden=128, k=3, dropout=0.15):
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(d + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.score = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(d * 3 + 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )

    def forward(self, x, age, mask, prior):
        h = self.token(torch.cat([x, age], dim=-1))
        logits = self.score(h).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = torch.softmax(logits, dim=1)
        pooled = (x * alpha.unsqueeze(-1)).sum(1)
        lengths = mask.sum(1).clamp_min(1)
        last = x[torch.arange(x.shape[0], device=x.device), (lengths - 1).long()]
        delta = last - pooled
        ent = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(1, keepdim=True)
        max_a = alpha.max(1, keepdim=True).values
        len_feat = torch.log1p(lengths.float()).unsqueeze(1)
        return self.head(torch.cat([last, pooled, delta, ent, max_a, len_feat], dim=1))


FACTORIES = {
    "last_ecg": LastECG,
    "mean_pool": MeanPool,
    "time_decay_pool": TimeDecayPool,
    "attention_pool": AttentionPool,
}


def evaluate_logits(model, X, Age, Mask, Prior, indices, batch=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for st in range(0, len(indices), batch):
            j = indices[st:st + batch]
            x = torch.tensor(X[j], dtype=torch.float32, device=DEVICE)
            a = torch.tensor(Age[j], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(Mask[j], dtype=torch.bool, device=DEVICE)
            p0 = torch.tensor(Prior[j], dtype=torch.float32, device=DEVICE)
            outs.append(model(x, a, m, p0).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_model(method, seed, X, Age, Mask, Prior, Y, split_arr):
    set_seed(seed)
    tr = split_arr == "train"
    va = split_arr == "val"
    te = split_arr == "test"
    tr_idx = np.where(tr)[0]
    va_idx = np.where(va)[0]
    te_idx = np.where(te)[0]
    model = FACTORIES[method](d=X.shape[-1]).to(DEVICE)
    pos = Y[tr].sum(0)
    neg = int(tr.sum()) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 80), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = -1.0
    best_state = None
    wait = 0
    hist = []
    for ep in range(EPOCHS):
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(tr_idx)
        model.train()
        losses = []
        for st in range(0, len(tr_idx), BS):
            j = tr_idx[st:st + BS]
            x = torch.tensor(X[j], dtype=torch.float32, device=DEVICE)
            a = torch.tensor(Age[j], dtype=torch.float32, device=DEVICE)
            m = torch.tensor(Mask[j], dtype=torch.bool, device=DEVICE)
            p0 = torch.tensor(Prior[j], dtype=torch.float32, device=DEVICE)
            y = torch.tensor(Y[j], dtype=torch.float32, device=DEVICE)
            logits = model(x, a, m, p0)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
            lv = evaluate_logits(model, X, Age, Mask, Prior, va_idx)
            score = metrics(Y[va], sigmoid_np(lv))["macro_auprc"]
            hist.append({
                "method": method,
                "seed": seed,
                "epoch": ep,
                "train_loss": float(np.mean(losses)),
                "val_macro_auprc": float(score),
            })
            print(f"[VAL] {method} seed={seed} ep={ep} loss={np.mean(losses):.4f} val_auprc={score:.4f}", flush=True)
            if score > best + 1e-5:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_logits = evaluate_logits(model, X, Age, Mask, Prior, va_idx)
    test_logits = evaluate_logits(model, X, Age, Mask, Prior, te_idx)
    return val_logits, test_logits, hist


def platt_calibrate_torch(Yv, Lv, Lt):
    """Fit per-label sigmoid(a * logit + b) on validation logits."""
    out = np.zeros_like(Lt, dtype=np.float32)
    for j in range(Yv.shape[1]):
        y_np = Yv[:, j].astype("float32")
        if len(np.unique(y_np)) < 2:
            out[:, j] = sigmoid_np(Lt[:, j])
            continue
        x = torch.tensor(Lv[:, j:j + 1], dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        xt = torch.tensor(Lt[:, j:j + 1], dtype=torch.float32, device=DEVICE)
        a = torch.nn.Parameter(torch.ones(1, 1, device=DEVICE))
        b = torch.nn.Parameter(torch.zeros(1, device=DEVICE))
        opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = x @ a + b
            loss = F.binary_cross_entropy_with_logits(logits, y)
            # Tiny regularizer prevents pathological slopes on rare endpoints.
            loss = loss + 1e-4 * ((a - 1.0) ** 2).mean() + 1e-4 * (b ** 2).mean()
            loss.backward()
            return loss

        try:
            opt.step(closure)
            with torch.no_grad():
                out[:, j] = torch.sigmoid(xt @ a + b).detach().cpu().numpy().ravel()
        except Exception:
            out[:, j] = sigmoid_np(Lt[:, j])
    return out


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(raw_rows):
    metric_cols = ["macro_auroc", "macro_auprc", "mean_brier", "mean_nll", "mean_ece"]
    groups = defaultdict(list)
    for r in raw_rows:
        groups[(r["backbone"], r["split"], r["method"], r["calibration"])].append(r)
    summary = []
    for key, rows in sorted(groups.items()):
        b, s, m, c = key
        out = {"backbone": b, "split": s, "method": m, "calibration": c}
        for col in metric_cols:
            vals = np.array([float(r[col]) for r in rows], dtype=float)
            out[col + "_mean"] = float(np.nanmean(vals))
            out[col + "_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(out)
    return summary


def run_cache(backbone, split_name, cache_path):
    print(f"[LOAD] {backbone} {split_name} {cache_path}", flush=True)
    dat = np.load(cache_path, allow_pickle=True)
    X = dat["Xseq"].astype("float32")
    Dseq = dat["Dseq"].astype("float32")
    Mask = dat["mask"].astype(bool)
    Y = dat["Y"].astype("float32")
    split_arr = dat["split"].astype(str)
    tr = split_arr == "train"
    print("[DATA]", backbone, X.shape, {s: int((split_arr == s).sum()) for s in ["train", "val", "test"]}, flush=True)
    standardize_inplace(X, Mask, tr)
    Age = make_age_features(Dseq)
    Prior = build_prior(Dseq, Mask)
    va = split_arr == "val"
    te = split_arr == "test"
    raw_rows, per_rows, hist_rows = [], [], []
    for method in METHODS:
        if method not in FACTORIES:
            raise ValueError(f"Unknown method={method}; options={list(FACTORIES)}")
        for seed in SEEDS:
            print("[TRAIN]", backbone, split_name, method, seed, flush=True)
            Lv, Lt, hist = train_model(method, seed, X, Age, Mask, Prior, Y, split_arr)
            hist_rows.extend([{"backbone": backbone, "split": split_name, **h} for h in hist])
            Ptest = sigmoid_np(Lt).astype("float32")
            Pcal = platt_calibrate_torch(Y[va], Lv, Lt).astype("float32")
            save_dir = OUT / backbone / split_name / method
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save_dir / f"seed{seed}_val_test_probs.npz",
                y_val=Y[va],
                logits_val=Lv.astype("float32"),
                y_test=Y[te],
                logits_test=Lt.astype("float32"),
                p_test=Ptest,
                p_test_platt=Pcal,
            )
            for calib, P in [("uncalibrated", Ptest), ("platt", Pcal)]:
                mm = metrics(Y[te], P)
                raw_rows.append({
                    "backbone": backbone,
                    "split": split_name,
                    "method": method,
                    "seed": seed,
                    "calibration": calib,
                    **{k: v for k, v in mm.items() if k != "per_label"},
                })
                for rr in mm["per_label"]:
                    per_rows.append({
                        "backbone": backbone,
                        "split": split_name,
                        "method": method,
                        "seed": seed,
                        "calibration": calib,
                        **rr,
                    })
    # Drop large arrays before moving to next backbone.
    del X, Dseq, Mask, Y, split_arr, Age, Prior
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_rows, per_rows, hist_rows


def main():
    started = time.time()
    print("[BOOT]", "device=", DEVICE, "out=", OUT, "methods=", METHODS, "seeds=", SEEDS, flush=True)
    all_raw, all_per, all_hist = [], [], []
    for (backbone, split_name), cache_path in CACHES.items():
        rows, per, hist = run_cache(backbone, split_name, cache_path)
        all_raw.extend(rows)
        all_per.extend(per)
        all_hist.extend(hist)
        write_csv(OUT / "pooling_baselines_raw_metrics.partial.csv", all_raw)
        write_csv(OUT / "pooling_baselines_per_label_metrics.partial.csv", all_per)
        write_csv(OUT / "pooling_baselines_training_history.partial.csv", all_hist)
        write_csv(OUT / "pooling_baselines_mean_std.partial.csv", summarize(all_raw))
    summary = summarize(all_raw)
    write_csv(OUT / "pooling_baselines_raw_metrics.csv", all_raw)
    write_csv(OUT / "pooling_baselines_per_label_metrics.csv", all_per)
    write_csv(OUT / "pooling_baselines_training_history.csv", all_hist)
    write_csv(OUT / "pooling_baselines_mean_std.csv", summary)
    payload = {
        "out_dir": str(OUT),
        "caches": {f"{b}:{s}": str(p) for (b, s), p in CACHES.items()},
        "methods": METHODS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "bs": BS,
        "lr": LR,
        "device": str(DEVICE),
        "elapsed_sec": float(time.time() - started),
    }
    (OUT / "pooling_baselines_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# PGRF Pooling Baselines",
        "",
        f"Device: {DEVICE}; seeds: {SEEDS}; methods: {METHODS}",
        "",
        "| Backbone | Method | Calibration | AUROC | AUPRC | Brier鈫?| ECE鈫?|",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["calibration"] != "platt":
            continue
        lines.append(
            f"| {r['backbone']} | {r['method']} | {r['calibration']} | "
            f"{r['macro_auroc_mean']:.4f}卤{r['macro_auroc_std']:.4f} | "
            f"{r['macro_auprc_mean']:.4f}卤{r['macro_auprc_std']:.4f} | "
            f"{r['mean_brier_mean']:.4f}卤{r['mean_brier_std']:.4f} | "
            f"{r['mean_ece_mean']:.4f}卤{r['mean_ece_std']:.4f} |"
        )
    (OUT / "PGRF_Pooling_Baselines_Report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("[DONE]", OUT, "elapsed_sec", time.time() - started, flush=True)


if __name__ == "__main__":
    main()


