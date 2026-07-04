#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
STAGE = BASE / "pgrf_longitudinal_count_ge2_v1"
OUT = STAGE / "paper_consistency_v1"
OUT.mkdir(parents=True, exist_ok=True)

BACKBONES = ["ST-MEM", "ECG-FM", "CLEAR-HUG", "MERL", "MELP"]
SEEDS = [42, 123, 1024]
LABELS = ["in_hospital_mortality", "mortality_30d", "mortality_1y"]
N_BOOT = 2000
RNG = np.random.default_rng(20260702)


def auprc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    npos = int(y.sum())
    if npos == 0:
        return np.nan
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / (np.arange(len(y)) + 1.0)
    return float(precision[ys == 1].sum() / npos)


def macro_auprc(Y, P, idx=None):
    if idx is not None:
        Y = Y[idx]
        P = P[idx]
    return float(np.nanmean([auprc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))


def load_ensemble(paths, key="p_test_platt"):
    ys, ps = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        ykey = "y_test" if "y_test" in d.files else "y_true"
        ys.append(d[ykey].astype("float32"))
        ps.append(d[key].astype("float32"))
    for y in ys[1:]:
        if not np.array_equal(ys[0], y):
            raise RuntimeError(f"Y mismatch for {paths[0]}")
    return ys[0], np.mean(ps, axis=0)


def paths_for(bb, method):
    if method == "Last ECG":
        return [STAGE / "last_ecg_calibrated_v1" / bb / "temporal" / "last_ecg" / f"seed{s}_val_test_probs.npz" for s in SEEDS]
    if method == "PGRF":
        group = "trajectory_models_groupA" if bb in {"ST-MEM", "ECG-FM", "CLEAR-HUG"} else "trajectory_models_groupB"
        return [STAGE / group / bb / "temporal" / "fusion" / f"seed{s}_val_test_probs.npz" for s in SEEDS]
    if method == "GRU trajectory":
        group = "trajectory_models_groupA" if bb in {"ST-MEM", "ECG-FM", "CLEAR-HUG"} else "trajectory_models_groupB"
        return [STAGE / group / bb / "temporal" / "gru" / f"seed{s}_val_test_probs.npz" for s in SEEDS]
    raise ValueError(method)


rows = []
for bb in BACKBONES:
    Y, P_pgrf = load_ensemble(paths_for(bb, "PGRF"))
    _, P_last = load_ensemble(paths_for(bb, "Last ECG"))
    _, P_gru = load_ensemble(paths_for(bb, "GRU trajectory"))
    for comp, base_p in [("PGRF vs Last ECG", P_last), ("PGRF vs GRU trajectory", P_gru)]:
        observed = macro_auprc(Y, P_pgrf) - macro_auprc(Y, base_p)
        boots = np.empty(N_BOOT, dtype=np.float64)
        n = len(Y)
        for b in range(N_BOOT):
            idx = RNG.integers(0, n, size=n)
            boots[b] = macro_auprc(Y, P_pgrf, idx) - macro_auprc(Y, base_p, idx)
        rows.append({
            "backbone": bb,
            "comparison": comp,
            "metric": "macro_AUPRC",
            "delta": float(observed),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "p_one_sided_delta_le_0": float((np.sum(boots <= 0) + 1) / (N_BOOT + 1)),
            "n_boot": N_BOOT,
            "n_test": int(n),
        })

with open(OUT / "pgrf_final_ensemble_platt_bootstrap_auprc.csv", "w", encoding="utf-8-sig", newline="") as f:
    fields = ["backbone", "comparison", "metric", "delta", "ci_low", "ci_high", "p_one_sided_delta_le_0", "n_boot", "n_test"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print("| Backbone | Comparison | 螖 Macro-AUPRC | 95% CI | p |")
print("|---|---|---:|---:|---:|")
for r in rows:
    print(f"| {r['backbone']} | {r['comparison']} | {r['delta']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['p_one_sided_delta_le_0']:.4f} |")
print("WROTE", OUT / "pgrf_final_ensemble_platt_bootstrap_auprc.csv")


