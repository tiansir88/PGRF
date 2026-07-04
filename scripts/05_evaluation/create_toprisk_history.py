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

BACKBONES = ["ST-MEM", "MERL", "ECG-FM", "MELP", "CLEAR-HUG"]
SEEDS = [42, 123, 1024]
LABELS = ["in_hospital_mortality", "mortality_30d", "mortality_1y"]


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


def macro_auprc(Y, P, mask=None):
    if mask is not None:
        Y = Y[mask]
        P = P[mask]
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
    raise ValueError(method)


def top_metrics(Y, P, frac=0.05):
    k = max(1, int(np.ceil(len(Y) * frac)))
    prec, captured = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j].astype(int)
        p = P[:, j]
        order = np.argsort(-p, kind="mergesort")[:k]
        hits = int(y[order].sum())
        total_pos = int(y.sum())
        prec.append(hits / k)
        captured.append(hits / total_pos if total_pos else np.nan)
    return float(np.nanmean(prec)), float(np.nanmean(captured))


top_rows, hist_rows = [], []
cache = np.load(STAGE / "caches" / "st_mem_temporal_count_ge2.npz", allow_pickle=True)
seq_len_all = cache["mask"].astype(bool).sum(axis=1)
split = cache["split"].astype(str)
test_seq_len = seq_len_all[split == "test"]

for bb in BACKBONES:
    Y, Plast = load_ensemble(paths_for(bb, "Last ECG"))
    _, Ppgrf = load_ensemble(paths_for(bb, "PGRF"))

    lp, lc = top_metrics(Y, Plast, frac=0.05)
    pp, pc = top_metrics(Y, Ppgrf, frac=0.05)
    top_rows.append({
        "backbone": bb,
        "top_frac": 0.05,
        "last_precision": lp,
        "pgrf_precision": pp,
        "delta_precision": pp - lp,
        "last_captured_events": lc,
        "pgrf_captured_events": pc,
        "delta_captured_events": pc - lc,
    })

    for th in [2, 5, 10]:
        m = test_seq_len >= th
        la = macro_auprc(Y, Plast, mask=m)
        pa = macro_auprc(Y, Ppgrf, mask=m)
        hist_rows.append({
            "backbone": bb,
            "subgroup": f"ECG count >= {th}",
            "threshold": th,
            "n_test": int(m.sum()),
            "events_in_hospital": int(Y[m, 0].sum()),
            "events_30d": int(Y[m, 1].sum()),
            "events_1y": int(Y[m, 2].sum()),
            "last_ecg_auprc": la,
            "pgrf_auprc": pa,
            "delta_auprc": pa - la,
        })

with open(OUT / "pgrf_final_ensemble_platt_toprisk_top5.csv", "w", encoding="utf-8-sig", newline="") as f:
    fields = ["backbone", "top_frac", "last_precision", "pgrf_precision", "delta_precision", "last_captured_events", "pgrf_captured_events", "delta_captured_events"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(top_rows)

with open(OUT / "pgrf_final_ensemble_platt_history_subgroup.csv", "w", encoding="utf-8-sig", newline="") as f:
    fields = ["backbone", "subgroup", "threshold", "n_test", "events_in_hospital", "events_30d", "events_1y", "last_ecg_auprc", "pgrf_auprc", "delta_auprc"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(hist_rows)

print("TOPRISK")
for r in top_rows:
    print(r)
print("HISTORY")
for r in hist_rows:
    print(r)
print("WROTE", OUT)


