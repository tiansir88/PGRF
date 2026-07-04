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


def auroc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    npos = int(y.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and p[order[j]] == p[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


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


def ece(y, p, bins=15):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)
        if m.any():
            out += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(out)


def metrics(Y, P):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    per = []
    for j, lab in enumerate(LABELS):
        y = Y[:, j]
        p = P[:, j]
        per.append({
            "label": lab,
            "auroc": auroc(y, p),
            "auprc": auprc(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": ece(y, p),
            "prevalence": float(y.mean()),
            "positives": int(y.sum()),
        })
    return {
        "macro_auroc": float(np.nanmean([x["auroc"] for x in per])),
        "macro_auprc": float(np.nanmean([x["auprc"] for x in per])),
        "mean_brier": float(np.nanmean([x["brier"] for x in per])),
        "mean_ece": float(np.nanmean([x["ece"] for x in per])),
        "per_label": per,
    }


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


def safe_backbone(bb):
    return bb


rows, per_rows = [], []
for bb in BACKBONES:
    specs = []
    specs.append(("Last ECG", [STAGE / "last_ecg_calibrated_v1" / bb / "temporal" / "last_ecg" / f"seed{s}_val_test_probs.npz" for s in SEEDS]))
    for method, pretty in [
        ("mean_pool", "Mean pooling"),
        ("time_decay_pool", "Time-decay pooling"),
        ("attention_pool", "Attention pooling"),
    ]:
        specs.append((pretty, [STAGE / "pooling_baselines_v1" / bb / "temporal" / method / f"seed{s}_val_test_probs.npz" for s in SEEDS]))
    group = "trajectory_models_groupA" if bb in {"ST-MEM", "ECG-FM", "CLEAR-HUG"} else "trajectory_models_groupB"
    specs.append(("GRU trajectory", [STAGE / group / bb / "temporal" / "gru" / f"seed{s}_val_test_probs.npz" for s in SEEDS]))
    specs.append(("PGRF", [STAGE / group / bb / "temporal" / "fusion" / f"seed{s}_val_test_probs.npz" for s in SEEDS]))

    for method, paths in specs:
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing for {bb} {method}: {missing}")
        Y, P = load_ensemble(paths, key="p_test_platt")
        mm = metrics(Y, P)
        rows.append({
            "backbone": bb,
            "method": method,
            "calibration": "platt",
            "aggregation": "3-seed probability ensemble",
            **{k: v for k, v in mm.items() if k != "per_label"},
        })
        for r in mm["per_label"]:
            per_rows.append({"backbone": bb, "method": method, **r})

with open(OUT / "pgrf_final_ensemble_platt_main_table.csv", "w", encoding="utf-8-sig", newline="") as f:
    fields = ["backbone", "method", "calibration", "aggregation", "macro_auroc", "macro_auprc", "mean_brier", "mean_ece"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

with open(OUT / "pgrf_final_ensemble_platt_per_endpoint.csv", "w", encoding="utf-8-sig", newline="") as f:
    fields = ["backbone", "method", "label", "auroc", "auprc", "brier", "ece", "prevalence", "positives"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(per_rows)

lines = [
    "| Backbone | Method | Macro-AUROC | Macro-AUPRC | Cal. Brier鈫?| Cal. ECE鈫?|",
    "|---|---|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(f"| {r['backbone']} | {r['method']} | {r['macro_auroc']:.4f} | {r['macro_auprc']:.4f} | {r['mean_brier']:.4f} | {r['mean_ece']:.4f} |")
(OUT / "PGRF_Final_Ensemble_Platt_Main_Table.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("WROTE", OUT)


