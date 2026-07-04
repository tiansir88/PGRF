#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
STAGE = BASE / "pgrf_longitudinal_count_ge2_v1"
OUT = STAGE / "per_endpoint_last_vs_pgrf_v1"
OUT.mkdir(parents=True, exist_ok=True)

BACKBONES = ["ST-MEM", "MERL", "ECG-FM", "MELP", "CLEAR-HUG"]
SEEDS = [42, 123, 1024]
LABELS = [
    ("in_hospital_mortality", "In-hospital"),
    ("mortality_30d", "30-day"),
    ("mortality_1y", "1-year"),
]


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


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins=15):
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            out += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(out)


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
    raise KeyError(method)


def bootstrap_delta(y, p_base, p_method, fn, n_boot=500, seed=20260702):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = fn(y[idx], p_method[idx]) - fn(y[idx], p_base[idx])
    return {
        "delta": float(fn(y, p_method) - fn(y, p_base)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "p_delta_le_0": float(np.mean(vals <= 0)),
        "n_boot": n_boot,
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    metric_rows = []
    delta_rows = []
    compact_rows = []

    for bb in BACKBONES:
        y_last, p_last = load_ensemble(paths_for(bb, "Last ECG"))
        y_pgrf, p_pgrf = load_ensemble(paths_for(bb, "PGRF"))
        if not np.array_equal(y_last, y_pgrf):
            raise RuntimeError(f"Y mismatch for {bb}")

        for method, P in [("Last ECG", p_last), ("PGRF", p_pgrf)]:
            aurocs, auprcs = [], []
            for j, (label, label_pretty) in enumerate(LABELS):
                y = y_last[:, j]
                p = P[:, j]
                r = {
                    "backbone": bb,
                    "method": method,
                    "endpoint": label,
                    "endpoint_pretty": label_pretty,
                    "auroc": auroc(y, p),
                    "auprc": auprc(y, p),
                    "brier": brier(y, p),
                    "ece": ece(y, p),
                    "prevalence": float(y.mean()),
                    "positives": int(y.sum()),
                    "n_test": int(len(y)),
                }
                metric_rows.append(r)
                aurocs.append(r["auroc"])
                auprcs.append(r["auprc"])
            compact_rows.append({
                "backbone": bb,
                "method": method,
                "in_hospital_auroc": metric_rows[-3]["auroc"],
                "in_hospital_auprc": metric_rows[-3]["auprc"],
                "mortality_30d_auroc": metric_rows[-2]["auroc"],
                "mortality_30d_auprc": metric_rows[-2]["auprc"],
                "mortality_1y_auroc": metric_rows[-1]["auroc"],
                "mortality_1y_auprc": metric_rows[-1]["auprc"],
                "macro_auroc": float(np.nanmean(aurocs)),
                "macro_auprc": float(np.nanmean(auprcs)),
            })

        for j, (label, label_pretty) in enumerate(LABELS):
            y = y_last[:, j]
            dr_ap = bootstrap_delta(y, p_last[:, j], p_pgrf[:, j], auprc, seed=20260702 + j + 17 * BACKBONES.index(bb))
            delta_auc = auroc(y, p_pgrf[:, j]) - auroc(y, p_last[:, j])
            delta_rows.append({
                "backbone": bb,
                "endpoint": label,
                "endpoint_pretty": label_pretty,
                "last_auroc": auroc(y, p_last[:, j]),
                "pgrf_auroc": auroc(y, p_pgrf[:, j]),
                "delta_auroc": delta_auc,
                "last_auprc": auprc(y, p_last[:, j]),
                "pgrf_auprc": auprc(y, p_pgrf[:, j]),
                "delta_auprc": dr_ap["delta"],
                "delta_auprc_ci_low": dr_ap["ci_low"],
                "delta_auprc_ci_high": dr_ap["ci_high"],
                "p_delta_auprc_le_0": dr_ap["p_delta_le_0"],
                "prevalence": float(y.mean()),
                "positives": int(y.sum()),
                "n_test": int(len(y)),
                "n_boot": dr_ap["n_boot"],
            })

    write_csv(OUT / "per_endpoint_last_pgrf_metrics.csv", metric_rows, [
        "backbone", "method", "endpoint", "endpoint_pretty", "auroc", "auprc",
        "brier", "ece", "prevalence", "positives", "n_test",
    ])
    write_csv(OUT / "per_endpoint_last_pgrf_compact.csv", compact_rows, [
        "backbone", "method",
        "in_hospital_auroc", "in_hospital_auprc",
        "mortality_30d_auroc", "mortality_30d_auprc",
        "mortality_1y_auroc", "mortality_1y_auprc",
        "macro_auroc", "macro_auprc",
    ])
    write_csv(OUT / "per_endpoint_pgrf_vs_last_bootstrap.csv", delta_rows, [
        "backbone", "endpoint", "endpoint_pretty",
        "last_auroc", "pgrf_auroc", "delta_auroc",
        "last_auprc", "pgrf_auprc", "delta_auprc", "delta_auprc_ci_low", "delta_auprc_ci_high", "p_delta_auprc_le_0",
        "prevalence", "positives", "n_test", "n_boot",
    ])

    # Paper-readable Markdown.
    lines = [
        "| Backbone | Method | In-hospital AUROC/AUPRC | 30-day AUROC/AUPRC | 1-year AUROC/AUPRC | Macro AUROC/AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in compact_rows:
        lines.append(
            f"| {r['backbone']} | {r['method']} | "
            f"{r['in_hospital_auroc']:.4f}/{r['in_hospital_auprc']:.4f} | "
            f"{r['mortality_30d_auroc']:.4f}/{r['mortality_30d_auprc']:.4f} | "
            f"{r['mortality_1y_auroc']:.4f}/{r['mortality_1y_auprc']:.4f} | "
            f"{r['macro_auroc']:.4f}/{r['macro_auprc']:.4f} |"
        )
    lines += [
        "",
        "| Backbone | Endpoint | Delta AUPRC | 95% CI | p(delta<=0) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in delta_rows:
        lines.append(
            f"| {r['backbone']} | {r['endpoint_pretty']} | "
            f"{r['delta_auprc']:.4f} | "
            f"[{r['delta_auprc_ci_low']:.4f}, {r['delta_auprc_ci_high']:.4f}] | "
            f"{r['p_delta_auprc_le_0']:.4f} |"
        )
    (OUT / "Per_Endpoint_Last_vs_PGRF_Report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()


