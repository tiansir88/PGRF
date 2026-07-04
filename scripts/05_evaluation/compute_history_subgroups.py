from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
STAGE = BASE / "pgrf_longitudinal_count_ge2_v1"
PAPER = BASE / "outputs" / "paper_ready_tables_v1"
PAPER.mkdir(parents=True, exist_ok=True)

BACKBONES = ["ST-MEM", "ECG-FM", "CLEAR-HUG", "MERL", "MELP"]
SAFE = {b: b.lower().replace("-", "_") for b in BACKBONES}
GROUP = {
    "ST-MEM": "trajectory_models_groupA",
    "ECG-FM": "trajectory_models_groupA",
    "CLEAR-HUG": "trajectory_models_groupA",
    "MERL": "trajectory_models_groupB",
    "MELP": "trajectory_models_groupB",
}
SEEDS = [42, 123, 1024]
LABELS = ["in_hospital", "30d", "1y"]
THRESHOLDS = [2, 5, 10]


def average_precision_binary(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(np.float64)
    score = np.asarray(score).astype(np.float64)
    n_pos = int(y.sum())
    if n_pos <= 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    precision = tp / ranks
    return float((precision * y_sorted).sum() / n_pos)


def macro_auprc(y: np.ndarray, p: np.ndarray) -> float:
    vals = [average_precision_binary(y[:, j], p[:, j]) for j in range(y.shape[1])]
    return float(np.nanmean(vals))


def load_last_ensemble(backbone: str) -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for seed in SEEDS:
        path = STAGE / SAFE[backbone] / "temporal_prior_residual" / f"last_ecg_seed{seed}_test_probs.npz"
        z = np.load(path)
        ys.append(z["y_true"].astype(np.float64))
        ps.append(z["y_prob"].astype(np.float64))
    for y in ys[1:]:
        assert np.array_equal(ys[0], y), (backbone, "last y mismatch")
    return ys[0], np.mean(ps, axis=0)


def load_pgrf_ensemble(backbone: str) -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for seed in SEEDS:
        path = STAGE / GROUP[backbone] / backbone / "temporal" / "fusion" / f"seed{seed}_val_test_probs.npz"
        z = np.load(path)
        ys.append(z["y_test"].astype(np.float64))
        ps.append(z["p_test"].astype(np.float64))
    for y in ys[1:]:
        assert np.array_equal(ys[0], y), (backbone, "pgrf y mismatch")
    return ys[0], np.mean(ps, axis=0)


def load_test_counts(backbone: str, y_ref: np.ndarray) -> np.ndarray:
    cache = np.load(STAGE / "caches" / f"{SAFE[backbone]}_temporal_count_ge2.npz", allow_pickle=True)
    split = cache["split"]
    mask = cache["mask"].astype(bool)
    y = cache["Y"].astype(np.float64)
    test_idx = np.where(split == "test")[0]
    y_test = y[test_idx]
    assert y_test.shape == y_ref.shape, (backbone, y_test.shape, y_ref.shape)
    assert np.array_equal(y_test, y_ref), (backbone, "cache/test prediction order mismatch")
    return mask[test_idx].sum(axis=1).astype(int)


def main() -> None:
    rows = []
    for backbone in BACKBONES:
        y_last, p_last = load_last_ensemble(backbone)
        y_pgrf, p_pgrf = load_pgrf_ensemble(backbone)
        assert np.array_equal(y_last, y_pgrf), backbone
        counts = load_test_counts(backbone, y_last)
        for threshold in THRESHOLDS:
            keep = counts >= threshold
            n = int(keep.sum())
            events = y_last[keep].sum(axis=0)
            last_ap = macro_auprc(y_last[keep], p_last[keep])
            pgrf_ap = macro_auprc(y_pgrf[keep], p_pgrf[keep])
            rows.append(
                {
                    "backbone": backbone,
                    "subgroup": f"ECG count >= {threshold}",
                    "threshold": threshold,
                    "n_test": n,
                    "events_in_hospital": int(events[0]),
                    "events_30d": int(events[1]),
                    "events_1y": int(events[2]),
                    "last_ecg_auprc": last_ap,
                    "pgrf_auprc": pgrf_ap,
                    "delta_auprc": pgrf_ap - last_ap,
                }
            )
    out = PAPER / "pgrf_history_length_subgroup_auprc.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()


