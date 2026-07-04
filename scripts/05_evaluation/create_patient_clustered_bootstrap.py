from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
STAGE = BASE / "pgrf_longitudinal_count_ge2_v1"
OUT = STAGE / "patient_clustered_bootstrap_v1"
OUT.mkdir(parents=True, exist_ok=True)

BACKBONES = ["ST-MEM", "MERL", "ECG-FM", "MELP", "CLEAR-HUG"]
SEEDS = [42, 123, 1024]
LABELS = [
    ("in_hospital_mortality", "In-hospital"),
    ("mortality_30d", "30-day"),
    ("mortality_1y", "1-year"),
]
N_BOOT = int(os.environ.get("PGRF_N_BOOT", "1000"))


def auprc(y, p):
    y = np.asarray(y).astype(np.int8)
    p = np.asarray(p).astype(np.float64)
    npos = int(y.sum())
    if npos == 0:
        return np.nan
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / (np.arange(len(y)) + 1.0)
    return float(precision[ys == 1].sum() / npos)


def auroc(y, p):
    y = np.asarray(y).astype(np.int8)
    p = np.asarray(p).astype(np.float64)
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
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def macro_auprc(Y, P, idx=None):
    if idx is None:
        return float(np.nanmean([auprc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))
    return float(np.nanmean([auprc(Y[idx, j], P[idx, j]) for j in range(Y.shape[1])]))


def load_ensemble(paths, key="p_test_platt"):
    ys, ps = [], []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        ykey = "y_test" if "y_test" in data.files else "y_true"
        ys.append(data[ykey].astype("float32"))
        ps.append(data[key].astype("float32"))
    for y in ys[1:]:
        if not np.array_equal(ys[0], y):
            raise RuntimeError(f"Y mismatch for {paths[0]}")
    return ys[0], np.mean(ps, axis=0)


def paths_for(backbone, method):
    if method == "Last ECG":
        return [
            STAGE / "last_ecg_calibrated_v1" / backbone / "temporal" / "last_ecg" / f"seed{seed}_val_test_probs.npz"
            for seed in SEEDS
        ]
    if method == "PGRF":
        group = "trajectory_models_groupA" if backbone in {"ST-MEM", "ECG-FM", "CLEAR-HUG"} else "trajectory_models_groupB"
        return [
            STAGE / group / backbone / "temporal" / "fusion" / f"seed{seed}_val_test_probs.npz"
            for seed in SEEDS
        ]
    raise KeyError(method)


def get_test_subject_ids():
    # The longitudinal-eligible cohort is obtained by applying mask.sum>=2 to
    # the original temporal backbone sequence cache; the selected-admission CSV
    # preserves the same row order.
    orig_cache = BASE / "pgrf_backbone_sequence_cache_v1" / "stmem" / "pgrf_stmem_sequence_cache_temporal.npz"
    manifest = BASE / "pgrf_backbone_sequence_cache_v1" / "stmem" / "selected_admissions_pgrf.csv"
    data = np.load(orig_cache, allow_pickle=True)
    keep = data["mask"].astype(bool).sum(axis=1) >= 2
    split = data["split"].astype(str)
    test_keep = keep & (split == "test")

    subjects = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subjects.append(row["subject_id"])
    subjects = np.asarray(subjects, dtype=object)
    if len(subjects) != len(keep):
        raise RuntimeError(f"Manifest/cache length mismatch: {len(subjects)} vs {len(keep)}")
    return subjects[test_keep]


def grouped_indices(subject_ids):
    order = {}
    for i, subject_id in enumerate(subject_ids):
        order.setdefault(str(subject_id), []).append(i)
    patients = np.array(list(order.keys()), dtype=object)
    groups = [np.asarray(order[subject_id], dtype=np.int64) for subject_id in patients]
    return patients, groups


def patient_macro_bootstrap(Y, p_base, p_method, subject_ids, seed):
    rng = np.random.default_rng(seed)
    patients, groups = grouped_indices(subject_ids)
    values = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        sampled = rng.integers(0, len(patients), size=len(patients))
        idx = np.concatenate([groups[i] for i in sampled])
        values[b] = macro_auprc(Y, p_method, idx) - macro_auprc(Y, p_base, idx)
    observed = macro_auprc(Y, p_method) - macro_auprc(Y, p_base)
    return observed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), float(np.mean(values <= 0)), len(patients)


def patient_endpoint_bootstrap(y, p_base, p_method, subject_ids, seed):
    rng = np.random.default_rng(seed)
    patients, groups = grouped_indices(subject_ids)
    values = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        sampled = rng.integers(0, len(patients), size=len(patients))
        idx = np.concatenate([groups[i] for i in sampled])
        values[b] = auprc(y[idx], p_method[idx]) - auprc(y[idx], p_base[idx])
    observed = auprc(y, p_method) - auprc(y, p_base)
    return observed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), float(np.mean(values <= 0)), len(patients)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    subject_ids = get_test_subject_ids()
    macro_rows, endpoint_rows = [], []

    for backbone_index, backbone in enumerate(BACKBONES):
        Y_last, P_last = load_ensemble(paths_for(backbone, "Last ECG"))
        Y_pgrf, P_pgrf = load_ensemble(paths_for(backbone, "PGRF"))
        if not np.array_equal(Y_last, Y_pgrf):
            raise RuntimeError(f"Y mismatch for {backbone}")
        if len(Y_last) != len(subject_ids):
            raise RuntimeError(f"N mismatch for {backbone}: {len(Y_last)} vs {len(subject_ids)}")

        delta, lo, hi, pval, n_patients = patient_macro_bootstrap(
            Y_last, P_last, P_pgrf, subject_ids, seed=20260704 + 101 * backbone_index
        )
        macro_rows.append({
            "backbone": backbone,
            "comparison": "PGRF vs Last ECG",
            "metric": "macro_AUPRC",
            "last_ecg": macro_auprc(Y_last, P_last),
            "pgrf": macro_auprc(Y_last, P_pgrf),
            "delta": delta,
            "ci_low": lo,
            "ci_high": hi,
            "p_delta_le_0": pval,
            "n_test_admissions": int(len(subject_ids)),
            "n_test_patients": int(n_patients),
            "n_boot": N_BOOT,
        })

        for j, (label, pretty) in enumerate(LABELS):
            y = Y_last[:, j]
            delta_e, lo_e, hi_e, pval_e, n_patients_e = patient_endpoint_bootstrap(
                y, P_last[:, j], P_pgrf[:, j], subject_ids, seed=20260704 + 1000 * backbone_index + j
            )
            endpoint_rows.append({
                "backbone": backbone,
                "endpoint": label,
                "endpoint_pretty": pretty,
                "last_auroc": auroc(y, P_last[:, j]),
                "pgrf_auroc": auroc(y, P_pgrf[:, j]),
                "delta_auroc": auroc(y, P_pgrf[:, j]) - auroc(y, P_last[:, j]),
                "last_auprc": auprc(y, P_last[:, j]),
                "pgrf_auprc": auprc(y, P_pgrf[:, j]),
                "delta_auprc": delta_e,
                "delta_auprc_ci_low": lo_e,
                "delta_auprc_ci_high": hi_e,
                "p_delta_auprc_le_0": pval_e,
                "prevalence": float(y.mean()),
                "positives": int(y.sum()),
                "n_test_admissions": int(len(y)),
                "n_test_patients": int(n_patients_e),
                "n_boot": N_BOOT,
            })

    write_csv(OUT / "patient_clustered_macro_bootstrap.csv", macro_rows, [
        "backbone", "comparison", "metric", "last_ecg", "pgrf", "delta", "ci_low", "ci_high",
        "p_delta_le_0", "n_test_admissions", "n_test_patients", "n_boot",
    ])
    write_csv(OUT / "patient_clustered_endpoint_bootstrap.csv", endpoint_rows, [
        "backbone", "endpoint", "endpoint_pretty", "last_auroc", "pgrf_auroc", "delta_auroc",
        "last_auprc", "pgrf_auprc", "delta_auprc", "delta_auprc_ci_low", "delta_auprc_ci_high",
        "p_delta_auprc_le_0", "prevalence", "positives", "n_test_admissions", "n_test_patients", "n_boot",
    ])

    print("WROTE", OUT)


if __name__ == "__main__":
    main()
