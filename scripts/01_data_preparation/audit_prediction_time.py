#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
MANIFEST = BASE / "pgrf_temporal_split_v1" / "selected_admissions_temporal_split.csv"
OUT = BASE / "pgrf_longitudinal_count_ge2_v1" / "prediction_time_audit_v1"
OUT.mkdir(parents=True, exist_ok=True)


def dt(x):
    x = (x or "").strip()
    if not x:
        return None
    try:
        return datetime.fromisoformat(x)
    except Exception:
        return None


def as_bool(x):
    s = str(x or "").strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    return False


def q(v):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if len(a) == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(len(a)), "median": float(med), "q1": float(q1), "q3": float(q3), "mean": float(a.mean())}


def fmt_q(d, digits=1):
    if not d["n"]:
        return "NA"
    return f"{d['median']:.{digits}f} [{d['q1']:.{digits}f}-{d['q3']:.{digits}f}]"


rows = []
with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        if int(float(row.get("history_ecg_count_including_index") or 0)) < 2:
            continue
        split = row.get("temporal_split") or row.get("split")
        if split not in {"train", "val", "test"}:
            continue
        index_time = dt(row.get("index_time"))
        admittime = dt(row.get("admittime"))
        dischtime = dt(row.get("dischtime"))
        deathtime = dt(row.get("deathtime"))
        dod = dt(row.get("dod"))
        known_death = dt(row.get("known_death_time")) or deathtime or dod
        def days_after(a, b):
            if a is None or b is None:
                return None
            return (a - b).total_seconds() / 86400.0
        rows.append({
            "split": split,
            "subject_id": row.get("subject_id"),
            "hadm_id": row.get("hadm_id"),
            "index_time": index_time,
            "admittime": admittime,
            "dischtime": dischtime,
            "known_death_time": known_death,
            "death_before_or_at_index_col": as_bool(row.get("death_before_or_at_index")),
            "candidate_valid_for_prediction": as_bool(row.get("candidate_valid_for_prediction")),
            "in_hospital": int(float(row.get("in_hospital_mortality") or 0)),
            "m30": int(float(row.get("mortality_30d") or 0)),
            "m1y": int(float(row.get("mortality_1y") or 0)),
            "index_after_admission_days": days_after(index_time, admittime),
            "discharge_after_index_days": days_after(dischtime, index_time),
            "death_after_index_days": days_after(known_death, index_time),
        })


def summarize(group_rows):
    n = len(group_rows)
    leaks_col = sum(x["death_before_or_at_index_col"] for x in group_rows)
    invalid = sum(not x["candidate_valid_for_prediction"] for x in group_rows)
    death_time_le_index = sum(
        1 for x in group_rows
        if x["known_death_time"] is not None and x["index_time"] is not None and x["known_death_time"] <= x["index_time"]
    )
    inh = [x for x in group_rows if x["in_hospital"] == 1]
    m30 = [x for x in group_rows if x["m30"] == 1]
    m1y = [x for x in group_rows if x["m1y"] == 1]
    out = {
        "n": n,
        "death_before_or_at_index_column_count": int(leaks_col),
        "candidate_invalid_count": int(invalid),
        "known_death_time_le_index_time_count": int(death_time_le_index),
        "index_after_admission_days": q([x["index_after_admission_days"] for x in group_rows]),
        "discharge_after_index_days": q([x["discharge_after_index_days"] for x in group_rows]),
        "in_hospital_positive_n": len(inh),
        "in_hospital_death_after_index_days": q([x["death_after_index_days"] for x in inh]),
        "mortality_30d_positive_n": len(m30),
        "mortality_30d_death_after_index_days": q([x["death_after_index_days"] for x in m30]),
        "mortality_1y_positive_n": len(m1y),
        "mortality_1y_death_after_index_days": q([x["death_after_index_days"] for x in m1y]),
    }
    # Near-index buckets for in-hospital deaths
    vals = [x["death_after_index_days"] for x in inh if x["death_after_index_days"] is not None]
    for cut in [0, 1, 3, 7, 14, 30]:
        out[f"in_hospital_death_within_{cut}d_n"] = int(sum(0 <= v <= cut for v in vals))
        out[f"in_hospital_death_within_{cut}d_pct_of_inh"] = float(100 * out[f"in_hospital_death_within_{cut}d_n"] / len(inh)) if inh else None
    return out


summary = {"overall": summarize(rows)}
for split in ["train", "val", "test"]:
    summary[split] = summarize([x for x in rows if x["split"] == split])

(OUT / "prediction_time_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

with open(OUT / "prediction_time_audit_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    fields = [
        "split", "n", "death_before_or_at_index_column_count", "candidate_invalid_count",
        "known_death_time_le_index_time_count", "index_after_admission_days_median_iqr",
        "discharge_after_index_days_median_iqr", "in_hospital_positive_n",
        "in_hospital_death_after_index_days_median_iqr",
        "in_hospital_death_within_1d_n", "in_hospital_death_within_1d_pct",
        "in_hospital_death_within_3d_n", "in_hospital_death_within_3d_pct",
        "in_hospital_death_within_7d_n", "in_hospital_death_within_7d_pct",
        "mortality_30d_positive_n", "mortality_30d_death_after_index_days_median_iqr",
        "mortality_1y_positive_n", "mortality_1y_death_after_index_days_median_iqr",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for split, s in summary.items():
        w.writerow({
            "split": split,
            "n": s["n"],
            "death_before_or_at_index_column_count": s["death_before_or_at_index_column_count"],
            "candidate_invalid_count": s["candidate_invalid_count"],
            "known_death_time_le_index_time_count": s["known_death_time_le_index_time_count"],
            "index_after_admission_days_median_iqr": fmt_q(s["index_after_admission_days"]),
            "discharge_after_index_days_median_iqr": fmt_q(s["discharge_after_index_days"]),
            "in_hospital_positive_n": s["in_hospital_positive_n"],
            "in_hospital_death_after_index_days_median_iqr": fmt_q(s["in_hospital_death_after_index_days"]),
            "in_hospital_death_within_1d_n": s["in_hospital_death_within_1d_n"],
            "in_hospital_death_within_1d_pct": f"{s['in_hospital_death_within_1d_pct_of_inh']:.1f}%" if s["in_hospital_death_within_1d_pct_of_inh"] is not None else "NA",
            "in_hospital_death_within_3d_n": s["in_hospital_death_within_3d_n"],
            "in_hospital_death_within_3d_pct": f"{s['in_hospital_death_within_3d_pct_of_inh']:.1f}%" if s["in_hospital_death_within_3d_pct_of_inh"] is not None else "NA",
            "in_hospital_death_within_7d_n": s["in_hospital_death_within_7d_n"],
            "in_hospital_death_within_7d_pct": f"{s['in_hospital_death_within_7d_pct_of_inh']:.1f}%" if s["in_hospital_death_within_7d_pct_of_inh"] is not None else "NA",
            "mortality_30d_positive_n": s["mortality_30d_positive_n"],
            "mortality_30d_death_after_index_days_median_iqr": fmt_q(s["mortality_30d_death_after_index_days"]),
            "mortality_1y_positive_n": s["mortality_1y_positive_n"],
            "mortality_1y_death_after_index_days_median_iqr": fmt_q(s["mortality_1y_death_after_index_days"]),
        })

md = [
    "# Prediction-time audit",
    "",
    "| Split | N | Death before/at index | Invalid | Index after admission, d | In-hospital deaths | Death after index, d | Death 鈮?d | Death 鈮?d | Death 鈮?d |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for split, s in summary.items():
    md.append(
        f"| {split} | {s['n']} | {s['death_before_or_at_index_column_count']} | {s['candidate_invalid_count']} | "
        f"{fmt_q(s['index_after_admission_days'])} | {s['in_hospital_positive_n']} | "
        f"{fmt_q(s['in_hospital_death_after_index_days'])} | "
        f"{s['in_hospital_death_within_1d_n']} ({s['in_hospital_death_within_1d_pct_of_inh']:.1f}%) | "
        f"{s['in_hospital_death_within_3d_n']} ({s['in_hospital_death_within_3d_pct_of_inh']:.1f}%) | "
        f"{s['in_hospital_death_within_7d_n']} ({s['in_hospital_death_within_7d_pct_of_inh']:.1f}%) |"
    )
(OUT / "Prediction_Time_Audit_Report.md").write_text("\n".join(md), encoding="utf-8")

print("\n".join(md))
print("WROTE", OUT)


