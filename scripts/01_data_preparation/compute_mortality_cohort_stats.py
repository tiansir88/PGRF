#!/usr/bin/env python3
"""
Compute paper-ready cohort statistics for the PGRF count>=2
serial-ECG mortality prediction cohort.

This script intentionally avoids pandas so it can run on minimal remote
environments. It uses:
  - PGRF temporal temporal split manifest for admission/patient/outcome counts.
  - PGRF count>=2 cache for actual max_len=10 sequence length/span.
  - ECG record manifest for unique ECG records used by the sequences.
"""

from __future__ import annotations

import csv
import json
import math
import os
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


BASE = Path(os.environ.get("PGRF_PROJECT_DIR", "/path/to/pgrf_project"))
TRAINING = BASE / "training_manifest"
STAGE4_MANIFEST = (
    TRAINING
    / "pgrf_temporal_split_v1"
    / "selected_admissions_temporal_split.csv"
)
PGRF_CACHE = (
    TRAINING
    / "pgrf_longitudinal_count_ge2_v1"
    / "caches"
    / "st_mem_temporal_count_ge2.npz"
)
RECORD_MANIFEST_CANDIDATES = [
    BASE / "mimic_ecg_record_manifest.csv",
    Path(os.environ.get("PGRF_MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg")) / "record_list.csv",
]
OUT_DIR = TRAINING / "outputs" / "paper_ready_tables_v1"

LABELS = ["in_hospital_mortality", "mortality_30d", "mortality_1y"]
SPLITS = ["overall", "train", "val", "test"]
MAX_LEN = 10


def parse_dt(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # MIMIC timestamps here are ISO-like: YYYY-MM-DD HH:MM:SS.
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def to_int(s, default=0) -> int:
    try:
        if s is None or s == "":
            return default
        return int(float(str(s)))
    except Exception:
        return default


def truthy_int(s) -> int:
    if isinstance(s, str):
        s2 = s.strip().lower()
        if s2 in {"true", "t", "yes", "y"}:
            return 1
        if s2 in {"false", "f", "no", "n", ""}:
            return 0
    return 1 if to_int(s, 0) != 0 else 0


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def median_iqr(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(q1), float(q3)


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def fmt_median_iqr(v: Tuple[float, float, float], decimals: int = 1) -> str:
    med, q1, q3 = v
    if any(math.isnan(z) for z in [med, q1, q3]):
        return "NA"
    if decimals == 0:
        return f"{med:.0f} [{q1:.0f}-{q3:.0f}]"
    return f"{med:.{decimals}f} [{q1:.{decimals}f}-{q3:.{decimals}f}]"


def fmt_event(n: int, d: int) -> str:
    return f"{n:,} ({pct(n, d):.1f}%)"


def read_pgrf_temporal_manifest() -> List[dict]:
    rows: List[dict] = []
    with open(STAGE4_MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if to_int(r.get("history_ecg_count_including_index"), 0) < 2:
                continue
            split = (r.get("temporal_split") or r.get("split") or "").strip()
            if split not in {"train", "val", "test"}:
                continue
            rows.append(
                {
                    "subject_id": str(r.get("subject_id", "")).strip(),
                    "hadm_id": str(r.get("hadm_id", "")).strip(),
                    "index_time": parse_dt(r.get("index_time")),
                    "index_path": str(r.get("index_path", "")).strip(),
                    "split": split,
                    "history_ecg_count_including_index": to_int(
                        r.get("history_ecg_count_including_index"), 0
                    ),
                    "history_span_days_manifest": float(
                        r.get("history_span_days") or "nan"
                    ),
                    **{lab: truthy_int(r.get(lab)) for lab in LABELS},
                }
            )
    return rows


def load_cache_stats():
    data = np.load(PGRF_CACHE, allow_pickle=True)
    mask = data["mask"].astype(bool)
    dseq = data["Dseq"].astype(np.float32)
    y = data["Y"].astype(np.float32)
    split = data["split"].astype(str)
    seq_len = mask.sum(axis=1).astype(np.int32)
    masked_d = np.where(mask, dseq, np.nan)
    span = np.nanmax(masked_d, axis=1)
    return {
        "mask": mask,
        "dseq": dseq,
        "Y": y,
        "split": split,
        "seq_len": seq_len,
        "span": span,
    }


def find_record_manifest() -> Path | None:
    for p in RECORD_MANIFEST_CANDIDATES:
        if p.exists():
            return p
    return None


def load_records_for_subjects(subject_ids: set[str], record_manifest: Path):
    by_subject: Dict[str, List[Tuple[datetime, str, str]]] = defaultdict(list)
    with open(record_manifest, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # mimic_ecg_record_manifest has path; record_list usually has path too.
        if "subject_id" not in fieldnames or "ecg_time" not in fieldnames:
            raise RuntimeError(f"record manifest missing required columns: {fieldnames}")
        for r in reader:
            sid = str(r.get("subject_id", "")).strip()
            if sid not in subject_ids:
                continue
            t = parse_dt(r.get("ecg_time"))
            if t is None:
                continue
            study_id = str(r.get("study_id") or r.get("file_name") or "").strip()
            path = str(r.get("path") or study_id).strip()
            key = path if path else study_id
            if not key:
                continue
            by_subject[sid].append((t, study_id, key))
    for sid in list(by_subject.keys()):
        by_subject[sid].sort(key=lambda z: (z[0], z[1]))
    return by_subject


def reconstruct_unique_ecg_records(rows: List[dict], seq_len: np.ndarray | None = None):
    record_manifest = find_record_manifest()
    if record_manifest is None:
        return None, {"error": "No record manifest found"}

    subject_ids = {r["subject_id"] for r in rows}
    by_subject = load_records_for_subjects(subject_ids, record_manifest)

    unique_by_split: Dict[str, set[str]] = {s: set() for s in SPLITS}
    recon_lens: List[int] = []
    missing_subject = 0
    zero_history = 0
    len_mismatch = 0

    for i, r in enumerate(rows):
        split = r["split"]
        recs = by_subject.get(r["subject_id"])
        if not recs:
            missing_subject += 1
            continue
        idx_t = r["index_time"]
        if idx_t is None:
            zero_history += 1
            continue
        times = [x[0] for x in recs]
        end = bisect_right(times, idx_t)
        start = max(0, end - MAX_LEN)
        chosen = recs[start:end]
        if not chosen:
            zero_history += 1
            continue
        keys = [x[2] for x in chosen]
        recon_lens.append(len(keys))
        if seq_len is not None and i < len(seq_len) and int(seq_len[i]) != len(keys):
            len_mismatch += 1
        unique_by_split[split].update(keys)
        unique_by_split["overall"].update(keys)

    return (
        {s: len(unique_by_split[s]) for s in SPLITS},
        {
            "record_manifest": str(record_manifest),
            "subjects_requested": len(subject_ids),
            "subjects_with_records": len(by_subject),
            "missing_subject_rows": missing_subject,
            "zero_history_rows": zero_history,
            "reconstructed_rows": len(recon_lens),
            "sequence_length_mismatch_vs_cache": len_mismatch,
        },
    )


def summarize():
    rows = read_pgrf_temporal_manifest()
    cache = load_cache_stats()
    seq_len = cache["seq_len"]
    span = cache["span"]
    cache_split = cache["split"]
    y = cache["Y"]

    if len(rows) != len(seq_len):
        raise RuntimeError(f"row/cache length mismatch: manifest={len(rows)} cache={len(seq_len)}")

    manifest_split = np.array([r["split"] for r in rows]).astype(str)
    split_match_rate = float(np.mean(manifest_split == cache_split))
    label_arr = np.array([[r[lab] for lab in LABELS] for r in rows], dtype=np.float32)
    label_match_rate = float(np.mean(label_arr == y))

    unique_ecg_counts, unique_diag = reconstruct_unique_ecg_records(rows, seq_len=seq_len)

    out = {
        "source": {
            "pgrf_temporal_manifest": str(STAGE4_MANIFEST),
            "pgrf_cache": str(PGRF_CACHE),
            "record_manifest": unique_diag.get("record_manifest") if unique_diag else None,
            "max_len": MAX_LEN,
            "definition": (
                "Primary longitudinal cohort: admissions with at least two ECGs up to "
                "index time. Sequence statistics use the PGRF max_len=10 model inputs."
            ),
        },
        "diagnostics": {
            "n_manifest_rows": len(rows),
            "n_cache_rows": int(len(seq_len)),
            "split_match_rate_manifest_vs_cache": split_match_rate,
            "label_element_match_rate_manifest_vs_cache": label_match_rate,
            "unique_reconstruction": unique_diag,
        },
        "splits": {},
    }

    for s in SPLITS:
        if s == "overall":
            row_idx = list(range(len(rows)))
            cache_idx = np.arange(len(rows))
        else:
            row_idx = [i for i, r in enumerate(rows) if r["split"] == s]
            cache_idx = np.where(cache_split == s)[0]

        denom = len(row_idx)
        subjects = {rows[i]["subject_id"] for i in row_idx}
        admissions = {rows[i]["hadm_id"] for i in row_idx}
        events = {lab: int(sum(rows[i][lab] for i in row_idx)) for lab in LABELS}

        out["splits"][s] = {
            "patients": len(subjects),
            "admissions": len(admissions),
            "ecg_records_unique": (
                int(unique_ecg_counts[s]) if unique_ecg_counts is not None else None
            ),
            "ecg_sequence_tokens": int(seq_len[cache_idx].sum()),
            "ecgs_per_admission_median_iqr": median_iqr(seq_len[cache_idx]),
            "history_span_days_median_iqr": median_iqr(span[cache_idx]),
            "events": events,
            "event_rates_percent": {lab: pct(events[lab], denom) for lab in LABELS},
        }

    return out


def write_outputs(stats: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "serial_ecg_mortality_prediction_cohort_stats.json"
    csv_path = OUT_DIR / "serial_ecg_mortality_prediction_cohort_stats.csv"
    md_path = OUT_DIR / "serial_ecg_mortality_prediction_cohort_stats.md"
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    chars = [
        ("Patients", lambda s: fmt_int(stats["splits"][s]["patients"])),
        ("Admissions", lambda s: fmt_int(stats["splits"][s]["admissions"])),
        (
            "ECG records",
            lambda s: fmt_int(stats["splits"][s]["ecg_records_unique"])
            if stats["splits"][s]["ecg_records_unique"] is not None
            else "NA",
        ),
        (
            "ECGs per admission, median [IQR]",
            lambda s: fmt_median_iqr(
                tuple(stats["splits"][s]["ecgs_per_admission_median_iqr"]), decimals=0
            ),
        ),
        (
            "History span, days, median [IQR]",
            lambda s: fmt_median_iqr(
                tuple(stats["splits"][s]["history_span_days_median_iqr"]), decimals=1
            ),
        ),
        (
            "In-hospital mortality, n (%)",
            lambda s: fmt_event(
                stats["splits"][s]["events"]["in_hospital_mortality"],
                stats["splits"][s]["admissions"],
            ),
        ),
        (
            "30-day mortality, n (%)",
            lambda s: fmt_event(
                stats["splits"][s]["events"]["mortality_30d"],
                stats["splits"][s]["admissions"],
            ),
        ),
        (
            "1-year mortality, n (%)",
            lambda s: fmt_event(
                stats["splits"][s]["events"]["mortality_1y"],
                stats["splits"][s]["admissions"],
            ),
        ),
    ]
    for name, fn in chars:
        rows.append(
            {
                "Characteristic": name,
                "Overall": fn("overall"),
                "Train": fn("train"),
                "Validation": fn("val"),
                "Test": fn("test"),
            }
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Characteristic", "Overall", "Train", "Validation", "Test"]
        )
        writer.writeheader()
        writer.writerows(rows)

    md = []
    md.append("| Characteristic | Overall | Train | Validation | Test |")
    md.append("|---|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['Characteristic']} | {r['Overall']} | {r['Train']} | "
            f"{r['Validation']} | {r['Test']} |"
        )
    md.append("")
    md.append(
        "Note: ECG records are unique ECG record paths used by at least one "
        "max_len=10 serial input sequence in the corresponding split. ECGs per "
        "admission and history span are computed from the final PGRF model "
        "input cache."
    )
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, csv_path, md_path


def main():
    stats = summarize()
    paths = write_outputs(stats)
    print(json.dumps(stats["diagnostics"], indent=2, ensure_ascii=False))
    print("WROTE")
    for p in paths:
        print(p)
    print((OUT_DIR / "serial_ecg_mortality_prediction_cohort_stats.md").read_text())


if __name__ == "__main__":
    main()


