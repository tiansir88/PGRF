import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
ECG = Path(os.environ.get("PGRF_MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg"))
RUN = BASE / "selective_trajecg_stage2_full_cache_gated_probe_v1"
OUT = BASE / "selective_trajecg_stage3_final_evidence_pack_v1"
OUT.mkdir(parents=True, exist_ok=True)

selected = pd.read_csv(RUN / "selected_admissions.csv", parse_dates=["index_time"])
rec = pd.read_csv(ECG / "record_list.csv", usecols=["subject_id", "study_id", "ecg_time", "path"])
rec["subject_id"] = rec["subject_id"].astype("int64")
rec["study_id"] = rec["study_id"].astype("int64")
rec["ecg_time"] = pd.to_datetime(rec["ecg_time"], errors="coerce")
rec = rec.dropna(subset=["ecg_time"])

selected["subject_id"] = selected["subject_id"].astype("int64")
selected["index_study_id"] = selected["index_study_id"].astype("int64")

subj_split_n = selected.groupby("subject_id")["split"].nunique()
leak_subj = subj_split_n[subj_split_n > 1]

rec_index = rec[["study_id", "ecg_time"]].drop_duplicates("study_id").rename(
    columns={"study_id": "index_study_id", "ecg_time": "record_list_index_time"}
)
chk = selected.merge(rec_index, on="index_study_id", how="left")
dt_sec = (chk["record_list_index_time"] - chk["index_time"]).dt.total_seconds().abs()

rec_by_subj = {sid: g[["study_id", "ecg_time"]].to_numpy() for sid, g in rec.groupby("subject_id")}
rows = []
bad_future = 0
missing_history = 0
index_not_in_history = 0
max_positive_lag_sec = 0.0
for row in selected.itertuples(index=False):
    arr = rec_by_subj.get(int(row.subject_id))
    if arr is None:
        missing_history += 1
        continue
    hist_ids = []
    hist_times = []
    for sid, t in arr:
        if t <= row.index_time:
            hist_ids.append(int(sid))
            hist_times.append(t)
    if not hist_ids:
        missing_history += 1
        continue
    if int(row.index_study_id) not in hist_ids:
        index_not_in_history += 1
    max_t = max(hist_times)
    lag = (max_t - row.index_time).total_seconds()
    max_positive_lag_sec = max(max_positive_lag_sec, lag)
    if lag > 1e-6:
        bad_future += 1
    rows.append({
        "subject_id": int(row.subject_id),
        "hadm_id": int(row.hadm_id),
        "index_study_id": int(row.index_study_id),
        "split": row.split,
        "history_count_recomputed": len(hist_ids),
        "max_history_lag_sec": lag,
        "index_in_history": int(int(row.index_study_id) in hist_ids),
    })

hist_df = pd.DataFrame(rows)
known_death_time = pd.to_datetime(selected["known_death_time"], errors="coerce") if "known_death_time" in selected else pd.Series(pd.NaT, index=selected.index)
delta_days = (known_death_time - selected["index_time"]).dt.total_seconds() / 86400.0
re_30d = ((delta_days >= 0) & (delta_days <= 30)).astype(int)
re_1y = ((delta_days >= 0) & (delta_days <= 365)).astype(int)

audit = {
    "n_rows": int(len(selected)),
    "n_subjects": int(selected["subject_id"].nunique()),
    "split_counts": {str(k): int(v) for k, v in selected["split"].value_counts().to_dict().items()},
    "subject_split_counts": {str(k): int(v) for k, v in selected.groupby("split")["subject_id"].nunique().to_dict().items()},
    "patient_split_leakage_subjects": int(len(leak_subj)),
    "patient_disjoint_split_pass": bool(len(leak_subj) == 0),
    "index_study_missing_in_record_list": int(chk["record_list_index_time"].isna().sum()),
    "index_time_abs_diff_sec_max": float(dt_sec.max()),
    "index_time_abs_diff_sec_mean": float(dt_sec.mean()),
    "missing_history_count": int(missing_history),
    "index_not_in_history_count": int(index_not_in_history),
    "history_future_violation_count": int(bad_future),
    "max_positive_history_lag_sec": float(max_positive_lag_sec),
    "history_count_min": int(hist_df["history_count_recomputed"].min()),
    "history_count_median": float(hist_df["history_count_recomputed"].median()),
    "history_count_max": int(hist_df["history_count_recomputed"].max()),
    "death_before_or_at_index_count": int(pd.to_numeric(selected.get("death_before_or_at_index", 0), errors="coerce").fillna(0).sum()),
    "candidate_valid_false_count": int((~selected["candidate_valid_for_prediction"].astype(bool)).sum()) if "candidate_valid_for_prediction" in selected else None,
    "mortality_30d_recompute_mismatch": int((re_30d != selected["mortality_30d"].astype(int)).sum()) if "mortality_30d" in selected else None,
    "mortality_1y_recompute_mismatch": int((re_1y != selected["mortality_1y"].astype(int)).sum()) if "mortality_1y" in selected else None,
}

with open(OUT / "remote_temporal_leakage_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
pd.DataFrame([audit]).to_csv(OUT / "remote_temporal_leakage_audit_summary.csv", index=False, encoding="utf-8-sig")
hist_df.groupby("split")["history_count_recomputed"].describe().to_csv(OUT / "history_count_by_split.csv", encoding="utf-8-sig")
print(json.dumps(audit, ensure_ascii=False, indent=2))


