import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
SRC_RUN = BASE / "selective_trajecg_stage2_full_cache_gated_probe_v1"
SRC_CACHE = SRC_RUN / "stage2_sequence_cache.npz"
SELECTED = SRC_RUN / "selected_admissions.csv"
OUT = BASE / "pgrf_temporal_split_v1"
OUT.mkdir(parents=True, exist_ok=True)

dat = np.load(SRC_CACHE, allow_pickle=True)
Xseq = dat["Xseq"].astype("float32")
Dseq = dat["Dseq"].astype("float32")
mask = dat["mask"].astype(bool)
Y = dat["Y"].astype("float32")

selected = pd.read_csv(SELECTED, parse_dates=["index_time"])
assert len(selected) == Xseq.shape[0], (len(selected), Xseq.shape[0])
selected["subject_id"] = selected["subject_id"].astype("int64")

sub_first = selected.groupby("subject_id")["index_time"].min().sort_values()
n = len(sub_first)
train_sub = set(sub_first.iloc[: int(0.70 * n)].index)
val_sub = set(sub_first.iloc[int(0.70 * n) : int(0.85 * n)].index)
test_sub = set(sub_first.iloc[int(0.85 * n) :].index)

split = selected["subject_id"].map(
    lambda s: "train" if int(s) in train_sub else "val" if int(s) in val_sub else "test"
).to_numpy(dtype=str)

selected["temporal_split"] = split
selected.to_csv(OUT / "selected_admissions_temporal_split.csv", index=False, encoding="utf-8-sig")
np.savez_compressed(OUT / "pgrf_sequence_cache_temporal_split.npz", Xseq=Xseq, Dseq=Dseq, mask=mask, Y=Y, split=split)

cross = selected.groupby("subject_id")["temporal_split"].nunique()
summary = {
    "split_type": "patient-disjoint pseudo-temporal split by each subject's earliest index_time",
    "note": "MIMIC dates are de-identified and shifted; this split preserves temporal ordering in shifted time while enforcing patient disjointness, but should be described cautiously as pseudo-temporal rather than real deployment calendar split.",
    "n_rows": int(len(selected)),
    "n_subjects": int(selected["subject_id"].nunique()),
    "split_counts": {str(k): int(v) for k, v in selected["temporal_split"].value_counts().to_dict().items()},
    "subject_split_counts": {str(k): int(v) for k, v in selected.groupby("temporal_split")["subject_id"].nunique().to_dict().items()},
    "subject_cross_split_count": int((cross > 1).sum()),
    "time_ranges": {
        str(k): {
            "min": str(v["min"]),
            "max": str(v["max"]),
        }
        for k, v in selected.groupby("temporal_split")["index_time"].agg(["min", "max"]).to_dict("index").items()
    },
    "prevalence": {
        str(k): {lab: float(val) for lab, val in vals.items()}
        for k, vals in selected.groupby("temporal_split")[["in_hospital_mortality", "mortality_30d", "mortality_1y"]].mean().to_dict("index").items()
    },
}

with open(OUT / "temporal_split_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
pd.DataFrame([summary]).to_csv(OUT / "temporal_split_summary.csv", index=False, encoding="utf-8-sig")
print(json.dumps(summary, ensure_ascii=False, indent=2))


