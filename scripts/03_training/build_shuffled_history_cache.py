#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import json
import numpy as np

BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
STAGE = BASE / 'pgrf_longitudinal_count_ge2_v1'
SRC = STAGE / 'caches' / 'st_mem_temporal_count_ge2.npz'
DST = STAGE / 'caches' / 'st_mem_temporal_count_ge2_shuffled_history_v1.npz'
MAN = STAGE / 'caches' / 'st_mem_temporal_count_ge2_shuffled_history_v1_manifest.json'

rng = np.random.default_rng(20260702)
z = np.load(SRC, allow_pickle=True)
X = z['Xseq'].astype('float32')
D = z['Dseq'].astype('float32')
M = z['mask'].astype(bool)
Y = z['Y'].astype('float32')
split = z['split'].astype(str)
Xsh = X.copy()

stats = {}
for sp in ['train', 'val', 'test']:
    rows = np.where(split == sp)[0]
    counts = M[rows].sum(1).astype(int)
    last_pos = counts - 1
    target_i = []
    target_j = []
    for local_pos, row in enumerate(rows):
        lp = last_pos[local_pos]
        if lp > 0:
            js = np.arange(lp, dtype=np.int64)
            target_i.append(np.full(len(js), row, dtype=np.int64))
            target_j.append(js)
    if not target_i:
        continue
    ti = np.concatenate(target_i)
    tj = np.concatenate(target_j)
    # Donor pool is historical ECG embeddings only, within the same split.
    donor_idx = rng.integers(0, len(ti), size=len(ti))
    di = ti[donor_idx]
    dj = tj[donor_idx]
    # Assign in chunks to avoid a large temporary copy.
    chunk = 200_000
    for s in range(0, len(ti), chunk):
        e = min(s + chunk, len(ti))
        Xsh[ti[s:e], tj[s:e]] = X[di[s:e], dj[s:e]]
    # Verify index ECG is unchanged.
    idx_i = rows
    idx_j = last_pos
    unchanged = bool(np.allclose(Xsh[idx_i, idx_j], X[idx_i, idx_j]))
    stats[sp] = {
        'n_rows': int(len(rows)),
        'n_shuffled_historical_slots': int(len(ti)),
        'index_ecg_unchanged': unchanged,
    }

np.savez_compressed(DST, Xseq=Xsh, Dseq=D, mask=M, Y=Y, split=split)
MAN.write_text(json.dumps({'source': str(SRC), 'destination': str(DST), 'random_seed': 20260702, 'shuffle_scope': 'within split; historical positions only; index ECG unchanged', 'stats': stats}, indent=2), encoding='utf-8')
print('WROTE', DST)
print(json.dumps(stats, indent=2))


