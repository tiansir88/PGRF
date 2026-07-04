#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from scipy.signal import butter, filtfilt, resample
from torch.utils.data import DataLoader, Dataset


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
RAW_ROOT = Path(os.environ.get("PGRF_MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg"))
BENCH = BASE / "external_baseline_benchmark_dmt_v1"
STMEM_REPO = BENCH / "repos" / "ST-MEM"
FAIRSEQ_SIGNALS = BENCH / "repos" / "fairseq-signals"
STMEM_CKPT = BENCH / "stmem" / "checkpoints" / "st_mem_encoder.pth"
ECGFM_CKPT = BENCH / "ecgfm" / "checkpoints" / "mimic_iv_ecg_physionet_pretrained.pt"

LABELS = ["in_hospital_mortality", "mortality_30d", "mortality_1y"]
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ALIASES = {
    "I": ["I", "i"], "II": ["II", "ii"], "III": ["III", "iii"],
    "aVR": ["aVR", "AVR", "avr"], "aVL": ["aVL", "AVL", "avl"], "aVF": ["aVF", "AVF", "avf"],
    "V1": ["V1", "v1"], "V2": ["V2", "v2"], "V3": ["V3", "v3"],
    "V4": ["V4", "v4"], "V5": ["V5", "v5"], "V6": ["V6", "v6"],
}


def resolve_record_path(p: str) -> str:
    p = str(p).replace("\\", "/").replace(".hea", "").replace(".dat", "")
    for c in [Path(p), RAW_ROOT / p]:
        if c.exists() or c.with_suffix(".hea").exists():
            return str(c)
    return str(RAW_ROOT / p)


def read_12lead_raw(path: str) -> tuple[np.ndarray, int]:
    rec = wfdb.rdrecord(resolve_record_path(path))
    sig = np.asarray(rec.p_signal, dtype=np.float32)
    fs = int(rec.fs) if rec.fs else 500
    names = list(rec.sig_name) if rec.sig_name else [str(i) for i in range(sig.shape[1])]
    out = np.zeros((sig.shape[0], 12), dtype=np.float32)
    for j, lead in enumerate(STANDARD_LEADS):
        found = None
        for alias in LEAD_ALIASES.get(lead, [lead]):
            if alias in names:
                found = names.index(alias)
                break
        if found is not None:
            out[:, j] = sig[:, found]
    return np.nan_to_num(out), fs


def safe_filter(sig: np.ndarray, fs: int, kind: str, cutoff: float) -> np.ndarray:
    try:
        nyq = 0.5 * fs
        btype = "highpass" if kind == "high" else "lowpass"
        b, a = butter(3, cutoff / nyq, btype=btype)
        return filtfilt(b, a, sig, axis=0).astype(np.float32)
    except Exception:
        return sig.astype(np.float32)


def read_stmem_tensor(path: str) -> np.ndarray:
    target_fs, target_len = 250, 2250
    sig, fs = read_12lead_raw(path)
    if fs != target_fs:
        sig = resample(sig, int(round(sig.shape[0] * target_fs / fs)), axis=0).astype(np.float32)
        fs = target_fs
    sig = safe_filter(sig, fs, "high", 0.67)
    sig = safe_filter(sig, fs, "low", 40.0)
    if sig.shape[0] >= target_len:
        start = (sig.shape[0] - target_len) // 2
        sig = sig[start:start + target_len]
    else:
        sig = np.concatenate([sig, np.zeros((target_len - sig.shape[0], 12), dtype=np.float32)], axis=0)
    mean = sig.mean(axis=(0, 1), keepdims=True)
    std = sig.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    sig = np.clip((sig - mean) / std, -10.0, 10.0)
    return sig.T.astype(np.float32)


def read_ecgfm_tensor(path: str) -> np.ndarray:
    # Practical PGRF extraction: one centered 5-second segment per record.
    # This avoids the memory-heavy segment cache used in the old pair extractor.
    target_fs, seg_len = 500, 2500
    sig, fs = read_12lead_raw(path)
    if fs != target_fs:
        sig = resample(sig, int(round(sig.shape[0] * target_fs / fs)), axis=0).astype(np.float32)
    mean = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    sig = np.clip((sig - mean) / std, -10.0, 10.0)
    if sig.shape[0] >= seg_len:
        start = (sig.shape[0] - seg_len) // 2
        sig = sig[start:start + seg_len]
    else:
        sig = np.concatenate([sig, np.zeros((seg_len - sig.shape[0], 12), dtype=np.float32)], axis=0)
    return sig.T.astype(np.float32)


class ECGPathDataset(Dataset):
    def __init__(self, paths: list[str], model: str):
        self.paths = list(paths)
        self.model = model

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.model == "stmem":
            x = read_stmem_tensor(path)
        else:
            x = read_ecgfm_tensor(path)
        return path, torch.tensor(x, dtype=torch.float32)


def collate(batch):
    paths, xs = zip(*batch)
    return list(paths), torch.stack(xs, 0)


def load_stmem(device: torch.device):
    sys.path.insert(0, str(STMEM_REPO))
    from models.encoder import st_mem_vit_base

    model = st_mem_vit_base(num_leads=12, seq_len=2250, patch_size=75)
    raw = torch.load(STMEM_CKPT, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ["model", "state_dict", "model_state_dict", "encoder", "module"]:
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    state = {}
    for k, v in raw.items():
        kk = k
        for prefix in ["module.", "model.", "encoder."]:
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        if not any(part in kk for part in ["decoder", "mask_embedding", "to_decoder_embedding"]):
            state[kk] = v
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"checkpoint": str(STMEM_CKPT), "missing_keys": list(missing)[:20], "unexpected_keys": list(unexpected)[:20]}


def load_ecgfm(device: torch.device):
    sys.path.insert(0, str(FAIRSEQ_SIGNALS))
    from fairseq_signals.models import build_model_from_checkpoint

    model = build_model_from_checkpoint(checkpoint_path=str(ECGFM_CKPT))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"checkpoint": str(ECGFM_CKPT), "segment_policy": "center_5s_single_segment"}


@torch.no_grad()
def embed_paths(model_name: str, paths: list[str], output: Path, batch_size: int, num_workers: int, device: torch.device):
    if output.exists():
        z = np.load(output, allow_pickle=True)
        return z["path"].astype(str), z["embedding"].astype(np.float32), {"loaded_existing": True}

    if model_name == "stmem":
        model, info = load_stmem(device)
    elif model_name == "ecgfm":
        model, info = load_ecgfm(device)
    else:
        raise ValueError(model_name)

    loader = DataLoader(
        ECGPathDataset(paths, model_name),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    all_paths, chunks = [], []
    started = time.time()
    for bi, (batch_paths, x) in enumerate(loader, start=1):
        if bi == 1 or bi % 25 == 0:
            print(f"[{model_name}] batch={bi}/{len(loader)} records_done={len(all_paths)} elapsed={time.time()-started:.1f}s", flush=True)
        x = x.to(device, non_blocking=True)
        if model_name == "stmem":
            emb = model(x)
        else:
            out = model(source=x, mask=False, features_only=True)
            emb = out["x"].mean(dim=1)
        emb = F.normalize(emb, dim=-1).detach().cpu().float().numpy()
        all_paths.extend(batch_paths)
        chunks.append(emb)
    arr = np.concatenate(chunks, axis=0).astype(np.float32)
    paths_arr = np.array(all_paths, dtype=object)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, path=paths_arr, embedding=arr)
    info.update({"loaded_existing": False, "records": int(len(paths_arr)), "embedding_dim": int(arr.shape[1])})
    output.with_suffix(".summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return paths_arr.astype(str), arr, info


def rebuild_sequences(max_len: int, row_limit: int = 0):
    selected = pd.read_csv(BASE / "selective_trajecg_stage2_full_cache_gated_probe_v1" / "selected_admissions.csv", parse_dates=["index_time"])
    temporal = pd.read_csv(BASE / "pgrf_temporal_split_v1" / "selected_admissions_temporal_split.csv", parse_dates=["index_time"])
    if row_limit > 0:
        selected = selected.head(row_limit).copy()
        temporal = temporal[temporal["hadm_id"].isin(set(selected["hadm_id"]))].copy()
    need = set(selected["subject_id"].astype("int64").tolist())
    chunks = []
    for ch in pd.read_csv(RAW_ROOT / "record_list.csv", chunksize=200000):
        ch = ch[ch["subject_id"].isin(need)]
        if len(ch):
            chunks.append(ch)
    rec = pd.concat(chunks, ignore_index=True)
    rec["ecg_time"] = pd.to_datetime(rec["ecg_time"])
    rec["subject_id"] = rec["subject_id"].astype("int64")
    rec = rec.sort_values(["subject_id", "ecg_time", "study_id"]).reset_index(drop=True)
    by = {}
    for sid, g in rec.groupby("subject_id"):
        gg = g[["study_id", "ecg_time", "path"]].reset_index(drop=True)
        by[int(sid)] = {
            "time_ns": gg["ecg_time"].astype("int64").to_numpy(),
            "study_id": gg["study_id"].to_numpy(),
            "path": gg["path"].astype(str).to_numpy(),
        }

    n = len(selected)
    seq_paths = np.empty((n, max_len), dtype=object)
    seq_paths[:] = ""
    dseq = np.zeros((n, max_len), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=bool)
    unique = set()
    for i, r in enumerate(selected.itertuples(index=False)):
        g = by.get(int(r.subject_id))
        if g is None:
            continue
        index_ns = pd.Timestamp(r.index_time).value
        end = int(np.searchsorted(g["time_ns"], index_ns, side="right"))
        start = max(0, end - max_len)
        paths = list(g["path"][start:end])
        times_ns = list(g["time_ns"][start:end])
        # Stable tie handling: make the explicit index ECG the final token when it is in the window.
        index_path = str(r.index_path)
        if index_path in paths and paths[-1] != index_path:
            k = paths.index(index_path)
            p = paths.pop(k)
            t = times_ns.pop(k)
            paths.append(p)
            times_ns.append(t)
        L = len(paths)
        if L == 0:
            continue
        seq_paths[i, :L] = paths
        mask[i, :L] = True
        dseq[i, :L] = [max(0.0, (index_ns - int(t)) / (86400.0 * 1e9)) for t in times_ns]
        unique.update(paths)

    y = selected[LABELS].to_numpy(dtype=np.float32)
    split_random = selected["split"].astype(str).to_numpy()
    # Align temporal split by row order; selected_admissions_temporal_split was generated from selected row order.
    temporal_split = temporal["temporal_split"].astype(str).to_numpy()
    if len(temporal_split) != n:
        raise RuntimeError(f"temporal split length mismatch: {len(temporal_split)} vs {n}")
    return selected, seq_paths, dseq, mask, y, split_random, temporal_split, sorted(unique)


def build_xseq(seq_paths: np.ndarray, emb_paths: np.ndarray, emb: np.ndarray):
    path_to_i = {str(p): i for i, p in enumerate(emb_paths)}
    n, t = seq_paths.shape
    d = emb.shape[1]
    xseq = np.zeros((n, t, d), dtype=np.float32)
    missing = 0
    for i in range(n):
        for j in range(t):
            p = str(seq_paths[i, j])
            if not p:
                continue
            k = path_to_i.get(p)
            if k is None:
                missing += 1
            else:
                xseq[i, j] = emb[k]
    if missing:
        raise RuntimeError(f"missing embeddings for {missing} sequence tokens")
    return xseq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["stmem", "ecgfm"], required=True)
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--row_limit", type=int, default=0)
    args = ap.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = BASE / "pgrf_backbone_sequence_cache_v1" / args.model
    out.mkdir(parents=True, exist_ok=True)
    selected, seq_paths, dseq, mask, y, split_random, split_temporal, unique_paths = rebuild_sequences(args.max_len, args.row_limit)
    print(json.dumps({
        "model": args.model,
        "rows": int(len(selected)),
        "unique_history_ecgs": int(len(unique_paths)),
        "mean_history_len": float(mask.sum(1).mean()),
        "device": str(device),
    }, indent=2), flush=True)

    emb_paths, emb, emb_info = embed_paths(
        args.model,
        unique_paths,
        out / f"{args.model}_unique_history_embeddings_maxlen{args.max_len}.npz",
        args.batch_size,
        args.num_workers,
        device,
    )
    xseq = build_xseq(seq_paths, emb_paths, emb)
    np.savez_compressed(out / f"pgrf_{args.model}_sequence_cache_random.npz", Xseq=xseq, Dseq=dseq, mask=mask, Y=y, split=split_random)
    np.savez_compressed(out / f"pgrf_{args.model}_sequence_cache_temporal.npz", Xseq=xseq, Dseq=dseq, mask=mask, Y=y, split=split_temporal)
    selected.to_csv(out / "selected_admissions_pgrf.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": args.model,
        "max_len": args.max_len,
        "rows": int(len(selected)),
        "unique_history_ecgs": int(len(unique_paths)),
        "embedding_dim": int(emb.shape[1]),
        "mean_history_len": float(mask.sum(1).mean()),
        "random_split_counts": {str(k): int(v) for k, v in pd.Series(split_random).value_counts().to_dict().items()},
        "temporal_split_counts": {str(k): int(v) for k, v in pd.Series(split_temporal).value_counts().to_dict().items()},
        "embedding_info": emb_info,
        "elapsed_sec": float(time.time() - started),
    }
    (out / "pgrf_cache_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


