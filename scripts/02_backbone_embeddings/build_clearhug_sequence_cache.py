#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from scipy import interpolate
from torch.utils.data import DataLoader, Dataset


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
RAW_ROOT = Path(os.environ.get("PGRF_MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg"))
BENCH = BASE / "external_baseline_benchmark_dmt_v1"
QRS_EXTRACT = BENCH / "extract_qrs_language_baseline_protocol_a.py"

LABELS = ["in_hospital_mortality", "mortality_30d", "mortality_1y"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ALIASES = {k: [k, k.lower(), k.upper()] for k in LEADS}
LEAD_ALIASES.update({"aVR": ["aVR", "AVR", "avr"], "aVL": ["aVL", "AVL", "avl"], "aVF": ["aVF", "AVF", "avf"]})


def import_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def resolve_record_path(p: str) -> str:
    p = str(p).replace("\\", "/").replace(".hea", "").replace(".dat", "")
    for c in [Path(p), RAW_ROOT / p]:
        if c.exists() or c.with_suffix(".hea").exists():
            return str(c)
    return str(RAW_ROOT / p)


def read_ecg_12x(path: str) -> tuple[np.ndarray, int]:
    rec = wfdb.rdrecord(resolve_record_path(path))
    sig = np.asarray(rec.p_signal, dtype=np.float32)
    fs = int(rec.fs) if rec.fs else 500
    names = list(rec.sig_name) if rec.sig_name else [str(i) for i in range(sig.shape[1])]
    out = np.zeros((12, sig.shape[0]), dtype=np.float32)
    for j, lead in enumerate(LEADS):
        found = None
        for alias in LEAD_ALIASES[lead]:
            if alias in names:
                found = names.index(alias)
                break
        if found is not None:
            out[j] = sig[:, found]
    return np.nan_to_num(out), fs


def resample_to_100hz(x: np.ndarray, fs: int) -> np.ndarray:
    if fs == 100:
        return x.astype(np.float32)
    old_t = np.linspace(0, x.shape[1] / fs, x.shape[1], endpoint=False)
    new_len = max(100, int(round(x.shape[1] * 100 / fs)))
    new_t = np.linspace(0, x.shape[1] / fs, new_len, endpoint=False)
    y = np.stack([
        interpolate.interp1d(old_t, ch, kind="linear", fill_value="extrapolate")(new_t)
        for ch in x
    ], axis=0)
    return y.astype(np.float32)


class ClearHugPathDataset(Dataset):
    def __init__(self, paths: list[str], tokenizer):
        self.paths = list(paths)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        x, fs = read_ecg_12x(path)
        x = resample_to_100hz(x, fs)
        seq, ch, tm, pad = self.tokenizer.encode(x)
        return path, torch.tensor(seq), torch.tensor(ch), torch.tensor(tm), torch.tensor(pad)


def collate_qrs(batch):
    paths, seq, ch, tm, pad = zip(*batch)
    return list(paths), torch.stack(seq), torch.stack(ch), torch.stack(tm), torch.stack(pad)


def load_clearhug(device: torch.device):
    qrs = import_file("qrs_extract_pgrf", QRS_EXTRACT)
    model, max_len = qrs.load_model("clear_hug", device)
    tokenizer = qrs.SimpleQRSTokenizer(max_len=max_len)
    return model, tokenizer, {
        "checkpoint": str(BENCH / "clear_hug/checkpoints/clear_pretrained_google_drive.pth"),
        "qrs_extract": str(QRS_EXTRACT),
        "max_len": int(max_len),
        "protocol": "Frozen CLEAR-HUG QRS-token embedding; records resampled to 100 Hz; cls-token mean over 12 leads.",
    }


@torch.no_grad()
def embed_paths(paths: list[str], output: Path, batch_size: int, num_workers: int, device: torch.device):
    if output.exists():
        z = np.load(output, allow_pickle=True)
        return z["path"].astype(str), z["embedding"].astype(np.float32), {"loaded_existing": True}

    model, tokenizer, info = load_clearhug(device)
    loader = DataLoader(
        ClearHugPathDataset(paths, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_qrs,
        pin_memory=torch.cuda.is_available(),
    )
    all_paths, chunks = [], []
    started = time.time()
    for bi, (batch_paths, x, ch, tm, pad) in enumerate(loader, start=1):
        if bi == 1 or bi % 25 == 0:
            print(f"[clear_hug] batch={bi}/{len(loader)} records_done={len(all_paths)} elapsed={time.time()-started:.1f}s", flush=True)
        x = x.to(device, non_blocking=True)
        ch = ch.to(device, non_blocking=True)
        tm = tm.to(device, non_blocking=True)
        tokens, _ = model(x, in_chan_matrix=ch, in_time_matrix=tm, return_all_tokens=True)
        emb = tokens[:, :12].mean(dim=1)
        emb = F.normalize(emb, dim=-1).detach().cpu().float().numpy()
        all_paths.extend(batch_paths)
        chunks.append(emb)

    arr = np.concatenate(chunks, axis=0).astype(np.float32)
    paths_arr = np.array(all_paths, dtype=object)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, path=paths_arr, embedding=arr)
    info.update({"loaded_existing": False, "records": int(len(paths_arr)), "embedding_dim": int(arr.shape[1])})
    output.with_suffix(".summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
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
    split_temporal = temporal["temporal_split"].astype(str).to_numpy()
    if len(split_temporal) != n:
        raise RuntimeError(f"temporal split length mismatch: {len(split_temporal)} vs {n}")
    return selected, seq_paths, dseq, mask, y, split_random, split_temporal, sorted(unique)


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
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--row_limit", type=int, default=0)
    args = ap.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "clear_hug"
    out = BASE / "pgrf_backbone_sequence_cache_v1" / model_name
    out.mkdir(parents=True, exist_ok=True)
    selected, seq_paths, dseq, mask, y, split_random, split_temporal, unique_paths = rebuild_sequences(args.max_len, args.row_limit)
    print(json.dumps({
        "model": model_name,
        "rows": int(len(selected)),
        "unique_history_ecgs": int(len(unique_paths)),
        "mean_history_len": float(mask.sum(1).mean()),
        "device": str(device),
    }, indent=2), flush=True)

    emb_paths, emb, emb_info = embed_paths(
        unique_paths,
        out / f"{model_name}_unique_history_embeddings_maxlen{args.max_len}.npz",
        args.batch_size,
        args.num_workers,
        device,
    )
    xseq = build_xseq(seq_paths, emb_paths, emb)
    np.savez_compressed(out / f"pgrf_{model_name}_sequence_cache_random.npz", Xseq=xseq, Dseq=dseq, mask=mask, Y=y, split=split_random)
    np.savez_compressed(out / f"pgrf_{model_name}_sequence_cache_temporal.npz", Xseq=xseq, Dseq=dseq, mask=mask, Y=y, split=split_temporal)
    selected.to_csv(out / "selected_admissions_pgrf.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": model_name,
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


