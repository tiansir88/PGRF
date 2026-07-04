#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
STAGE = BASE / 'pgrf_longitudinal_count_ge2_v1'
OUT = STAGE / 'utilization_confounding_control_v1'
OUT.mkdir(parents=True, exist_ok=True)

BACKBONES = ['ST-MEM', 'MERL', 'ECG-FM', 'MELP', 'CLEAR-HUG']
SAFE = {
    'ST-MEM': 'st_mem',
    'ECG-FM': 'ecg_fm',
    'CLEAR-HUG': 'clear_hug',
    'MERL': 'merl',
    'MELP': 'melp',
}
SEEDS = [42, 123, 1024]
LABELS = ['in_hospital_mortality', 'mortality_30d', 'mortality_1y']
EPOCHS = int(os.environ.get('EPOCHS', '30'))
BS = int(os.environ.get('BS', '1024'))
LR = float(os.environ.get('LR', '8e-4'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auroc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y); npos = int(y.sum()); nneg = n - npos
    if npos == 0 or nneg == 0: return np.nan
    order = np.argsort(p, kind='mergesort')
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
    if npos == 0: return np.nan
    order = np.argsort(-p, kind='mergesort')
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / (np.arange(len(y)) + 1.0)
    return float(precision[ys == 1].sum() / npos)


def ece(y, p, bins=15):
    y = np.asarray(y).astype(float); p = np.asarray(p).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            out += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(out)


def metrics(Y, P):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    per = []
    for j, lab in enumerate(LABELS):
        y = Y[:, j]; p = P[:, j]
        per.append({
            'label': lab,
            'auroc': auroc(y, p),
            'auprc': auprc(y, p),
            'brier': float(np.mean((p - y) ** 2)),
            'ece': ece(y, p),
            'prevalence': float(y.mean()),
            'positives': int(y.sum()),
        })
    return {
        'macro_auroc': float(np.nanmean([r['auroc'] for r in per])),
        'macro_auprc': float(np.nanmean([r['auprc'] for r in per])),
        'mean_brier': float(np.nanmean([r['brier'] for r in per])),
        'mean_ece': float(np.nanmean([r['ece'] for r in per])),
        'per_label': per,
    }


def build_meta(D, M):
    # D is days before index ECG; valid entries sorted oldest -> newest, index normally has D=0.
    n = D.shape[0]
    feats = []
    count = M.sum(1).astype(np.float32)
    hist_mask = M.copy()
    # Exclude index ECG from some history-specific stats when count > 1.
    last_idx = count.astype(int) - 1
    hist_mask[np.arange(n), last_idx] = False
    D_valid = np.where(M, D, np.nan)
    D_hist = np.where(hist_mask, D, np.nan)

    def nan_stat(arr, fn, default=0.0):
        with np.errstate(all='ignore'):
            out = fn(arr, axis=1)
        out = np.where(np.isfinite(out), out, default).astype(np.float32)
        return out

    hist_count = hist_mask.sum(1).astype(np.float32)
    max_gap = nan_stat(D_hist, np.nanmax)
    min_gap = nan_stat(D_hist, np.nanmin)
    mean_gap = nan_stat(D_hist, np.nanmean)
    std_gap = nan_stat(D_hist, np.nanstd)
    span = (max_gap - min_gap).astype(np.float32)

    # Time-decay prior stats over all valid ECGs, matching PGRF prior construction.
    W = np.exp(-np.clip(D, 0, 3650) / 30.0).astype(np.float32)
    W[~M] = 0.0
    S = W.sum(1, keepdims=True)
    P = W / np.maximum(S, 1e-8)
    entropy = -(np.where(P > 0, P * np.log(np.maximum(P, 1e-8)), 0.0)).sum(1).astype(np.float32)
    maxw = P.max(1).astype(np.float32)
    eff_n = np.exp(entropy).astype(np.float32)

    # Recent ECG density features.
    within_1d = ((D <= 1) & M).sum(1).astype(np.float32)
    within_7d = ((D <= 7) & M).sum(1).astype(np.float32)
    within_30d = ((D <= 30) & M).sum(1).astype(np.float32)
    within_365d = ((D <= 365) & M).sum(1).astype(np.float32)

    raw = np.stack([
        count, hist_count,
        np.log1p(count), np.log1p(hist_count),
        np.log1p(max_gap), np.log1p(min_gap), np.log1p(mean_gap), np.log1p(std_gap), np.log1p(span),
        np.exp(-max_gap / 30.0), np.exp(-mean_gap / 30.0),
        entropy, maxw, eff_n,
        within_1d, within_7d, within_30d, within_365d,
        within_7d / np.maximum(count, 1), within_30d / np.maximum(count, 1), within_365d / np.maximum(count, 1),
    ], axis=1).astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw


def standardize(A, tr):
    mu = A[tr].mean(0, keepdims=True)
    sd = A[tr].std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((A - mu) / sd).astype(np.float32)


def last_embedding(X, M):
    idx = M.sum(1).astype(int) - 1
    return X[np.arange(X.shape[0]), idx].astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d, k=3, hidden=128, dropout=0.15):
        super().__init__()
        h = min(hidden, max(32, d * 2)) if d < hidden else hidden
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, k),
        )
    def forward(self, x): return self.net(x)


def predict_logits(model, X, bs=4096):
    model.eval(); outs=[]
    with torch.no_grad():
        for st in range(0, len(X), bs):
            xb=torch.tensor(X[st:st+bs], dtype=torch.float32, device=DEVICE)
            outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def fit_platt(val_logits, Yv, test_logits):
    out=[]
    xv=torch.tensor(val_logits, dtype=torch.float32, device=DEVICE)
    xt=torch.tensor(test_logits, dtype=torch.float32, device=DEVICE)
    for j in range(Yv.shape[1]):
        y=torch.tensor(Yv[:,j:j+1], dtype=torch.float32, device=DEVICE)
        m=nn.Linear(1,1).to(DEVICE)
        opt=torch.optim.LBFGS(m.parameters(), lr=0.1, max_iter=80)
        col=xv[:,j:j+1]
        pos=float(Yv[:,j].sum()); neg=float(len(Yv)-pos)
        pw=torch.tensor([neg/max(pos,1.0)], dtype=torch.float32, device=DEVICE)
        def closure():
            opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(m(col), y, pos_weight=pw); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            out.append(torch.sigmoid(m(xt[:,j:j+1])).cpu().numpy().reshape(-1))
    return np.stack(out, axis=1).astype(np.float32)


def train_control(X, Y, split, seed):
    set_seed(seed)
    tr=split=='train'; va=split=='val'; te=split=='test'
    model=MLP(X.shape[1]).to(DEVICE)
    pos=Y[tr].sum(0); neg=tr.sum()-pos
    pw=torch.tensor(np.clip(neg/np.maximum(pos,1),1,80), dtype=torch.float32, device=DEVICE)
    opt=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    train_idx=np.where(tr)[0]
    best=-1; best_state=None; wait=0
    for ep in range(EPOCHS):
        rng=np.random.default_rng(seed+ep); rng.shuffle(train_idx)
        model.train()
        for st in range(0, len(train_idx), BS):
            idx=train_idx[st:st+BS]
            xb=torch.tensor(X[idx], dtype=torch.float32, device=DEVICE)
            yb=torch.tensor(Y[idx], dtype=torch.float32, device=DEVICE)
            logits=model(xb)
            loss=F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if ep % 3 == 0 or ep == EPOCHS-1:
            vl=predict_logits(model, X[va])
            vp=1/(1+np.exp(-vl))
            score=metrics(Y[va], vp)['macro_auprc']
            if score > best + 1e-5:
                best=score; wait=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= 8: break
    model.load_state_dict(best_state)
    val_logits=predict_logits(model, X[va])
    test_logits=predict_logits(model, X[te])
    raw=1/(1+np.exp(-test_logits))
    platt=fit_platt(val_logits, Y[va], test_logits)
    return Y[te], raw.astype(np.float32), platt.astype(np.float32)


def load_ensemble(paths, key='p_test_platt'):
    ys=[]; ps=[]
    for p in paths:
        z=np.load(p, allow_pickle=True)
        ykey='y_test' if 'y_test' in z.files else 'y_true'
        ys.append(z[ykey].astype(np.float32)); ps.append(z[key].astype(np.float32))
    assert all(np.array_equal(ys[0], y) for y in ys)
    return ys[0], np.mean(ps, axis=0)


def paths_for_existing(bb, method):
    if method == 'Last ECG':
        return [STAGE/'last_ecg_calibrated_v1'/bb/'temporal'/'last_ecg'/f'seed{s}_val_test_probs.npz' for s in SEEDS]
    if method == 'PGRF':
        group='trajectory_models_groupA' if bb in {'ST-MEM','ECG-FM','CLEAR-HUG'} else 'trajectory_models_groupB'
        return [STAGE/group/bb/'temporal'/'fusion'/f'seed{s}_val_test_probs.npz' for s in SEEDS]
    raise KeyError(method)


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    started=time.time()
    rows=[]; per_rows=[]; seed_rows=[]
    print('[BOOT]', DEVICE, 'out', OUT, 'epochs', EPOCHS, flush=True)
    for bb in BACKBONES:
        cache=STAGE/'caches'/f'{SAFE[bb]}_temporal_count_ge2.npz'
        print('[CACHE]', bb, cache, flush=True)
        dat=np.load(cache, allow_pickle=True)
        Xseq=dat['Xseq'].astype(np.float32); D=dat['Dseq'].astype(np.float32); M=dat['mask'].astype(bool)
        Y=dat['Y'].astype(np.float32); split=dat['split'].astype(str)
        tr=split=='train'
        meta=standardize(build_meta(D,M), tr)
        last=standardize(last_embedding(Xseq,M), tr)
        specs=[('Metadata only', meta), ('Last ECG + metadata', np.concatenate([last, meta], axis=1).astype(np.float32))]
        for method, feat in specs:
            seed_probs=[]; seed_y=[]
            for seed in SEEDS:
                print('[TRAIN]', bb, method, 'seed', seed, 'dim', feat.shape[1], flush=True)
                yte, raw, platt = train_control(feat, Y, split, seed)
                seed_y.append(yte); seed_probs.append(platt)
                m=metrics(yte, platt)
                seed_rows.append({'backbone':bb,'method':method,'seed':seed,'calibration':'platt',**{k:v for k,v in m.items() if k!='per_label'}})
                np.savez_compressed(OUT/f'{SAFE[bb]}_{method.lower().replace(" ","_").replace("+","plus")}_seed{seed}_probs.npz', y_test=yte, p_test_platt=platt, p_test_raw=raw)
            assert all(np.array_equal(seed_y[0], y) for y in seed_y)
            ens=np.mean(seed_probs, axis=0)
            mm=metrics(seed_y[0], ens)
            rows.append({'backbone':bb,'method':method,'input':'history metadata' if method=='Metadata only' else 'index ECG embedding + history metadata','aggregation':'3-seed probability ensemble','calibration':'platt',**{k:v for k,v in mm.items() if k!='per_label'}})
            for r in mm['per_label']:
                per_rows.append({'backbone':bb,'method':method,**r})
        # Existing comparison rows for same table.
        for method in ['Last ECG','PGRF']:
            Yt, P=load_ensemble(paths_for_existing(bb, method), 'p_test_platt')
            mm=metrics(Yt, P)
            rows.append({'backbone':bb,'method':method,'input':'index ECG embedding' if method=='Last ECG' else 'serial ECG embeddings + time features','aggregation':'3-seed probability ensemble','calibration':'platt',**{k:v for k,v in mm.items() if k!='per_label'}})
            for r in mm['per_label']:
                per_rows.append({'backbone':bb,'method':method,**r})
        # release cache arrays
        del Xseq, D, M, Y, split, meta, last
    # Sort rows for readability.
    method_order={'Last ECG':0,'Metadata only':1,'Last ECG + metadata':2,'PGRF':3}
    bb_order={b:i for i,b in enumerate(BACKBONES)}
    rows.sort(key=lambda r:(bb_order[r['backbone']], method_order[r['method']]))
    per_rows.sort(key=lambda r:(bb_order[r['backbone']], method_order[r['method']], r['label']))
    seed_rows.sort(key=lambda r:(bb_order[r['backbone']], method_order[r['method']], r['seed']))
    fields=['backbone','method','input','aggregation','calibration','macro_auroc','macro_auprc','mean_brier','mean_ece']
    write_csv(OUT/'utilization_confounding_control_ensemble.csv', rows, fields)
    write_csv(OUT/'utilization_confounding_control_per_endpoint.csv', per_rows, ['backbone','method','label','auroc','auprc','brier','ece','prevalence','positives'])
    write_csv(OUT/'utilization_confounding_control_seed_metrics.csv', seed_rows, ['backbone','method','seed','calibration','macro_auroc','macro_auprc','mean_brier','mean_ece'])
    # Compact markdown table.
    lines=['| Backbone | Method | Input | Macro-AUROC | Macro-AUPRC | Cal. Brier | Cal. ECE |', '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['backbone']} | {r['method']} | {r['input']} | {r['macro_auroc']:.4f} | {r['macro_auprc']:.4f} | {r['mean_brier']:.4f} | {r['mean_ece']:.4f} |")
    (OUT/'Utilization_Confounding_Control_Report.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({'out':str(OUT),'backbones':BACKBONES,'seeds':SEEDS,'epochs':EPOCHS,'device':str(DEVICE),'elapsed_sec':time.time()-started}, indent=2), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print('[DONE]', OUT, 'elapsed', time.time()-started, flush=True)

if __name__ == '__main__':
    main()


