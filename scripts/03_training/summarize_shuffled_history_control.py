#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
from pathlib import Path
import numpy as np

BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
STAGE = BASE / 'pgrf_longitudinal_count_ge2_v1'
OUT = STAGE / 'shuffled_history_control_stmem_v1'
SEEDS = [42, 123, 1024]
LABELS = ['in_hospital_mortality','mortality_30d','mortality_1y']

def auroc(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p).astype(float); n=len(y); npos=int(y.sum()); nneg=n-npos
    if npos==0 or nneg==0: return np.nan
    order=np.argsort(p, kind='mergesort'); ranks=np.empty(n,dtype=float); i=0
    while i<n:
        j=i+1
        while j<n and p[order[j]]==p[order[i]]: j+=1
        ranks[order[i:j]]=(i+1+j)/2.0; i=j
    return float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def auprc(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p).astype(float); npos=int(y.sum())
    if npos==0: return np.nan
    order=np.argsort(-p, kind='mergesort'); ys=y[order]; tp=np.cumsum(ys); prec=tp/(np.arange(len(y))+1.0)
    return float(prec[ys==1].sum()/npos)

def ece(y,p,bins=15):
    edges=np.linspace(0,1,bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=((p>=lo)&(p<hi)) if hi<1 else ((p>=lo)&(p<=hi))
        if m.any(): out += float(m.mean())*abs(float(y[m].mean())-float(p[m].mean()))
    return out

def metrics(Y,P):
    P=np.clip(P,1e-6,1-1e-6)
    return {
        'macro_auroc': float(np.nanmean([auroc(Y[:,j],P[:,j]) for j in range(3)])),
        'macro_auprc': float(np.nanmean([auprc(Y[:,j],P[:,j]) for j in range(3)])),
        'mean_brier': float(np.nanmean([np.mean((P[:,j]-Y[:,j])**2) for j in range(3)])),
        'mean_ece': float(np.nanmean([ece(Y[:,j],P[:,j]) for j in range(3)])),
    }

def load_ens(paths, key='p_test_platt'):
    ys=[]; ps=[]
    for p in paths:
        z=np.load(p, allow_pickle=True); ykey='y_test' if 'y_test' in z.files else 'y_true'
        ys.append(z[ykey].astype('float32')); ps.append(z[key].astype('float32'))
    assert all(np.array_equal(ys[0],y) for y in ys)
    return ys[0], np.mean(ps,axis=0)

def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

rows=[]
# Existing ST-MEM rows from final/control outputs.
existing = [
    ('Last ECG','index ECG embedding',[STAGE/'last_ecg_calibrated_v1'/'ST-MEM'/'temporal'/'last_ecg'/f'seed{s}_val_test_probs.npz' for s in SEEDS], 'p_test_platt'),
    ('PGRF','serial ECG embeddings + time features',[STAGE/'trajectory_models_groupA'/'ST-MEM'/'temporal'/'fusion'/f'seed{s}_val_test_probs.npz' for s in SEEDS], 'p_test_platt'),
    ('PGRF w/ shuffled history ECG','shuffled historical embeddings',[OUT/f'full_pgrf_seed{s}_test_probs.npz' for s in SEEDS], 'platt_prob'),
]
for method,input_desc,paths,key in existing:
    Y,P=load_ens(paths,key); m=metrics(Y,P); rows.append({'model_control':method,'input':input_desc,**m})
# Metadata rows from previous control experiment.
uc=STAGE/'utilization_confounding_control_v1'/'utilization_confounding_control_ensemble.csv'
with uc.open(encoding='utf-8-sig', newline='') as f:
    for r in csv.DictReader(f):
        if r['backbone']=='ST-MEM' and r['method'] in {'Metadata only','Last ECG + metadata'}:
            rows.append({'model_control':r['method'],'input':r['input'],'macro_auroc':float(r['macro_auroc']),'macro_auprc':float(r['macro_auprc']),'mean_brier':float(r['mean_brier']),'mean_ece':float(r['mean_ece'])})
order={'Last ECG':0,'Metadata only':1,'Last ECG + metadata':2,'PGRF':3,'PGRF w/ shuffled history ECG':4}
rows.sort(key=lambda r: order[r['model_control']])
last = next(r for r in rows if r['model_control']=='Last ECG')['macro_auprc']
for r in rows:
    r['delta_vs_last_ecg'] = r['macro_auprc'] - last if r['model_control']!='Last ECG' else ''
interp = {
    'Last ECG':'baseline',
    'Metadata only':'utilization signal',
    'Last ECG + metadata':'controls history richness',
    'PGRF':'full model',
    'PGRF w/ shuffled history ECG':'tests ECG-content dependence',
}
for r in rows: r['interpretation']=interp[r['model_control']]
fields=['model_control','input','macro_auroc','macro_auprc','delta_vs_last_ecg','mean_brier','mean_ece','interpretation']
write_csv(OUT/'stmem_shuffled_history_control_table.csv', rows, fields)
lines=['| Model / Control | Input | Macro-AUPRC | Delta vs Last ECG | Interpretation |','|---|---|---:|---:|---|']
for r in rows:
    d='-' if r['delta_vs_last_ecg']=='' else f"{r['delta_vs_last_ecg']:+.4f}"
    lines.append(f"| {r['model_control']} | {r['input']} | {r['macro_auprc']:.4f} | {d} | {r['interpretation']} |")
(OUT/'STMEM_Shuffled_History_Control_Table.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
print('WROTE', OUT)


