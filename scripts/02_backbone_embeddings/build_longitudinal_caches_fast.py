import os
from pathlib import Path
import json
import numpy as np
BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
OUT = BASE / 'pgrf_longitudinal_count_ge2_v1'
CACHEDIR = OUT / 'caches'
CACHEDIR.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'ST-MEM': BASE/'pgrf_backbone_sequence_cache_v1/stmem/pgrf_stmem_sequence_cache_temporal.npz',
    'ECG-FM': BASE/'pgrf_backbone_sequence_cache_v1/ecgfm/pgrf_ecgfm_sequence_cache_temporal.npz',
    'CLEAR-HUG': BASE/'pgrf_backbone_sequence_cache_v1/clear_hug/pgrf_clear_hug_sequence_cache_temporal.npz',
    'MERL': BASE/'pgrf_backbone_sequence_cache_v1/merl/pgrf_merl_sequence_cache_temporal.npz',
    'MELP': BASE/'pgrf_backbone_sequence_cache_v1/melp/pgrf_melp_sequence_cache_temporal.npz',
}
summary=[]
for name, src in SOURCES.items():
    print('[build]', name, src, flush=True)
    dat=np.load(src, allow_pickle=True)
    keys=list(dat.files)
    n=dat['mask'].shape[0]
    keep=dat['mask'].astype(bool).sum(axis=1)>=2
    payload={}
    for k in keys:
        arr=dat[k]
        if hasattr(arr,'shape') and len(arr.shape)>0 and arr.shape[0]==n:
            payload[k]=arr[keep]
        else:
            payload[k]=arr
    dst=CACHEDIR / f'{name.lower().replace("-","_")}_temporal_count_ge2.npz'
    # non-compressed npz for speed; fully compatible with np.load
    np.savez(dst, **payload)
    split=payload['split'].astype(str); y=payload['Y']; cnt=payload['mask'].astype(bool).sum(axis=1)
    row={'backbone':name,'src':str(src),'dst':str(dst),'original_n':int(n),'kept_n':int(keep.sum()),'dropped_n':int((~keep).sum()),'train':int((split=='train').sum()),'val':int((split=='val').sum()),'test':int((split=='test').sum()),'mean_ecg_count':float(cnt.mean()),'min_ecg_count':int(cnt.min()),'max_ecg_count':int(cnt.max()),'events_in_hospital':int(y[:,0].sum()),'events_30d':int(y[:,1].sum()),'events_1y':int(y[:,2].sum())}
    summary.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
(OUT/'count_ge2_cache_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print('[DONE]', OUT, flush=True)


