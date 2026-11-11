"""Can a hidden terrain-specific height datum be inferred from other training TXs? Oracle only."""
import json,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c88_mesh';D=R.parent/'data'
q=json.load(open(R/'audit/c16b_nearfield_affine_probe/records.json'))
rows=list(csv.DictReader(open(D/'train_index.csv')));hh=np.load(D/'train_hm.npy',mmap_mode='r')
rec=[]
for e in q:
    b=e['bands']['415']
    if not b['success'] or b['residual_RMSE_db']>=.5:continue
    tr,tc=e['tx_row'],e['tx_col'];h=hh[int(rows[e['index_row']]['hm_row'])]
    hs=h[min(tr+1,255),tc];hr=h[tr,tc];obs=b['effective_vertical_separation']+hr+1
    rec.append((e['terrain_id'],hs,obs, int(e['index_row'])))
a=np.array([r[1:3] for r in rec],float);terr=np.array([r[0] for r in rec]);t=np.arange(-.50,.501,.01)
y0=np.floor(a[:,0])+3
yh=np.floor(a[:,0,None]+t)-t+3
rng=np.random.default_rng(88031)
trn=np.unique(terr);rng.shuffle(trn);learn=np.isin(terr,trn[::2])
# The fitted direct field is an effective area response, so allow an instrument intercept fit on other terrains.
bias=np.median((a[:,1]-y0)[learn]);off=a[:,1,None]-yh-bias
global_phase=t[np.argmin((off[learn]**2).mean(0))]
evidence=[];pairs=0
for en in trn[1::2]:
    v=np.flatnonzero(terr==en);rng.shuffle(v)
    # Collapse any duplicate TX location. Each unseen query TX must be distinct from its support.
    by={}
    for k in v:
        row=rows[rec[k][3]];by[(row['tx_row'],row['tx_col'])]=k
    v=np.array(list(by.values()))
    if len(v)<2:continue
    for f in (0,1):
        fit=v[f::2];test=v[1-f::2]
        if not len(fit) or not len(test):continue
        ix=np.argmin((off[fit]**2).mean(0)+.02*t*t)
        evidence.extend(np.stack([off[test,50],off[test,np.argmin(abs(t-global_phase))],off[test,ix]],1).tolist());pairs+=1
d=np.array(evidence)
out=dict(scope='TRAIN FIT only, two-band near-source power fits are effective height proxies, not actual recorded TX altitude',eligible=len(rec),train_instrument_bias_m=bias,
         heldout_terrains_with_multiple_unique_TX=pairs//2,distinct_query_TX_evaluations=len(d),global_phase_m=global_phase,
         rms_query_apparent_source_height_m=(d**2).mean(0)**.5,names=['published datum zero','global phase','same-terrain other-TX phase oracle'])
out['rms_query_apparent_source_height_m']=out['rms_query_apparent_source_height_m'].tolist()
(A/'altitude.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
