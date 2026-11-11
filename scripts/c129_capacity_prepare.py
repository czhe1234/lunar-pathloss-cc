import json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'audit/c129_fitgap'
old=json.loads((R/'audit/c127_joint/protocol.json').read_text())
units=[u for u in old['groups'].values() if len(u)==4]
selected=units[:4]+[units[j] for j in np.random.default_rng(129073).permutation(np.arange(4,len(units)))[:12]]
ids64=[old['fit'][j] for u in selected for j in u]
assert len(ids64)==64 and len(set(ids64))==64
plan=dict(fit16=ids64[:16],fit64=ids64,monitor=old['monitor'],batch=16,
 max_exposures=1700,report_every_exposures=25,minimum_exposures=100,stop_both_FIT_below=.95,
 reference='C82',objective='ordinary MSE only, exact full-subset valid-pixel normalization per epoch; no joint common-state supervision',
 lr=.0004,weight_decay=.01,optimizer='AdamWconstantLR,clip2',seed=129041,sample_seed=129073,
 arms={'n16':dict(maps=16,width=1),'n64':dict(maps=64,width=1),'w64':dict(maps=64,width=2)},
 initialization='same C99 pretrained TX body,zero readout,C127 scene; width2 via channel duplication with fixed attention head_dim; relative1e-3body jitter in both widths',
 scope='capacity/exposure diagnosis only, FIT-selected; no officialVAL; not fullTRAIN expansion',
 promotion='same128monitor both >=.20 then original484each>=.30+positiveCI; capacity fit not a promotion score')
(A/'capacity_protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({k:v for k,v in plan.items() if k not in ['fit16','fit64','monitor']},ensure_ascii=False))
