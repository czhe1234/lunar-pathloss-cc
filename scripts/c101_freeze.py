"""Select by monitor only; freeze all arms before confirmation preparation/scoring."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'audit/c101_residual'
plan=json.loads((A/'protocol.json').read_text());choices={};means={}
for mode in ['old','coarse','precise']:
 d=R/'runs'/('c101_'+mode);r=json.loads((d/'result.json').read_text());assert r['status']=='complete'
 cfg=json.loads((d/'config.json').read_text());assert cfg['initial_state_sha256']==plan['initial_state_sha256']
 initial=json.loads((d/'initial.json').read_text())['rmse']
 choices[mode]=dict(step=r['best_step'],rmse=r['monitor']['rmse'],sha256=hashlib.sha256((d/'best.pt').read_bytes()).hexdigest())
 means[mode]=sum(r['monitor']['rmse'])/2
means['C82']=sum(initial)/2;eligible=min(means,key=means.get)
value=dict(choices=choices,monitor_mean=means,eligible=eligible,nonzero_modes=[m for m,c in choices.items() if c['step']>0])
p=A/'selection_frozen.json'
if p.exists():assert json.loads(p.read_text())==value
else:p.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)
