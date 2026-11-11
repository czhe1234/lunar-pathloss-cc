"""Freeze monitor-only C105 choices before confirmation."""
import json,hashlib
from pathlib import Path
R=Path.cwd();A=R/'audit/c105_context';plan=json.loads((A/'protocol.json').read_text());choices={};means={}
for mode in ['local','context']:
 d=R/'runs'/('c105_'+mode);r=json.loads((d/'result.json').read_text());assert r['status']=='complete'
 cfg=json.loads((d/'config.json').read_text());assert cfg['initial_state_sha256']==plan['initial_state_sha256']
 initial=json.loads((d/'initial.json').read_text())['rmse']
 choices[mode]=dict(epoch=r['best_epoch'],rmse=r['monitor']['rmse'],sha256=hashlib.sha256((d/'best.pt').read_bytes()).hexdigest())
 means[mode]=sum(r['monitor']['rmse'])/2
means['C82']=sum(initial)/2;eligible=min(means,key=means.get)
value=dict(choices=choices,monitor_mean=means,eligible=eligible,nonzero_modes=[m for m,c in choices.items() if c['epoch']>0])
p=A/'selection_frozen.json'
if p.exists():assert json.loads(p.read_text())==value
else:p.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)
