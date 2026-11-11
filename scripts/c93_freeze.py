"""Choose between prespecified input-only locators only on 48 TRAIN selection terrains."""
import json,hashlib
from pathlib import Path
A=Path(__file__).resolve().parents[1]/'audit/c93_joint_miss';modes=['angles','flow','flowzero','coarse','rotated']
rows={m:json.load(open(A/('lookup_'+m)/'select.json')) for m in modes}
selected=max(modes[:3],key=lambda m:sum(s['AP'] for s in rows[m]['bands']))
dest=A/'freeze.json';out=dict(selected=selected,thresholds={m:rows[m]['threshold'] for m in modes},
   selection={m:rows[m]['bands'] for m in modes},purpose='mean equal-band AP on 48 terrain maps; 484-terrain confirm labels not queried by selection',
   refs={m:hashlib.sha256((A/('lookup_'+m)/'select.npz').read_bytes()).hexdigest() for m in modes})
if dest.exists():assert json.loads(dest.read_text())==out
else:dest.write_text(json.dumps(out,indent=2)+'\n')
print(selected,rows[selected]['bands'])
