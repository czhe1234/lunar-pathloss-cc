"""Define a terrain-disjoint model-selection split using training inputs/indices only."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json
import numpy as np
root=Path(__file__).resolve().parents[1]
source=root.parent/'data'/'train_index.csv'
rows=list(csv.DictReader(source.open()))
terrain_group={r['terrain_id']:r['group'] for r in rows}
rng=np.random.default_rng(20261031)
selected=[]
for group,n in [('multi',39),('single',573)]:
 pool=sorted(k for k,v in terrain_group.items() if v==group)
 selected.extend(str(x) for x in rng.choice(pool,n,replace=False))
selected=set(selected)
parts={phase:[r for r in rows if (r['terrain_id'] in selected)==(phase=='dev')] for phase in ['fit','dev']}
sets={k:{r['terrain_id'] for r in v} for k,v in parts.items()}
assert not sets['fit']&sets['dev']
assert len(parts['fit'])+len(parts['dev'])==len(rows)
assert len(selected)==612
out={'purpose':'Model selection on official training data only. Official validation data is excluded. This file defines a plan; no training has run.','seed':20261031,'source_train_index':str(source),'source_train_index_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'dev_terrain_ids':sorted(selected),'fit_rule':'All train_index rows whose terrain_id is not in dev_terrain_ids','summary':{k:{'samples_per_band':len(v),'terrains':len(sets[k]),'sample_groups':dict(Counter(r['group'] for r in v)),'terrain_groups':dict(Counter(terrain_group[t] for t in sets[k]))} for k,v in parts.items()}}
(root/'audit'/'internal_split.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out['summary'],ensure_ascii=False))
