import json,csv,hashlib
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c127_joint'
old=json.loads((R/'audit/c105_context/protocol.json').read_text());rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
groups=defaultdict(list)
for j,i in enumerate(old['fit']):groups[rows[i]['terrain_id']].append(j)
assert Counter(map(len,groups.values()))=={1:1707,4:277}
batch=16;epochs=24;units=list(groups.values());orders=[]
for ep in range(epochs):
 rng=np.random.default_rng(127073+ep);seq=[];buf=[]
 for u in rng.permutation(len(units)):
  items=np.array(units[u])[rng.permutation(len(units[u]))].tolist()
  if len(buf)+len(items)>batch:seq.append(buf);buf=[]
  buf.extend(items)
 if buf:seq.append(buf)
 assert sorted(sum(seq,[]))==list(range(len(old['fit'])))
 orders.append(seq)
rng=np.random.default_rng(127101)
multi=[v for v in units if len(v)==4];single=[v for v in units if len(v)==1]
fit_eval=sum([multi[i] for i in rng.permutation(len(multi))[:32]],[])+sum([single[i] for i in rng.permutation(len(single))[:64]],[])
plan=dict(fit=old['fit'],monitor=old['monitor'],fit_eval_positions=fit_eval,
 batch=batch,max_epochs=epochs,min_epochs=8,patience=6,initial_seed=127041,
 lr=.0003,weight_decay=.01,aux_component=.125,aux_band=.25,
 orders=orders,groups={k:v for k,v in groups.items()},group_sizes=dict(Counter(map(len,groups.values()))),
 input='56H/TXfeatures;noC82channel,nootherTXlabels,noIDembedding,no masks',
 arms=['residual_independent','residual_joint','direct_independent','direct_joint'],
 scope='same C101 TRAIN only; no officialVAL; monitor used historically',
 promotion=dict(monitor_gain_each=.20,confirm_gain_each=.30,paired_ci_lower=0))
(A/'protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in plan.items() if k not in ['fit','monitor','groups','orders','fit_eval_positions']},ensure_ascii=False))
print('batches',min(map(len,orders)),max(map(len,orders)),'fit_eval',len(fit_eval))
