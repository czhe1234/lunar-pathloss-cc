"""Freeze all near-source-coordinate pairs using only official TRAIN input metadata."""
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss'
rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
split={e['terrain']:e['group'] for e in json.load(open(R/'audit/c79_receiver/protocol.json'))['panel']}
terr=defaultdict(list)
for i,row in enumerate(rows):
    if split.get(row['terrain_id'])=='confirm':terr[row['terrain_id']].append(i)
pairs=[];pix={}
for name,ids in terr.items():
    unique={};coords={}
    for i in ids:
        pos=tuple(int(rows[i][k]) for k in ('tx_row','tx_col'));coords[i]=np.array(pos)
        if pos in unique:pairs.append(dict(terrain=name,i=unique[pos],j=i,separation=0.,group='identical'))
        else:unique[pos]=i
    uu=list(unique.values())
    near=[]
    for k,i in enumerate(uu):
        for j in uu[k+1:]:
            dist=float(np.linalg.norm(coords[i]-coords[j]))
            if 0<dist<=3:
                near.append(dict(terrain=name,i=i,j=j,separation=dist,group='near'))
    pairs.extend(near)
    if near:
        rng=np.random.default_rng([93460,int(name)]);farset=set()
        for row in near:
            i=row['i'];candidates=[j for j in uu if np.linalg.norm(coords[i]-coords[j])>=30 and tuple(sorted((i,j))) not in farset]
            if candidates:
                j=int(rng.choice(candidates));farset.add(tuple(sorted((i,j))));pairs.append(dict(terrain=name,i=i,j=j,separation=float(np.linalg.norm(coords[i]-coords[j])),group='far'))
for p in pairs:
    name=p['terrain']
    if name not in pix:
        pix[name]=np.sort(np.random.default_rng([93939,int(name)]).choice(65536,4096,replace=False)).tolist()
nodes=sorted({p[k] for p in pairs for k in ('i','j')})
(A/'tx_plan.json').write_text(json.dumps(dict(pairs=pairs,ids=nodes,pixels=pix,method='all unique nonzero <=3m pairs in the 484 confirmation terrains; 1 exact input duplicate as a check, matched >=30m queries when available; same 4096 uniform RX cells for both maps. Truth is used for paired diagnosis only. No neighbor TX label is used in the blind locator.'),separators=(',',':'))+'\n')
print(len(nodes),len(pairs),[(g,sum(p['group']==g for p in pairs)) for g in ('identical','near','far')])
