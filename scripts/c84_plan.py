"""Match unseen and seen FIT transmitters within the same terrain, before running new models."""
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c84_tx';A.mkdir(exist_ok=True)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
dev=set(json.load(open(R/'audit/internal_split.json'))['dev_terrain_ids']);old=json.load(open(R/'runs/c73_wide_real/protocol.json'))['fit'];oldset=set(old)
known=defaultdict(list);novel=defaultdict(list);singles=[]
for i,r in enumerate(rows):
    if r['terrain_id'] in dev:continue
    if r['group']=='single':
        if i in oldset:singles.append(i)
    elif i in oldset:known[r['terrain_id']].append(i)
    else:novel[r['terrain_id']].append(i)
rng=np.random.default_rng(84617);p_old=[];p_new=[];distance=[];newids=[]
def xy(i):return np.array([int(rows[i]['tx_row']),int(rows[i]['tx_col'])])
for t in sorted(known):
    seen={tuple(xy(i)):i for i in known[t]}
    extra={tuple(xy(i)):i for i in novel[t] if tuple(xy(i)) not in seen}
    oo=list(seen.values());nn=list(extra.values())
    if len(oo)<4 or len(nn)<2:continue
    rng.shuffle(oo);rng.shuffle(nn)
    for i in range(min(len(oo)//2,len(nn))):
        anchor,control=oo[2*i:2*i+2];offset=xy(control)-xy(anchor);dc=np.linalg.norm(offset)
        # Find the nearest fresh physical transmitter to the old partner. This keeps the same anchor, terrain and approximate direction/range.
        kk=min(range(len(nn)),key=lambda j:np.linalg.norm(xy(nn[j])-xy(control)))
        new=nn.pop(kk);dn=np.linalg.norm(xy(new)-xy(anchor))
        p_old.append([anchor,control]);p_new.append([anchor,new]);distance.append([dc,dn]);newids.append(new)
rng.shuffle(singles);single=np.array(singles[:max(1024,len(p_old))])
if len(single)%2:single=single[:-1]
pairs0=p_old+single.reshape(-1,2).tolist();pairs1=p_new+single.reshape(-1,2).tolist()
assert len(pairs0)==len(pairs1) and all(rows[i]['terrain_id'] not in dev for pair in pairs1 for i in pair)
assert len(set(newids))==len(newids) and not set(newids)&oldset
maps=np.array(distance)
for name,items in [('old',pairs0),('new',pairs1),('extra',sorted(newids))]:(A/(name+'.json')).write_text(json.dumps(items)+'\n')
cal=json.load(open(R/'audit/c79_receiver/protocol.json'))['panel'];keep={r['terrain'] for r in cal if r['group']=='select'};where=defaultdict(list)
for i,r in enumerate(rows):
    if r['terrain_id'] in keep:where[r['terrain_id']].append(i)
mon=[i for t in sorted(where) for i in where[t]]
checkpoint=R/'runs/c73_wide_real/best.pt'
with checkpoint.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
s=dict(hypothesis='C73 learned a real visibility graph but sampled only 35.4% multi-TX maps versus 77.7% of the official FIT images. Matched fresh TX exposure may improve illumination response. Conditional difference loss tests response to TX changes on identical terrain.',base='C76 research counterpart. Hold C72 fixed and replace only C73 with each checkpoint. Standalone C73 is also scored.',
    start_checkpoint_sha=h,multi_pairs=len(p_old),anchor_single_maps=len(single),maps_per_epoch=2*len(pairs0),multi_terrains=len({rows[p[0]]['terrain_id'] for p in p_old}),
    old_and_new_pair_distance_m_mean=maps.mean(0).tolist(),new_inputs=len(newids),monitor=mon,overlap_fit_confirm=False,
    treatments=['old original source pairs','new replacing only matched partner','delta same new pairs plus different-TX loss on jointly valid pixels'],
    design='All branches start at exactly the same frozen C73 weights. Same seed, pairs, single-TX anchors, batch size 8, optimizer, 6 epochs and update count. Tune epochs on 48 disjoint monitoring terrains (148 full maps). Confirm selected cross-arm choice on remaining 484 terrains (1934 full maps). No use of official VAL in fitting or choices.',
    gating='If the strongest arm improves BOTH full RMSE by >=0.08 dB over C76, with terrain-paired intervals positive, measure the complete branch on deployment and consider longer training. A major version still requires official improvements >=0.30 dB at both bands. If below gate stop this training line.')
(A/'protocol.json').write_text(json.dumps(s,indent=2)+'\n');print({k:s[k] for k in ('maps_per_epoch','new_inputs','multi_terrains','old_and_new_pair_distance_m_mean')})
