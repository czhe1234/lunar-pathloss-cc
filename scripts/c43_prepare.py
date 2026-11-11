"""Freeze nested terrain subsets and a C38-comparable internal DEV panel."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'audit/c43_architecture_20261102';O.mkdir(exist_ok=True)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
official=list(csv.DictReader((R.parent/'data/val_index.csv').open()))
dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
rng=np.random.default_rng(20261102);by={}
for i,r in enumerate(rows):by.setdefault(r['terrain_id'],[]).append(i)
def groups(part):
    ts=[t for t in by if (t in dev)==(part=='dev')]
    single=np.array(sorted(t for t in ts if len(by[t])==1))
    multi=np.array(sorted(t for t in ts if len(by[t])==51))
    assert len(single)+len(multi)==len(ts)
    rng.shuffle(single);rng.shuffle(multi);return single,multi
fs,fm=groups('fit');ds,dm=groups('dev')
def ids(ts):return sorted(i for t in ts for i in by[t])
fit64=list(fs[:60])+list(fm[:4]);fit256=list(fs[:240])+list(fm[:16])
dev64=list(ds[:60])+list(dm[:4])
assert set(fit64).issubset(fit256) and not set(fit256)&set(dev64)
assert not (set(fit256)|set(dev64))&{r['terrain_id'] for r in official}
monitor_fit=np.array(ids(fit64));rng.shuffle(monitor_fit);monitor_fit=monitor_fit[:16].tolist()
monitor_dev=np.array(ids(dev64));rng.shuffle(monitor_dev);monitor_dev=monitor_dev[:32].tolist()
pix=rng.choice(65536,8192,replace=False)
p={'created':'2026-11-02','seed':20261102,
   'question':'Does high-resolution terrain encoding plus sparse path-guided sampling improve use of H/TX information?',
   'variants':{'cnn':'high-resolution multiscale CNN, no attention','grid':'same CNN plus sparse learnable-offset cross-attention at uniform global references','path':'identical grid parameters and initialization; reference positions follow TX-RX corridor and RX surrounding terrain'},
   'fit64_terrains':fit64,'fit256_terrains':fit256,'dev64_terrains':dev64,
   'fit64_ids':ids(fit64),'fit256_ids':ids(fit256),'dev64_ids':ids(dev64),
   'monitor_fit_ids':monitor_fit,'monitor_dev_ids':monitor_dev,'monitor_pixels':pix.tolist(),
   'official_VAL_used_for_training_selection':False,
   'baseline':'C38_1024_continue, trained on FIT only, saved complete DEV predictions',
   'stages':{'fit64':'64000 map presentations','fit256':'128000 additional map presentations, start from matched fit64 LAST weights; optimizer reset'},
   'points_per_map':2048,'loss':'valid-pixel masked MSE per band; mean of band losses; FSPL residual/20',
   'selection':'minimum mean of band RMSE on frozen monitor DEV; final score over all valid pixels of same DEV64 panel',
   'small_fit_gate':'Report fit/generalization curves; no <=1dB claim from small-set fit',
   'full_FIT_promotion':'At 256 terrains, promote if both bands improve >=0.10dB over CNN and scaling 64->256 improves both, OR both bands beat C38 on same full DEV64. Require finite/stable implementation and active spatial interaction.',
   'stop':'If no promotion, analyze fit/generation/mechanism before more training; do not promote a negative candidate to official VAL',
   'hardware':'UUID constrained physical 1/3/4; B and C on H100s, A on A800; benchmark common batch before training',
   'sources':['https://arxiv.org/abs/1908.07919','https://arxiv.org/abs/2010.04159'],
   'new_design_is_unvalidated':True}
# Score the identical DEV panel from frozen input-only C38 predictions.
mapping={}
for d in sorted((R/'runs/c38_1024_continue/predictions').glob('train_dev_c34_fresh_shard*')):
    ids0=json.loads((d/'indices.json').read_text())['index_rows']
    a=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
    for j,i in enumerate(ids0):mapping[i]=(a,j)
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
def score(wanted,pixels=None):
    s=np.zeros(2);c=np.zeros(2)
    for i in wanted:
        a,j=mapping[i]
        for b in range(2):
            m=np.unpackbits(M[b][i],bitorder='big').astype(bool)
            pr=np.asarray(a[b][j]).reshape(-1);yy=np.asarray(Y[b][i]).reshape(-1)
            if pixels is not None:m=m[pixels];pr=pr[pixels];yy=yy[pixels]
            e=pr[m].astype(float)-yy[m].astype(float);s[b]+=e@e;c[b]+=m.sum()
    rm=np.sqrt(s/c);return {'RMSE':rm.tolist(),'combined':float(rm.mean()),'SSE':s.tolist(),'count':c.tolist(),'maps':len(wanted)}
p['C38_DEV64']=score(p['dev64_ids'])
p['C38_monitor_DEV']=score(monitor_dev,pix)
path=O/'protocol.json'
assert not path.exists()
path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'fit64_maps':len(p['fit64_ids']),'fit256_maps':len(p['fit256_ids']),'DEV64_maps':len(p['dev64_ids']),'C38_DEV64':p['C38_DEV64'],'C38_monitor_DEV':p['C38_monitor_DEV']},ensure_ascii=False),flush=True)
(O/'PLAN.md').write_text('''# C43 architecture experiment\n\nUser authorization: try high-resolution multiscale terrain encoding and path-guided sparse attention. All work in CC; GPUs 1/3/4 by UUID only. Official VAL remains outside fitting and selection. Existing C38 weights and prediction files stay read-only.\n\nA/B/C: CNN only / uniform-reference sparse attention / path-reference sparse attention. B/C have identical parameterization and initial state. The intervention is reference-coordinate selection. All variants share data order, supervision points, optimizer schedule and checkpoint rule; hardware differences recorded. This is an independently implemented HRNet/Deformable-Attention-inspired design, not a reproduced published whole model or a validated propagation solver.\n\nFreeze terrain-stratified nested FIT64 and FIT256 plus disjoint DEV64, all from official TRAIN. Preserve every TX map of selected terrains. Pilot tests check pixel coordinates, finite gradients, nonzero offset/backbone gradients, spatial-feature sensitivity and actual throughput. Choose one batch from measured common candidates. Stage FIT64: 64000 map presentations. Continue each LAST model on FIT256 for 128000 additional map presentations, reset optimizer. Evaluate full-pixel DEV64 and fixed FIT panel, independently score saved predictions. The matched C38 reference is its internal-FIT model, not the full-TRAIN model that has seen these DEV labels.\n\nPromotion and stopping rules are in protocol.json. Positive pilot results do not complete the task. If there is no promotion, diagnose train fitting, scaling and feature use before spending a full-FIT budget. Main claim and full validation remain pending actual results.\n\nThe optional deepscientist experiment skill was inspected but its required bash_exec/artifact runtime is unavailable in this session. Execution uses the existing authorized CC SSH runner and durable JSON/source/log records.\n''')
(O/'CHECKLIST.md').write_text('- [x] Read CC rules and current C38/C21 records\n- [x] Freeze subsets, metric and baseline\n- [ ] Implement and verify encoder and sparse attention\n- [ ] Benchmark batch and run FIT64 A/B/C\n- [ ] Analyze FIT64 and run FIT256 matched continuation\n- [ ] Independently score and decide promotion\n- [ ] Report full results and limitations\n')
