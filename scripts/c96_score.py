"""Freeze readout choices, then compare all candidates on identical held-out pixels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from c96_train import load,features
R=Path(__file__).resolve().parents[1];A=R/'audit/c96_information'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 choices={}
 for v in ['old','strip']:
  choices[v]=[]
  for b in [0,1]:
   p=A/f'{v}_band{b}'
   result=json.loads((p/'result.json').read_text())
   choices[v].append(dict(band=b,selected_iteration=result['selected_iteration'],model_sha256=sha(p/'model.txt')))
 freeze=A/'selection_frozen.json'
 if freeze.exists():assert json.loads(freeze.read_text())==choices
 else:dump(freeze,choices)
 d=load('dev',True);n=len(d['x']);dm=json.loads((A/'dev/done.json').read_text())
 yy=np.array(d['y'],dtype='float64');mask=np.array(d['mask'],dtype=bool)
 base=np.array(d['base'],dtype='float64')
 pred={'C38':base,'C82':np.array(np.load(A/'dev/c82.npy',mmap_mode='r'),dtype='float64')}
 for v in choices:
  pp=base.copy()
  for b,c in enumerate(choices[v]):
   if c['selected_iteration']:
    model=lgb.Booster(model_file=str(A/f'{v}_band{b}/model.txt'))
    for start in range(0,n,16384):
     sl=slice(start,min(n,start+16384));xx=features(d,sl,v=='strip')
     pp[sl,b]+=model.predict(xx,num_iteration=c['selected_iteration'],num_threads=8)
  assert np.isfinite(pp).all()
  pred[v]=pp
  np.save(A/f'dev_prediction_{v}.npy',pp.astype('float32'))
 rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
 terrain_names=[rows[int(i)]['terrain_id'] for i in dm['maps']]
 unique=sorted(set(terrain_names));lookup={t:k for k,t in enumerate(unique)}
 terrain=np.repeat([lookup[t] for t in terrain_names],256)
 group=np.repeat(dm['groups'],256)
 metrics={}
 perterrain={}
 for name,pp in pred.items():
  se=np.where(mask,(pp-yy)**2,0.)
  perterrain[name]=np.stack([np.bincount(terrain,weights=se[:,b],minlength=len(unique)) for b in [0,1]],1)
  metrics[name]={}
  for panel,which in [('monitor',group!='confirm'),('confirmation',group=='confirm'),('all_dev',np.ones(n,bool))]:
   num=se[which].sum(0);den=mask[which].sum(0)
   metrics[name][panel]=dict(rmse=np.sqrt(num/den).tolist(),sse=num.tolist(),count=den.tolist())
 ct=np.array([t for t in unique if any(g=='confirm' for tt,g in zip(terrain_names,dm['groups']) if tt==t)])
 ids=np.array([lookup[t] for t in ct])
 counts=np.stack([np.bincount(terrain,weights=mask[:,b],minlength=len(unique)) for b in [0,1]],1)
 rng=np.random.default_rng(960427); draws=rng.integers(0,len(ids),(2000,len(ids)))
 ci={}
 for lhs,rhs in [('C82','strip'),('old','strip'),('C38','strip')]:
  s1=perterrain[lhs][ids][draws].sum(1);s2=perterrain[rhs][ids][draws].sum(1);den=counts[ids][draws].sum(1)
  gains=np.sqrt(s1/den)-np.sqrt(s2/den)
  ci[lhs+'-minus-'+rhs]=dict(terrain_paired_95ci=np.quantile(gains,[.025,.975],axis=0).T.tolist(),bootstrap_draws=2000,terrains=len(ids))
 gain=np.array(metrics['C82']['confirmation']['rmse'])-np.array(metrics['strip']['confirmation']['rmse'])
 positive=np.array(ci['C82-minus-strip']['terrain_paired_95ci'])[:,0]>0
 promoted=bool(np.all(gain>=.30)&np.all(positive))
 result=dict(metrics=metrics,confirmation_gain_vs_C82=gain.tolist(),paired_bootstrap=ci,
  expanded_training_authorized_by_result=promoted,criterion='both bands >=0.30 dB better than C82 on same confirmation panel, with positive paired-terrain 95% lower bound',
  official_val_used=False,selection_frozen_before_confirmation_scoring=True,points_per_map=256,fit_terrains=5508,monitor_terrains=128,confirmation_terrains=484,
  warning='This is a bounded sparse-point information screen. Failure is not proof that terrain/TX lack predictive information.')
 dump(A/'summary.json',result)
 np.savez(A/'per_terrain_sums.npz',terrain=np.array(unique),counts=counts,**perterrain)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
