"""Freeze both profile-network choices before opening confirmation labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from c97_profile_cpu import Net
R=Path(__file__).resolve().parents[1];A=R/'audit/c97_profile';D=R/'audit/c96_information'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
@torch.inference_mode()
def main():
 torch.set_num_threads(12);torch.set_num_interop_threads(1)
 choices={}
 for v in ['mean','strip']:
  result=json.loads((A/v/'result.json').read_text())
  assert result['status']=='complete'
  choices[v]=dict(best_step=result['best_step'],sha256=digest(A/v/'best.pt'))
 frozen=A/'selection_frozen.json'
 if frozen.exists():assert json.loads(frozen.read_text())==choices
 else:dump(frozen,choices)
 old=json.loads((D/'summary.json').read_text())
 dm=json.loads((D/'dev/done.json').read_text());n=dm['points']
 arrays={k:np.load(D/'dev'/(k+'.npy'),mmap_mode='r') for k in ['x','strip','y','mask','base','c82']}
 pred={'C38':np.array(arrays['base'],dtype='float64'),'C82':np.array(arrays['c82'],dtype='float64')}
 for v,c in choices.items():
  if c['best_step']==0:pred[v]=pred['C38'].copy();continue
  ck=torch.load(A/v/'best.pt',map_location='cpu',weights_only=False);st=ck['model']
  net=Net(v,st['mean'],st['std']);net.load_state_dict(st);net.eval()
  pp=np.empty((n,2),np.float64)
  for start in range(0,n,2048):
   sl=slice(start,start+2048)
   x=torch.from_numpy(np.array(arrays['x'][sl]));s=torch.from_numpy(np.array(arrays['strip'][sl]))
   base=torch.from_numpy(np.array(arrays['base'][sl]))
   pp[sl]=(base+20*net(x,s)).numpy()
  assert np.isfinite(pp).all();pred[v]=pp
  np.save(A/f'dev_prediction_{v}.npy',pp.astype('float32'))
 yy=np.array(arrays['y'],dtype='float64');mask=np.array(arrays['mask'],dtype=bool)
 group=np.repeat(dm['groups'],256)
 raw=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
 names=[raw[int(i)]['terrain_id'] for i in dm['maps']];unique=sorted(set(names));lookup={t:k for k,t in enumerate(unique)}
 terrain=np.repeat([lookup[t] for t in names],256)
 terrain_group={}
 for t,g in zip(names,dm['groups']):
  if t in terrain_group:assert terrain_group[t]==g
  terrain_group[t]=g
 counts=np.stack([np.bincount(terrain,weights=mask[:,b],minlength=len(unique)) for b in [0,1]],1)
 stats={};metrics={}
 for name,pp in pred.items():
  se=np.where(mask,(pp-yy)**2,0.)
  stats[name]=np.stack([np.bincount(terrain,weights=se[:,b],minlength=len(unique)) for b in [0,1]],1)
  metrics[name]={}
  for label,which in [('monitor',group!='confirm'),('confirmation',group=='confirm')]:
   ss=se[which].sum(0);cc=mask[which].sum(0)
   metrics[name][label]=dict(rmse=np.sqrt(ss/cc).tolist(),sse=ss.tolist(),count=cc.tolist())
 for panel in ['monitor','confirmation']:
  assert np.allclose(metrics['C82'][panel]['rmse'],old['metrics']['C82'][panel]['rmse'],rtol=0,atol=1e-10)
 ids=np.array([lookup[t] for t in unique if terrain_group[t]=='confirm'])
 rng=np.random.default_rng(970447);sample=rng.integers(0,len(ids),(2000,len(ids)));den=counts[ids][sample].sum(1)
 ci={}
 for lhs,rhs in [('C82','strip'),('mean','strip'),('C38','strip')]:
  a=stats[lhs][ids][sample].sum(1);b=stats[rhs][ids][sample].sum(1)
  g=np.sqrt(a/den)-np.sqrt(b/den)
  ci[lhs+'-minus-'+rhs]=np.quantile(g,[.025,.975],axis=0).T.tolist()
 gain=np.array(metrics['C82']['confirmation']['rmse'])-np.array(metrics['strip']['confirmation']['rmse'])
 lower=np.array(ci['C82-minus-strip'])[:,0]
 promoted=bool((gain>=.30).all() and (lower>0).all())
 result=dict(metrics=metrics,confirmation_gain_vs_C82=gain.tolist(),paired_terrain_95ci=ci,
  expand=promoted,criterion='each confirmation band>=0.30dB gain againstC82 and positive pairedterrain95CI',
  fit_terrains=5508,monitor_terrains=128,confirmation_terrains=484,points_per_map=256,
  choices=choices,official_val_used=False,confirmation_labels_used_in_selection=False,
  limitation='Same historical TRAIN-heldout terrains used by C82/C96. Sparse sample, not official fixedVAL or an input information lowerbound.')
 dump(A/'summary.json',result);np.savez(A/'per_terrain_sums.npz',terrain=np.array(unique),counts=counts,**stats)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
