"""Independent CPU scoring and paired terrain uncertainty for the C21 control."""
import json,csv,time,hashlib
from pathlib import Path
import numpy as np
from c21_data import R,D
run=R/'runs/c21r1_pair_s20261123';out=R/'audit/c21_execution_20261101'
def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rmse(ss,cc):return np.sqrt(np.asarray(ss).sum(0)/np.asarray(cc).sum(0))
done=json.loads((run/'complete.json').read_text());assert done['status']=='paired_control_complete'
rows=list(csv.DictReader((D/'train_index.csv').open()));ids=json.loads((run/'dev_A/ids.json').read_text());assert ids==json.loads((run/'dev_B/ids.json').read_text())
anchor=R/'runs/c10_points_phase_geometry_s20261101/predictions/train_dev_gpu_c10_fast_fresh';assert ids==json.loads((anchor/'indices.json').read_text())['index_rows']
preds={k:np.load(run/f'dev_{k}/pred.npy',mmap_mode='r') for k in ['A','B']};anc=[np.load(anchor/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
yy=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];mm=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']];ph=np.load(R/'features/c21_phase/train.npy',mmap_mode='r')
N=len(ids);names=['C10','A','B'];sse=np.zeros((3,N,2));count=np.zeros((N,2));zs=np.zeros((3,N,3,2));zc=np.zeros((N,3,2));ds=np.zeros((3,N));dc=np.zeros(N);gs=np.zeros((3,N,2));gc=np.zeros((N,2));t0=time.time()
for j,i in enumerate(ids):
 y=np.stack([np.asarray(a[i],np.float64) for a in yy]);m=np.stack([np.unpackbits(a[i],bitorder='big').reshape(256,256).astype(bool) for a in mm]);count[j]=m.sum((1,2))
 emax=2*np.sinh(np.asarray(ph[i,:8],np.float32).max(0)*3);z=[emax<-.25,np.abs(emax)<=.25,emax>.25]
 common=m[0]&m[1];dc[j]=common.sum()
 for zi,v in enumerate(z):zc[j,zi]=(m&v).sum((1,2))
 gm=m[:,:,1:]&m[:,:,:-1];gc[j]=gm.sum((1,2))
 for k,name in enumerate(names):
  p=np.stack([a[j] for a in anc]) if name=='C10' else preds[name][j]
  e=p.astype(np.float64)-y;ee=e*e;sse[k,j]=(ee*m).sum((1,2));ds[k,j]=np.square(e[1]-e[0])[common].sum();gs[k,j]=(np.diff(e,axis=-1)**2*gm).sum((1,2))
  for zi,v in enumerate(z):zs[k,j,zi]=(ee*(m&v)).sum((1,2))
terr=sorted(set(rows[i]['terrain_id'] for i in ids));ti={t:j for j,t in enumerate(terr)};tss=np.zeros((3,len(terr),2));tcc=np.zeros((len(terr),2))
for j,i in enumerate(ids):k=ti[rows[i]['terrain_id']];tss[:,k]+=sse[:,j];tcc[k]+=count[j]
sample=np.random.default_rng(20261201).integers(len(terr),size=(2000,len(terr)));br=np.stack([np.sqrt(s[sample].sum(1)/tcc[sample].sum(1)) for s in tss])
result={'scope':'complete2562 DEV maps,612 held-out terrain IDs; official masked RMSE formula, not officialVAL','models':{},'comparisons':{},'bootstrap':'2000 paired resamples of terrain IDs; includes all TX maps of sampled terrain; conditional on this training seed, not seed uncertainty','seconds':time.time()-t0}
for k,name in enumerate(names):
 r=rmse(sse[k],count);result['models'][name]={'rmse':r.tolist(),'score':float(r.mean()),'sse':sse[k].sum(0).tolist(),'count':count.sum(0).astype(int).tolist(),'dual_band_difference_rmse_common_valid':float(np.sqrt(ds[k].sum()/dc.sum())),'horizontal_1m_error_gradient_rmse':rmse(gs[k],gc).tolist(),'zones':{zn:{'rmse':rmse(zs[k,:,zi],zc[:,zi]).tolist(),'sse':zs[k,:,zi].sum(0).tolist(),'count':zc[:,zi].sum(0).astype(int).tolist()} for zi,zn in enumerate(['clear','boundary','blocked'])}}
 if name!='C10':
  old=json.loads((run/f'dev_{name}/metrics.json').read_text());assert np.allclose(r,old['rmse'],atol=1e-8,rtol=0),(name,r,old['rmse']);result['models'][name]['independent_score_max_difference_db']=float(np.max(np.abs(r-np.asarray(old['rmse']))))
for target,reference in [('A','C10'),('B','C10'),('B','A')]:
 a=names.index(target);b=names.index(reference);delta=rmse(sse[a],count)-rmse(sse[b],count);bd=br[a]-br[b]
 result['comparisons'][f'{target}_minus_{reference}']={'delta_rmse_db':delta.tolist(),'bootstrap95ci_db':np.percentile(bd,[2.5,97.5],axis=0).T.tolist(),'delta_score_db':float(delta.mean()),'score_bootstrap95ci_db':np.percentile(bd.mean(1),[2.5,97.5]).tolist()}
eligible=[]
for name in ['A','B']:
 c=result['comparisons'][f'{name}_minus_C10'];passed=max(c['delta_rmse_db'])<=-.20 and max(v[1] for v in c['bootstrap95ci_db'])<0
 result['models'][name]['promotion_gate_passed']=bool(passed)
 if passed:eligible.append(name)
result['promoted_candidate']=min(eligible,key=lambda n:result['models'][n]['score']) if eligible else None
write(out/'r1_full_comparison.json',result);np.savez(out/'r1_paired_per_map.npz',ids=np.array(ids),sse=sse,count=count,zone_sse=zs,zone_count=zc)
print(json.dumps(result,ensure_ascii=False),flush=True)
