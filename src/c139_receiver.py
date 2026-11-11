"""Paired CPU diagnostic: unpooled receiver donor states versus pooled states."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,time,argparse,hashlib
import numpy as np
from catboost import CatBoostRegressor,Pool
R=Path(__file__).resolve().parents[1];A=R/'audit/c139_bottleneck';C=R/'audit/c136_native'
def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def build():
 plan=json.loads((C/'protocol.json').read_text())
 lookup=np.load(R/'src/c70_centroid_lookup0.npz');wi=np.maximum(lookup['weights'],0);wi=wi/wi.sum(1,keepdims=True);fi=lookup['ids']
 rawrows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
 verify=[]
 for part in ['fit','monitor']:
  out=A/part;out.mkdir(exist_ok=False)
  P=R/'audit/c101_residual'/part;m=json.loads((P/'done.json').read_text());pp=m['points_per_map'];oldindex={i:j for j,i in enumerate(m['ids'])};ids=plan[part];n=len(ids)*pp
  old=np.load(P/'common.npy',mmap_mode='r').reshape(-1,pp,121);precise=np.load(P/'precise.npy',mmap_mode='r').reshape(-1,pp,48)
  bb=np.load(P/'base.npy',mmap_mode='r').reshape(-1,pp,2);tt=np.load(P/'target.npy',mmap_mode='r').reshape(-1,pp,4);ww=np.load(P/'weight.npy',mmap_mode='r').reshape(-1,pp)
  common=np.lib.format.open_memmap(out/'common.npy','w+',dtype='float32',shape=(n,215))
  donor=np.lib.format.open_memmap(out/'donor.npy','w+',dtype='float32',shape=(n,3,17))
  targets=np.lib.format.open_memmap(out/'target.npy','w+',dtype='float32',shape=(n,4))
  base=np.lib.format.open_memmap(out/'base.npy','w+',dtype='float32',shape=(n,2))
  weight=np.lib.format.open_memmap(out/'weight.npy','w+',dtype='float32',shape=(n,))
  geom=np.lib.format.open_memmap(out/'weights.npy','w+',dtype='float32',shape=(n,3))
  pixels=np.lib.format.open_memmap(out/'pixels.npy','w+',dtype='int32',shape=(n,))
  start=time.monotonic()
  for j,i in enumerate(ids):
   k=oldindex[i];sl=slice(j*pp,(j+1)*pp)
   with np.load(C/'cache'/f'{i}.npz') as z:
    px=z['pixels'];inv=z['inv'];uf=z['uf'];df=z['df'].astype('float32');dw=z['dw'];pw=z['pw'];active=z['active']
   assert np.array_equal(uf[inv],fi[px])
   w=wi[px].astype('float32')
   # Each relative donor power is normalized by total pixel power; no label enters.
   relative=np.log10(np.maximum(pw,1e-20)/np.maximum(w[:,:,None],1e-20)).clip(-12,12)
   val=np.concatenate([df[inv],w[:,:,None],relative,dw[inv]],2)
   assert val.shape==(pp,3,17)
   with np.load(R/'audit/c135_paths/cache'/f'{i}.npz') as z:power=z['power']
   total=power.sum(1,keepdims=True)
   fields=np.concatenate([np.log10(np.maximum(power,1e-30)).reshape(pp,-1)/10,
    (power/np.maximum(total,1e-30)).reshape(pp,-1)],1)
   common[sl]=np.concatenate([old[k],precise[k],bb[k]/100,fields],1)
   donor[sl]=val;targets[sl]=tt[k];base[sl]=bb[k];weight[sl]=ww[k] if part=='fit' else 1.;geom[sl]=w;pixels[sl]=px
   if j%100==0:print('BUILD',part,j,len(ids),round(time.monotonic()-start,2),flush=True)
  for x in [common,donor,targets,base,weight,geom,pixels]:x.flush()
  save(out/'meta.json',dict(ids=ids,points_per_map=pp,terrains=[rawrows[i]['terrain_id'] for i in ids],features_common=215,features_donor=51,labels_or_masks_in_inputs=False))
  # Independently compare stored targets/masks to the release, including endpoints.
  for band,b in [('415',0),('58',1)]:
   raw=np.load(R.parent/'data'/f'train_pl{band}.npy',mmap_mode='r');mask=np.load(R.parent/'data'/f'train_mask{band}.npy',mmap_mode='r')
   md=0;mm=0
   for j in [0,len(ids)//2,len(ids)-1]:
    sl=slice(j*pp,(j+1)*pp);px=pixels[sl];i=ids[j]
    expected=raw[i].reshape(-1)[px].astype('float32');vv=np.unpackbits(mask[i],bitorder='big')[px]
    md=max(md,float(np.max(abs(targets[sl,b]-expected))));mm+=int(np.count_nonzero(targets[sl,b+2]!=vv))
   assert md==0 and mm==0,(part,band,md,mm)
   verify.append(dict(part=part,band=band,target_max_error=md,mask_mismatches=mm))
  print('BUILT',part,'seconds',time.monotonic()-start,flush=True)
 save(A/'data_check.json',dict(status='passed',release_target_check=verify,inputs_from_H_TX_and_frozen_C82_only=True,official_VAL_used=False))
def data(part,mode):
 p=A/part;meta=json.loads((p/'meta.json').read_text())
 common=np.load(p/'common.npy',mmap_mode='r');dd=np.load(p/'donor.npy',mmap_mode='r');ww=np.load(p/'weights.npy',mmap_mode='r')
 x=np.empty((len(common),266),np.float32);x[:,:215]=common
 if mode=='pooled':
  avg=(dd*ww[:,:,None]).sum(1);x[:,215:]=np.repeat(avg[:,None,:],3,axis=1).reshape(-1,51)
 else:x[:,215:]=dd.reshape(-1,51)
 assert np.isfinite(x).all()
 return dict(x=x,target=np.load(p/'target.npy'),base=np.load(p/'base.npy'),weight=np.load(p/'weight.npy'),meta=meta)
def stats(d,p):
 y=d['target'][:,:2].astype('float64');v=d['target'][:,2:]>0;base=d['base'].astype('float64');e=np.where(v,p-y,0);r=np.where(v,base-y,0);n=v.sum(0);tail=v&(abs(r)>=5);pp=d['meta']['points_per_map'];s=(e*e).reshape(-1,pp,2).sum(1);nn=v.reshape(-1,pp,2).sum(1)
 rm=np.sqrt(s.sum(0)/n);old=np.sqrt((r*r).sum(0)/n)
 return dict(RMSE=rm.tolist(),C82=old.tolist(),gain=(old-rm).tolist(),valid_count=n.tolist(),tail_RMSE=np.sqrt((e*e*tail).sum(0)/tail.sum(0)).tolist(),tail_sign_accuracy=(((p-base)*(y-base)>0)&tail).sum(0).__truediv__(tail.sum(0)).tolist()),np.c_[s,nn]
def train(mode):
 start=time.monotonic();cfg=json.loads((A/'receiver_protocol.json').read_text());out=R/'runs'/f'c139_{mode}';out.mkdir(exist_ok=False)
 fit=data('fit',mode);mon=data('monitor',mode);preds={k:d['base'].astype('float64').copy() for k,d in [('fit',fit),('monitor',mon)]};trace=[]
 save(out/'config.json',dict(cfg,mode=mode,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 for b in range(2):
  v=fit['target'][:,b+2]>0
  model=CatBoostRegressor(**cfg['optimizer'],train_dir=str(out/f'logs{b}'),allow_writing_files=True)
  pool=Pool(fit['x'][v],fit['target'][v,b]-fit['base'][v,b],weight=fit['weight'][v])
  model.fit(pool,verbose=64);model.save_model(str(out/f'band{b}.cbm'))
  for part,d in [('fit',fit),('monitor',mon)]:
   pred=model.predict(d['x'],thread_count=2).astype('float32');preds[part][:,b]+=pred
  for step in [64,128,256]:
   p=model.predict(mon['x'],ntree_end=step,thread_count=2);vm=mon['target'][:,b+2]>0;res=mon['target'][:,b]-mon['base'][:,b]
   trace.append(dict(band=b,trees=step,RMSE=float(np.sqrt(np.mean((p[vm].astype('float64')-res[vm])**2)))))
  print('BAND_DONE',mode,b,round(time.monotonic()-start,2),trace[-3:],flush=True)
 result=dict(status='complete',mode=mode,scope='same fixed TRAIN monitor sampled points, not official VAL',seconds=time.monotonic()-start,trace=trace,CPU_threads=2,GPU_used=False)
 for part,d in [('fit',fit),('monitor',mon)]:
  result[part],bymap=stats(d,preds[part]);np.save(out/f'{part}_prediction.npy',preds[part].astype('float32'));np.save(out/f'{part}_per_map.npy',bymap)
 save(out/'done.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('--mode',choices=['pooled','donors']);a=p.parse_args()
 if a.build:build()
 else:train(a.mode)
