"""C105 confirmation, same saved2048position panel and independent FP64 scoring."""
import sys,json,time,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c105_context import create,reader,ResidualPanel,A
@torch.inference_mode()
def main():
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 frozen=json.loads((A/'selection_frozen.json').read_text());modes=['local','context'];names=['C82']+modes
 point=R/'audit/c101_residual/confirm';meta=json.loads((point/'done.json').read_text())
 ids=meta['ids'];rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));terrain=[r['terrain_id'] for i in ids for r in [rows[i]]];pp=2048
 target=np.load(point/'target.npy');y=target[:,:2].astype('float64');mask=target[:,2:4].astype(bool)
 base=np.load(point/'base.npy');pixels=np.load(point/'pixel.npy').reshape(len(ids),pp)
 assert len(ids)==1934 and len(set(terrain))==484 and len(y)==len(ids)*pp
 pred={'C82':base};data=None;t=time.monotonic()
 if frozen['nonzero_modes']:
  data=ResidualPanel(reader(),ids,'local');pixel=torch.from_numpy(pixels.astype('int64')).cuda()
 for mode in modes:
  c=frozen['choices'][mode];path=R/'runs'/('c105_'+mode)/'best.pt'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256']
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==c['epoch']
  if c['epoch']==0:pred[mode]=base.copy();continue
  net=create().cuda().to(memory_format=torch.channels_last);net.load_state_dict(ck['state']);net.eval();data.mode=mode
  out=np.empty_like(base)
  for start in range(0,len(ids),16):
   stop=min(start+16,len(ids));sl=slice(start,stop);x,p0,_,_=data.batch(sl)
   with torch.autocast('cuda',torch.bfloat16):p=p0+net(x).float()
   def take(q):return q.flatten(2).gather(2,pixel[sl,None,:].expand(-1,2,-1)).transpose(1,2).reshape(-1,2)
   assert np.array_equal(take(p0).cpu().numpy(),base[start*pp:stop*pp])
   out[start*pp:stop*pp]=take(p).cpu().numpy()
  np.save(A/(mode+'_confirm_prediction.npy'),out);pred[mode]=out;del net;torch.cuda.empty_cache()
 counts=mask.reshape(len(ids),pp,2).sum(1);sse=[];metrics={}
 for name in names:
  e=pred[name].astype('float64')-y;ss=np.where(mask,e*e,0).reshape(len(ids),pp,2).sum(1);sse.append(ss)
  metrics[name]=np.sqrt(ss.sum(0)/counts.sum(0)).tolist()
 sse=np.array(sse);unique=sorted(set(terrain));where={t:j for j,t in enumerate(unique)};ix=np.array([where[t] for t in terrain])
 ts=np.stack([np.stack([np.bincount(ix,weights=sse[k,:,b],minlength=len(unique)) for b in [0,1]],1) for k in range(3)])
 tc=np.stack([np.bincount(ix,weights=counts[:,b],minlength=len(unique)) for b in [0,1]],1)
 rng=np.random.default_rng(101097);draw=rng.integers(0,len(unique),(2000,len(unique)));den=tc[draw].sum(1);ci={}
 for lhs,rhs in [('C82','local'),('C82','context'),('local','context')]:
  z=np.sqrt(ts[names.index(lhs)][draw].sum(1)/den)-np.sqrt(ts[names.index(rhs)][draw].sum(1)/den)
  ci[lhs+'-minus-'+rhs]=np.quantile(z,[.025,.975],axis=0).T.tolist()
 gain={m:(np.array(metrics['C82'])-np.array(metrics[m])).tolist() for m in modes};eligible=frozen['eligible']
 promoted=eligible!='C82' and bool((np.array(gain[eligible])>=.3).all() and (np.array(ci['C82-minus-'+eligible])[:,0]>0).all())
 np.savez(A/'confirmation_sums.npz',names=np.array(names),mapid=np.array(ids),terrain=np.array(terrain),sse=sse,count=counts)
 result=dict(metrics=metrics,gain_vs_C82=gain,terrain_paired_95ci=ci,eligible=eligible,promote_to_full_native_check=promoted,
  scope='same484TRAINterrain/1934maps/2048uniformpositions permap; not officialVAL',valid_pixels=counts.sum(0).tolist(),
  confirmation_choices_frozen=True,independent_scoring='NumPyfloat64 onsavedfloat32predictions and rawTRAINlabels',
  seconds=time.monotonic()-t,GPU=gpu,official_val_used=False,website_submission=False)
 atomic_json(A/'summary.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
