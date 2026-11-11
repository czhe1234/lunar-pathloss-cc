"""Confirm on the same1934map2048point panel, with independent float64 scoring."""
import sys,json,time,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c104_train import Data,create,decoded,Sampler,NORM
A=R/'audit/c104_tail'
@torch.inference_mode()
def main():
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 frozen=json.loads((A/'selection_frozen.json').read_text());modes=['mse','state'];names=['C82']+modes
 source=R/'audit/c90_facet';folders=sorted(source.glob('confirm[0-9]'));metas=[json.loads((f/'done.json').read_text()) for f in folders]
 ids=sum([x['ids'] for x in metas],[]);terrain=sum([x['terrains'] for x in metas],[]);pp=2048
 target=np.concatenate([np.load(f/'targets.npy',mmap_mode='r').reshape(-1,6) for f in folders])
 y=target[:,:2].astype('float64');mask=target[:,2:4].astype(bool);base=target[:,4:6].copy();n=len(y)
 assert len(ids)==1934 and len(set(terrain))==484 and n==len(ids)*pp
 pred={'C82':base};nm=np.load(NORM);sm=Sampler();data=None;t=time.monotonic()
 if frozen['nonzero_modes']:
  assert (R/'audit/c101_residual/confirm/done.json').exists()
  data=Data('confirm','terrain')
  assert torch.equal(data.base.cpu(),torch.from_numpy(base))
 for mode in modes:
  c=frozen['choices'][mode];p=R/'runs'/('c104_'+mode)/'best.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256']
  ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['step']==c['step']
  if c['step']==0:
   pred[mode]=base.copy();continue
  net=create(mode,torch.tensor(nm['mean']),torch.tensor(nm['std']),sm.prior.cpu(),sm.proposal.cpu(),sm.means.cpu()).cuda();net.load_state_dict(ck['state']);net.eval()
  out=np.empty((n,2),np.float32)
  for start in range(0,n,32768):
   sl=slice(start,start+32768)
   with torch.autocast('cuda',torch.bfloat16):p=data.base[sl]+decoded(net(data.x(sl)),mode,sm.prior,sm.proposal)[0].mean(1)
   out[sl]=p.cpu().numpy()
  pred[mode]=out;np.save(A/(mode+'_confirm_prediction.npy'),out);del net;torch.cuda.empty_cache()
 counts=mask.reshape(len(ids),pp,2).sum(1);sse=[];metrics={}
 for name in names:
  e=pred[name].astype('float64')-y;ss=np.where(mask,e*e,0).reshape(len(ids),pp,2).sum(1);sse.append(ss)
  metrics[name]=np.sqrt(ss.sum(0)/counts.sum(0)).tolist()
 sse=np.array(sse);unique=sorted(set(terrain));where={t:j for j,t in enumerate(unique)};ix=np.array([where[t] for t in terrain])
 ts=np.stack([np.stack([np.bincount(ix,weights=sse[k,:,b],minlength=len(unique)) for b in [0,1]],1) for k in range(3)])
 tc=np.stack([np.bincount(ix,weights=counts[:,b],minlength=len(unique)) for b in [0,1]],1)
 rng=np.random.default_rng(101097);draw=rng.integers(0,len(unique),(2000,len(unique)));den=tc[draw].sum(1);ci={}
 for lhs,rhs in [('C82','mse'),('C82','state'),('mse','state')]:
  z=np.sqrt(ts[names.index(lhs)][draw].sum(1)/den)-np.sqrt(ts[names.index(rhs)][draw].sum(1)/den)
  ci[lhs+'-minus-'+rhs]=np.quantile(z,[.025,.975],axis=0).T.tolist()
 gain={m:(np.array(metrics['C82'])-np.array(metrics[m])).tolist() for m in modes};eligible=frozen['eligible']
 promoted=eligible!='C82' and bool((np.array(gain[eligible])>=.3).all() and (np.array(ci['C82-minus-'+eligible])[:,0]>0).all())
 np.savez(A/'confirmation_sums.npz',names=np.array(names),mapid=np.array(ids),terrain=np.array(terrain),sse=sse,count=counts)
 result=dict(metrics=metrics,gain_vs_C82=gain,terrain_paired_95ci=ci,eligible=eligible,promote_to_full_native_check=promoted,
  scope='same484TRAINterrain/1934maps/2048uniformpositions permap; sparse panel, not officialVAL',
  valid_pixels=counts.sum(0).tolist(),confirmation_choices_frozen=True,independent_scoring='NumPyfloat64 on savedfloat32predictions and original cachedTRAINlabels',
  seconds=time.monotonic()-t,GPU=gpu,official_val_used=False,website_submission=False)
 atomic_json(A/'summary.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
