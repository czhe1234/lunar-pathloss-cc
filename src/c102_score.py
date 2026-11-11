"""Freeze choices then score the full2048point heldout panel. Zero-head models reuse exact parent."""
import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c102_train import load_encoders
from src.c102_data import Bank
from src.c102_transport import Transport
A=R/'audit/c102_transport'
@torch.inference_mode()
def main():
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 choices={};means={};initial=None
 for mode in ['coarse','native']:
  d=R/'runs'/('c102_'+mode);r=json.loads((d/'result.json').read_text());assert r['status']=='complete'
  choices[mode]=dict(step=r['best_step'],sha256=hashlib.sha256((d/'best.pt').read_bytes()).hexdigest())
  means[mode]=float(np.mean(r['monitor']['rmse']));initial=json.loads((d/'initial.json').read_text())
 means['C82']=float(np.mean(initial['C82']));frozen=dict(choices=choices,means=means,eligible=min(means,key=means.get))
 p=A/'selection_frozen.json'
 if p.exists():assert json.loads(p.read_text())==frozen
 else:atomic_json(p,frozen)
 folders=sorted((R/'audit/c90_facet').glob('confirm[0-9]'));meta=[json.loads((p/'done.json').read_text()) for p in folders]
 ids=sum([p['ids'] for p in meta],[]);terrains=sum([p['terrains'] for p in meta],[])
 target=np.concatenate([np.load(p/'targets.npy',mmap_mode='r').reshape(-1,6) for p in folders]);pp=2048;nmap=len(ids)
 base=target[:,4:6];parent=np.load(R/'audit/c101_residual/coarse_confirm_prediction.npy')
 pred={'C82':base,'parent':parent};bank=None;t=time.monotonic()
 if any(c['step'] for c in choices.values()):
  point,source=load_encoders();bank=Bank('confirm',source);del source;assert bank.ids==ids;torch.cuda.empty_cache()
 for mode in choices:
  ck=torch.load(R/'runs'/('c102_'+mode)/'best.pt',map_location='cpu',weights_only=False)
  assert ck['step']==choices[mode]['step']
  if not ck['step']:
   assert torch.count_nonzero(ck['state']['head.3.weight'])==0 and torch.count_nonzero(ck['state']['head.3.bias'])==0
   pred[mode]=np.array(parent);continue
  net=Transport().cuda();net.load_state_dict(ck['state']);net.eval();scale=4 if mode=='coarse' else 1
  out=np.empty((nmap,pp,2),np.float32)
  for start in range(0,nmap,8):
   maps=torch.arange(start,min(nmap,start+8),device='cuda')
   for sub in range(0,pp,128):
    pos=torch.arange(sub,min(pp,sub+128),device='cuda')[None].expand(len(maps),-1)
    q,n,bias,mass,p0,tt,w=bank.batch(maps,pos,scale,point)
    with torch.autocast('cuda',torch.bfloat16):v=p0+net(q,n,bias,mass)
    out[start:start+len(maps),sub:sub+128]=v.cpu().numpy()
   if start%256==0:print('confirm',mode,start+len(maps),nmap,'seconds',time.monotonic()-t,flush=True)
  pred[mode]=out.reshape(-1,2);np.save(A/(mode+'_confirm_prediction.npy'),pred[mode]);del net;torch.cuda.empty_cache()
 names=['C82','parent','coarse','native'];mask=target[:,2:4].astype(bool);count=mask.reshape(nmap,pp,2).sum(1)
 se=[];metrics={}
 for name in names:
  err=pred[name].astype('float64')-target[:,:2]
  ss=np.where(mask,err*err,0.).reshape(nmap,pp,2).sum(1);se.append(ss);metrics[name]=np.sqrt(ss.sum(0)/count.sum(0)).tolist()
 se=np.array(se);unique=sorted(set(terrains));ix=np.array([unique.index(t) for t in terrains]);ss=np.zeros((4,len(unique),2));cc=np.zeros((len(unique),2))
 for b in [0,1]:
  cc[:,b]=np.bincount(ix,weights=count[:,b],minlength=len(unique))
  for k in range(4):ss[k,:,b]=np.bincount(ix,weights=se[k,:,b],minlength=len(unique))
 rng=np.random.default_rng(102097);draw=rng.integers(0,len(unique),(2000,len(unique)));den=cc[draw].sum(1);ci={}
 for left,right in [('C82','coarse'),('C82','native'),('parent','coarse'),('parent','native'),('coarse','native')]:
  gains=np.sqrt(ss[names.index(left)][draw].sum(1)/den)-np.sqrt(ss[names.index(right)][draw].sum(1)/den)
  ci[left+'-minus-'+right]=np.quantile(gains,[.025,.975],axis=0).T.tolist()
 gain={m:(np.array(metrics['C82'])-metrics[m]).tolist() for m in ['coarse','native']};eligible=frozen['eligible']
 promote=eligible!='C82' and bool((np.array(gain[eligible])>=.3).all() and (np.array(ci['C82-minus-'+eligible])[:,0]>0).all())
 np.savez(A/'confirmation_sums.npz',names=np.array(names),ids=np.array(ids),terrain=np.array(terrains),sse=se,count=count)
 result=dict(metrics=metrics,gain_vs_C82=gain,paired_terrain95ci=ci,eligible=eligible,promote_to_full_native_check=promote,
  confirmation_maps=nmap,confirmation_terrains=len(unique),points_per_map=pp,valid_pixels=count.sum(0).tolist(),
  choices_frozen=True,official_val_used=False,GPU=gpu,seconds=time.monotonic()-t)
 atomic_json(A/'summary.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
