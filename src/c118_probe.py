import sys,json,time,argparse,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c118_location import AuxField
from src.c105_context import reader,ResidualPanel
from src.c106_metrics import average_precision
A=R/'audit/c118_location'
@torch.inference_mode()
def main():
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 P=R/'audit/c101_residual/monitor';meta=json.loads((P/'done.json').read_text());ids=meta['ids'];pp=meta['points_per_map']
 d=ResidualPanel(reader(),ids,'context');px=torch.tensor(np.load(P/'pixel.npy').reshape(len(ids),pp),device='cuda',dtype=torch.int64)
 raw=np.load(P/'target.npy');base=np.load(P/'base.npy');res=raw[:,:2]-base;mask=raw[:,2:4].astype(bool)
 allresult={};start=time.monotonic()
 for mode in ['plain','location']:
  path=R/'runs'/('c118_'+mode)/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False);net=AuxField().cuda().to(memory_format=torch.channels_last);net.load_state_dict(ck['state']);net.eval()
  prior=torch.tensor(ck['config']['prior'],device='cuda');prob=[];balanced=[]
  for st in range(0,len(ids),16):
   sl=slice(st,min(st+16,len(ids)));x,b,y,w=d.batch(sl)
   with torch.autocast('cuda',torch.bfloat16):delta,logit=net.outputs(x)
   rawp=logit.float().softmax(2)
   p=(logit.float()+prior.log()[None,:,:,None,None]).softmax(2)
   def take(v):return v.flatten(3).gather(3,px[sl,None,None].expand(-1,2,3,-1)).permute(0,3,1,2).reshape(-1,2,3).cpu().numpy()
   prob.append(take(p));balanced.append(take(rawp))
  prob=np.concatenate(prob);rawp=np.concatenate(balanced);np.save(A/(mode+'_monitor_probability.npy'),prob)
  reports=[]
  for band in range(2):
   valid=mask[:,band];rr=res[valid,band];p=prob[valid,band];rp=rawp[valid,band];cls=np.where(rr>=5,2,np.where(rr<=-5,0,1));tail=cls!=1;pos=cls==2
   bc=float(np.mean([-np.log(np.maximum(rp[cls==k,k],1e-8)).mean() for k in range(3)]))
   order=np.argsort(-p[:,2]);top={}
   for f in [.01,.03,.10]:
    ix=order[:int(np.ceil(f*len(order)))];top[str(f)]={'weak_precision':float(pos[ix].mean()),'negative_precision':float((cls[ix]==0).mean()),'abs5_precision':float(tail[ix].mean()),'mean_residual':float(rr[ix].mean()),'residual_RMS':float(np.sqrt(np.mean(rr[ix].astype('float64')**2)))}
   pred=p[:,2]>=.5
   reports.append(dict(valid=int(valid.sum()),balancedCE=bc,positive_AP=average_precision(p[:,2],pos),negative_AP=average_precision(p[:,0],cls==0),abs5_AP=average_precision(p[:,0]+p[:,2],tail),
    positive_prevalence=float(pos.mean()),abs5_prevalence=float(tail.mean()),precision_at_probability_half=float(pos[pred].mean()) if pred.any() else None,recall_at_probability_half=float((pos&pred).sum()/pos.sum()),top=top,
    sign_accuracy_given_true_abs5=float(np.mean((p[tail,2]>p[tail,0])==(cls[tail]==2))),sign_majority_given_true_abs5=float(max((cls[tail]==2).mean(),(cls[tail]==0).mean()))))
  allresult[mode]={'epoch':ck['epoch'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bands':reports}
  print(mode,reports,flush=True);del net;torch.cuda.empty_cache()
 atomic_json(A/'location_probe.json',dict(results=allresult,GPU=gpu,seconds=time.monotonic()-start,scope='128TRAINmonitor sampled2048points;diagnostic only; no officialVAL',probability='softmax of training-balanced logits plus log FIT class prior; no threshold fitting'))
if __name__=='__main__':main()
