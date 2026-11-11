"""Remaining rare-tail fitting on the exact16maps; only loss weighting differs."""
import sys,json,time,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c105_context import create,reader,ResidualPanel
from src.c106_fit import evaluate
A=R/'audit/c107_tailfit'
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['control','tail8'],required=True);a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 plan=json.loads((A/'protocol.json').read_text());parent=R/plan['parent']
 import hashlib
 assert hashlib.sha256(parent.read_bytes()).hexdigest()==plan['parent_sha256']
 net=create().cuda().to(memory_format=torch.channels_last);ck=torch.load(parent,map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'],fused=True)
 panel=ResidualPanel(reader(),plan['fit16'],'context');x,b,y,w=panel.batch(slice(0,16));tail=(y-b)>=5
 out=R/'runs'/('c107_'+a.mode);out.mkdir(exist_ok=False);t=time.monotonic()
 config={**plan,'mode':a.mode,'GPU':gpu,'parameters':sum(p.numel() for p in net.parameters()),'input':'Same58channels;no labels/masks/ID ininput'}
 atomic_json(out/'config.json',config);initial=evaluate(net,x,b,y,w);atomic_json(out/'initial.json',initial)
 def passed(v):return v['passed'] and bool((np.array(v['residual_group_rmse'][1])<.95).all())
 def criterion(v):return max(np.max(v['max_map_band_rmse'])/.98,np.max(v['residual_group_rmse'][1])/.95)
 best=criterion(initial);beststep=0;history=[];gen=torch.Generator().manual_seed(plan['sampling_seed']);ok=False
 torch.save(dict(state=net.state_dict(),step=0,config=config),out/'best.pt')
 for step in range(1,plan['max_steps']+1):
  ix=torch.randperm(16,generator=gen)[:8].cuda()
  with torch.autocast('cuda',torch.bfloat16):p=b[ix]+net(x[ix]).float()
  e=torch.where(w[ix],p-y[ix],0.);weight=w[ix].float()
  if a.mode=='tail8':weight=weight*(1+7*tail[ix])
  loss=((e.square()*weight).sum((0,2,3))/weight.sum((0,2,3)).clamp_min(1)).mean()/25
  opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
  assert torch.isfinite(loss) and torch.isfinite(grad)
  if step%100==0 or step==plan['max_steps']:
   val=evaluate(net,x,b,y,w);crit=criterion(val)
   if crit<best:
    best=crit;beststep=step;torch.save(dict(state=net.state_dict(),step=step,config=config),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
   row=dict(step=step,fit=val,loss=float(loss),gradient_norm=float(grad),seconds=time.monotonic()-t,best_step=beststep,tailfit_passed=passed(val))
   history.append(row);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',row);print('FIT',a.mode,json.dumps(row),flush=True)
   torch.save(dict(state=net.state_dict(),optimizer=opt.state_dict(),step=step,order=gen.get_state(),config=config),out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
   if passed(val):
    ok=True;torch.save(dict(state=net.state_dict(),step=step,config=config),out/'first_pass.pt');break
 ck=torch.load(out/('first_pass.pt' if ok else 'best.pt'),map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 final=evaluate(net,x,b,y,w,save=out/'fit_prediction.npy')
 np.savez(out/'fit_truth.npz',target=y.cpu().numpy(),mask=w.cpu().numpy(),base=b.cpu().numpy(),mapid=np.array(plan['fit16']))
 result=dict(status='complete',passed=ok,selected_step=ck['step'],last_step=step,fit=final,selected_checkpoint='first_pass.pt' if ok else 'best.pt',
  seconds=time.monotonic()-t,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',a.mode,json.dumps(result),flush=True)
if __name__=='__main__':main()
