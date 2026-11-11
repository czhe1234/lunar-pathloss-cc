"""C27 matched profile-resolution intervention on the fixed128 FIT panel."""
from cc_core import ROOT,atomic_json,check_source_unchanged
import argparse,json,hashlib,time,math,shutil
import numpy as np
import torch
from c21_train_r1 import setup,model as parent_model,gather,digest_state
from c21_pair_r1 import loader,gpu_batch,full_dev
from c22_train import monitor
from c27_model import Model
A=ROOT/"audit/c27_path_representation_20261101"
def main(args):
 p=json.loads((A/"protocol.json").read_text());gpu=setup(p["seed"]);check_source_unchanged()
 run=ROOT/"runs"/("c27_fit128_"+args.variant+("_smoke" if args.smoke else ""));run.mkdir(exist_ok=False);(run/"source").mkdir()
 n=np.load(ROOT/"audit/c21_execution_20261101/normalization.npz")
 net=Model(torch.from_numpy(n["mean"]),torch.from_numpy(n["std"]),args.variant).cuda()
 parent=ROOT/p["parent"];assert hashlib.sha256(parent.read_bytes()).hexdigest()==p["parent_sha256"]
 ck=torch.load(parent,map_location="cpu",weights_only=False);missing,unexpected=net.load_state_dict(ck["model"],strict=False)
 assert not unexpected and missing and all(k.startswith("path_encoder.") for k in missing)
 cfg={"protocol":p,"variant":args.variant,"gpu":gpu,"parameters":sum(v.numel() for v in net.parameters()),
      "initial_state_sha256":digest_state(net),"optimizer":"fresh matched AdamW","smoke":args.smoke}
 for name in ["c27_train.py","c27_model.py","c22_train.py","cc_core.py","c21_model.py","c21_model_r1.py","c21_data.py","c21_train_r1.py","c21_pair_r1.py"]:shutil.copy2(ROOT/"src"/name,run/"source"/name)
 atomic_json(run/"config.json",cfg)
 if args.smoke:
  old=parent_model(False);old.load_state_dict(ck["model"])
  x,tx,fs,y,m=gpu_batch(next(iter(loader(p["fit_panel"][:2],2,0))));pix=torch.arange(1024,device="cuda")[None].expand(2,-1)
  with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
   oldp=old(x,tx,pix,512);newp=net(x,tx,pix,512)
  delta=float((oldp-newp).abs().max())*20;assert delta<1e-5,delta
  atomic_json(run/"initial_prediction_check.json",{"max_difference_db":delta,"pixels":2048})
  del old,x,tx,fs,y,m,oldp,newp;torch.cuda.empty_cache()
 del ck
 opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=p["lr"],weight_decay=1e-5)
 steps=2 if args.smoke else p["steps"];ids=np.array(p["fit_panel"]);rng=np.random.default_rng(p["seed"])
 order=np.concatenate([rng.permutation(ids) for _ in range(math.ceil(steps*8/len(ids)))])[:steps*8]
 np.save(run/"map_order.npy",order);pixels=np.load(ROOT/"runs/c21r1_pair_s20261123/monitor_pixels.npy")
 gen=torch.Generator().manual_seed(p["seed"]+1);t0=time.time();log=(run/"history.jsonl").open("a",buffering=1);qh=hashlib.sha256()
 def emit(r):
  r["seconds"]=time.time()-t0;print(json.dumps(r),flush=True);log.write(json.dumps(r)+"\n");atomic_json(run/"progress.json",r)
 def evaluate(step):
  fi=p["fit_panel"][:4] if args.smoke else p["fit_panel"];di=p["dev_panel"][:4] if args.smoke else p["dev_panel"]
  fit=monitor(net,fi,pixels);dev=monitor(net,di,pixels);row={"event":"monitor","step":step,"FIT":fit,"DEV":dev}
  emit(row);atomic_json(run/"latest_eval.json",row)
  torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),"step":step,"metrics":row,"config":cfg},run/"last.pt")
 evaluate(0)
 for step,batch in enumerate(loader(order,8,6),1):
  x,tx,fs,y,m=gpu_batch(batch);pix_cpu=torch.randint(65536,(len(x),2048),generator=gen);qh.update(pix_cpu.numpy().tobytes());pix=pix_cpu.cuda()
  lr=p["lr"]*min(step/p["warmup"],1)*(.1+.9*.5*(1+math.cos(math.pi*step/p["steps"])))
  for group in opt.param_groups:group["lr"]=lr
  opt.zero_grad(set_to_none=True)
  with torch.autocast("cuda",dtype=torch.bfloat16):pred=net(x,tx,pix,512)*20+gather(fs,pix)
  mm=gather(m,pix);loss=(((pred-gather(y,pix)).square()*mm).sum((0,1))/mm.sum((0,1)).clamp_min(1)).mean()/400
  if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
  if not torch.isfinite(grad):raise RuntimeError("nonfinite gradient")
  branch_norm=math.sqrt(sum(float(v.grad.float().square().sum()) for v in net.path_encoder.parameters() if v.grad is not None)) if step<=2 or step%100==0 else None
  opt.step()
  if step<=2 or step%100==0:emit({"event":"train","step":step,"sampled_rmse":float(torch.sqrt(loss*400)),"grad_norm":float(grad),"path_grad_norm":branch_norm,"lr":lr,"peak_GB":torch.cuda.max_memory_allocated()/1e9})
  if step%p["eval_every"]==0 or step==steps:evaluate(step)
 fit=None
 if not args.smoke:fit=full_dev(net,p["fit_panel"],run/"fit_complete",scope="C27 complete valid pixels fixed128FIT; representation diagnostic, not officialVAL")
 atomic_json(run/"complete.json",{"status":"smoke_complete" if args.smoke else "representation_check_complete","steps":steps,"full_FIT":fit,"query_order_sha256":qh.hexdigest(),"map_order_sha256":hashlib.sha256(order.tobytes()).hexdigest(),"gpu":gpu,"seconds":time.time()-t0})
 check_source_unchanged()
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--variant",choices=["none","coarse","fine"],required=True);ap.add_argument("--smoke",action="store_true");main(ap.parse_args())
