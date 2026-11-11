"""C28: fixed focal-map exposure with other-map gradient controls."""
from cc_core import ROOT,atomic_json,check_source_unchanged
import argparse,json,hashlib,time,math,shutil
import numpy as np
import torch
from c21_train_r1 import setup,model,gather,digest_state
from c21_pair_r1 import loader,gpu_batch,full_dev
from c22_train import monitor
A=ROOT/"audit/c28_matched_exposure_20261101"
def main(args):
 p=json.loads((A/"protocol.json").read_text());gpu=setup(p["seed"]);check_source_unchanged()
 run=ROOT/"runs"/("c28_"+args.variant+("_smoke" if args.smoke else ""));run.mkdir(exist_ok=False);(run/"source").mkdir()
 net=model(False);parent=ROOT/p["parent"]
 assert hashlib.sha256(parent.read_bytes()).hexdigest()==p["parent_sha256"]
 ck=torch.load(parent,map_location="cpu",weights_only=False);net.load_state_dict(ck["model"]);del ck
 steps=2 if args.smoke else p["steps"];others=np.array(p["others"]);rng=np.random.default_rng(p["seed"])
 oo=np.concatenate([rng.permutation(others) for _ in range(math.ceil(steps*4/len(others)))])[:steps*4].reshape(steps,4)
 order=np.concatenate((np.tile(np.array(p["anchors"]),(steps,1)),oo),1).reshape(-1)
 np.save(run/"map_order.npy",order)
 cfg={"protocol":p,"variant":args.variant,"gpu":gpu,"parameters":sum(v.numel() for v in net.parameters()),"initial_state_sha256":digest_state(net),"smoke":args.smoke,"map_order_sha256":hashlib.sha256(order.tobytes()).hexdigest()}
 for name in ["c28_train.py","c22_train.py","cc_core.py","c21_model.py","c21_model_r1.py","c21_data.py","c21_train_r1.py","c21_pair_r1.py"]:shutil.copy2(ROOT/"src"/name,run/"source"/name)
 atomic_json(run/"config.json",cfg)
 params=[v for v in net.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=p["lr"],weight_decay=1e-5)
 pixels=np.load(ROOT/"runs/c21r1_pair_s20261123/monitor_pixels.npy")
 gen=torch.Generator().manual_seed(p["seed"]+1);t0=time.time();log=(run/"history.jsonl").open("a",buffering=1);qh=hashlib.sha256()
 neg=0;cos_sum=0.;clipped=0
 def emit(r):
  r["seconds"]=time.time()-t0;print(json.dumps(r),flush=True);log.write(json.dumps(r)+"\n");atomic_json(run/"progress.json",r)
 def evaluate(step):
  anchor=monitor(net,p["anchors"],pixels);other=monitor(net,p["others"][:4] if args.smoke else p["others"],pixels)
  row={"event":"monitor","step":step,"anchor":anchor,"other":other}
  if args.smoke or step%1500==0:row["DEV"]=monitor(net,p["dev_panel"][:4] if args.smoke else p["dev_panel"],pixels)
  emit(row);atomic_json(run/"latest_eval.json",row)
  torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),"step":step,"metrics":row,"config":cfg},run/"last.pt")
 evaluate(0)
 for step,batch in enumerate(loader(order,8,6),1):
  assert batch[0][:4].tolist()==p["anchors"]
  x,tx,fs,y,m=gpu_batch(batch);pix_cpu=torch.randint(65536,(len(x),2048),generator=gen);qh.update(pix_cpu.numpy().tobytes());pix=pix_cpu.cuda()
  lr=p["lr"]*min(step/p["warmup"],1)*(.05+.95*.5*(1+math.cos(math.pi*step/p["steps"])))
  for group in opt.param_groups:group["lr"]=lr
  opt.zero_grad(set_to_none=True)
  with torch.autocast("cuda",dtype=torch.bfloat16):pred=net(x,tx,pix,512)*20+gather(fs,pix)
  mm=gather(m,pix);errors=(pred-gather(y,pix)).square()*mm
  la=(errors[:4].sum((0,1))/mm[:4].sum((0,1)).clamp_min(1)).mean()/400
  lo=(errors[4:].sum((0,1))/mm[4:].sum((0,1)).clamp_min(1)).mean()/400
  if not torch.isfinite(la+lo):raise RuntimeError("nonfinite loss")
  ga=torch.autograd.grad(la,params,retain_graph=True);go=torch.autograd.grad(lo,params)
  dot=torch.stack([(a*b).sum() for a,b in zip(ga,go)]).sum()
  na=torch.stack([a.square().sum() for a in ga]).sum()
  no=torch.stack([b.square().sum() for b in go]).sum()
  cosine=dot/(na*no).clamp_min(1e-30).sqrt();isneg=bool(dot<0);neg+=int(isneg);cos_sum+=float(cosine)
  coef=torch.minimum(dot,torch.zeros_like(dot))/na.clamp_min(1e-30)
  for param,a,b in zip(params,ga,go):
   if args.variant=="anchor_only":param.grad=a*.5
   elif args.variant=="joint":param.grad=(a+b)*.5
   else:param.grad=(a+b-coef*a)*.5
  grad=torch.nn.utils.clip_grad_norm_(params,5)
  if not torch.isfinite(grad):raise RuntimeError("nonfinite gradient")
  clipped+=int(grad>5);opt.step()
  if step<=2 or step%100==0:
   emit({"event":"train","step":step,"anchor_sampled_rmse":float(torch.sqrt(la*400)),"other_sampled_rmse":float(torch.sqrt(lo*400)),
         "gradient_cosine":float(cosine),"mean_cosine":cos_sum/step,"negative_fraction":neg/step,"project_coefficient":float(coef) if args.variant=="project_other" else 0.,
         "grad_norm":float(grad),"clipped_steps":clipped,"lr":lr,"peak_GB":torch.cuda.max_memory_allocated()/1e9})
  del ga,go,pred,errors,la,lo
  if step%p["eval_every"]==0 or step==steps:evaluate(step)
 fit=None
 if not args.smoke:fit=full_dev(net,p["fit_panel"],run/"fit_complete",scope="C28 complete valid pixels fixed128FIT,4 focal maps oversampled; diagnostic, not officialVAL")
 atomic_json(run/"complete.json",{"status":"smoke_complete" if args.smoke else "exposure_control_complete","steps":steps,"full_FIT":fit,"query_order_sha256":qh.hexdigest(),"map_order_sha256":cfg["map_order_sha256"],"initial_state_sha256":cfg["initial_state_sha256"],"negative_gradient_fraction":neg/steps,"mean_gradient_cosine":cos_sum/steps,"clipped_steps":clipped,"gpu":gpu,"seconds":time.time()-t0})
 check_source_unchanged()
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--variant",choices=["anchor_only","joint","project_other"],required=True);ap.add_argument("--smoke",action="store_true");main(ap.parse_args())
