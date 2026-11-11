"""C32: learn C10 residuals with the full terrain-query network."""
from cc_core import ROOT,atomic_json,check_source_unchanged
import argparse,json,hashlib,time,math,shutil
import numpy as np
import torch
from c21_train_r1 import setup,gather,digest_state
from c21_pair_r1 import gpu_batch,combine
from c32_model import Model
from c32_data import loader
A=ROOT/"audit/c32_neural_residual_20261101"
def build(variant,protocol):
 n=np.load(ROOT/"audit/c21_execution_20261101/normalization.npz")
 mean=torch.from_numpy(np.r_[n["mean"],np.zeros(2,np.float32)]);std=torch.from_numpy(np.r_[n["std"],np.ones(2,np.float32)])
 net=Model(mean,std,geometry=False)
 if variant!="cold":
  path=ROOT/protocol["parent"];assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol["parent_sha256"]
  ck=torch.load(path,map_location="cpu",weights_only=False)["model"];state=net.state_dict()
  for k,v in state.items():
   if k in ["mean","std"]:continue
   if v.shape==ck[k].shape:v.copy_(ck[k])
   elif k in ["encoder.0.net.0.weight","encoder.0.skip.weight"]:
    v.zero_();v[:,:45].copy_(ck[k])
   elif k=="query.weight" or (k.startswith("readout.") and k.endswith(".0.weight")):
    v.zero_();v[:,:141].copy_(ck[k][:,:141]);v[:,143:].copy_(ck[k][:,141:])
   else:raise RuntimeError("Unexpected shape "+k)
  net.load_state_dict(state)
 for head in net.readout:torch.nn.init.zeros_(head[-1].weight);torch.nn.init.zeros_(head[-1].bias)
 if variant=="warm_frozen":
  for name,v in net.named_parameters():
   if name.startswith(("encoder.","localproj.","local.","scene.","position.","scale","knorm.","kv.")):v.requires_grad_(False)
 return net.cuda()
@torch.inference_mode()
def evaluate(net,ids,part,pixels=None,dest=None):
 net.eval();rows=[];at=0;t0=time.time()
 if dest:
  dest.mkdir(exist_ok=False);out=np.lib.format.open_memmap(dest/"pred.npy",mode="w+",dtype=np.float32,shape=(len(ids),2,256,256))
 for batch in loader(ids,part,4,4):
  idx=batch[0].tolist();x,tx,base,y,m=gpu_batch(batch);B=len(x)
  with torch.autocast("cuda",dtype=torch.bfloat16):state=net.encode(x,tx)
  if pixels is None:
   pred=torch.empty((B,65536,2),device="cuda",dtype=torch.float32)
   for lo in range(0,65536,1024):
    pix=torch.arange(lo,min(lo+1024,65536),device="cuda")[None].expand(B,-1)
    with torch.autocast("cuda",dtype=torch.bfloat16):pred[:,lo:lo+pix.shape[1]]=net.decode(state,pix,1024)*5+gather(base,pix)
   yy=y.flatten(2).transpose(1,2);mm=m.flatten(2).transpose(1,2)
   if dest:out[at:at+B]=pred.transpose(1,2).reshape(B,2,256,256).cpu().numpy()
  else:
   pix=torch.as_tensor(pixels,device="cuda")[None].expand(B,-1)
   with torch.autocast("cuda",dtype=torch.bfloat16):pred=net.decode(state,pix,512)*5+gather(base,pix)
   yy=gather(y,pix);mm=gather(m,pix)
  ss=((pred.double()-yy.double()).square()*mm).sum(1).cpu().numpy();cc=mm.sum(1,dtype=torch.float64).cpu().numpy()
  rows.extend({"id":i,"sse":s.tolist(),"count":c.tolist()} for i,s,c in zip(idx,ss,cc));at+=B
  if dest and at%128==0:print(json.dumps({"event":"complete_pixel_eval","maps":at,"seconds":time.time()-t0}),flush=True)
 result=combine(rows)
 if dest:
  out.flush();atomic_json(dest/"ids.json",list(map(int,ids)));atomic_json(dest/"metrics.json",result);atomic_json(dest/"per_map.json",rows)
 net.train();return result
def main(args):
 p=json.loads((A/"protocol.json").read_text());gpu=setup(p["seed"]);check_source_unchanged()
 run=ROOT/"runs"/("c32_"+args.variant+("_smoke" if args.smoke else ""));run.mkdir(exist_ok=False);(run/"source").mkdir()
 net=build(args.variant,p);cfg={"protocol":p,"variant":args.variant,"gpu":gpu,"parameters":sum(v.numel() for v in net.parameters()),"trainable":sum(v.numel() for v in net.parameters() if v.requires_grad),"initial_state_sha256":digest_state(net),"smoke":args.smoke}
 for name in ["c32_train.py","c32_model.py","c32_data.py","c21_data.py","c21_model_r1.py","c21_model.py","c21_train_r1.py","cc_core.py"]:shutil.copy2(ROOT/"src"/name,run/"source"/name)
 atomic_json(run/"config.json",cfg)
 steps=2 if args.smoke else p["steps"];rng=np.random.default_rng(p["seed"]);ids=np.array(p["fit_ids"])
 order=np.concatenate([rng.permutation(ids) for _ in range(math.ceil(steps*8/len(ids)))])[:steps*8];np.save(run/"map_order.npy",order)
 gen=torch.Generator().manual_seed(p["seed"]+1);qh=hashlib.sha256();pixels=np.load(ROOT/"runs/c21r1_pair_s20261123/monitor_pixels.npy")
 opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=p["lr"],weight_decay=1e-5)
 t0=time.time();best=1e9;baseline=None;bestrow=None;log=(run/"history.jsonl").open("a",buffering=1)
 def emit(row):
  row["seconds"]=time.time()-t0;print(json.dumps(row),flush=True);log.write(json.dumps(row)+"\n");atomic_json(run/"progress.json",row)
 def monitor(step):
  nonlocal best,bestrow,baseline
  fit=evaluate(net,p["fit_monitor"][:4] if args.smoke else p["fit_monitor"],"fit",pixels)
  dev=evaluate(net,p["dev_ids"][:4] if args.smoke else p["dev_ids"],"dev",pixels)
  row={"event":"monitor","step":step,"FIT":fit,"DEV":dev};emit(row);atomic_json(run/"latest_eval.json",row)
  if step==0:baseline=dev;atomic_json(run/"baseline.json",row)
  ck={"model":net.state_dict(),"optimizer":opt.state_dict(),"step":step,"metrics":row,"config":cfg}
  torch.save(ck,run/"last.pt")
  if dev["score"]<best:
   best=dev["score"];bestrow=row;shutil.copy2(run/"last.pt",run/"best.pt");atomic_json(run/"best.json",row)
 monitor(0);stop_reason="fixed_budget"
 for step,batch in enumerate(loader(order,"fit",8,4),1):
  x,tx,base,y,m=gpu_batch(batch);pix_cpu=torch.randint(65536,(len(x),2048),generator=gen);qh.update(pix_cpu.numpy().tobytes());pix=pix_cpu.cuda()
  lr=p["lr"]*min(step/p["warmup"],1)*(.1+.9*.5*(1+math.cos(math.pi*step/p["steps"])))
  for group in opt.param_groups:group["lr"]=lr
  opt.zero_grad(set_to_none=True)
  with torch.autocast("cuda",dtype=torch.bfloat16):pred=net(x,tx,pix,512)*5+gather(base,pix)
  mm=gather(m,pix);loss=(((pred-gather(y,pix)).square()*mm).sum((0,1))/mm.sum((0,1)).clamp_min(1)).mean()/25
  if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
  if not torch.isfinite(grad):raise RuntimeError("nonfinite gradient")
  opt.step()
  if step<=2 or step%100==0:emit({"event":"train","step":step,"sampled_rmse":float(torch.sqrt(loss*25)),"grad_norm":float(grad),"lr":lr,"peak_GB":torch.cuda.max_memory_allocated()/1e9})
  if step%p["eval_every"]==0 or step==steps:monitor(step)
  if step==3000 and baseline["score"]-best<.02:stop_reason="insufficient_DEV_gain_at3000";break
 gain=np.array(baseline["rmse"])-np.array(bestrow["DEV"]["rmse"])
 complete={"status":"smoke_complete" if args.smoke else "training_complete","steps":step,"stop_reason":stop_reason,"best_step":bestrow["step"],"baseline_DEV":baseline,"best_DEV":bestrow["DEV"],"gain_db":gain.tolist(),"full_DEV_gate":bool(np.all(gain>=.15)),"query_order_sha256":qh.hexdigest(),"map_order_sha256":hashlib.sha256(order.tobytes()).hexdigest(),"gpu":gpu,"seconds":time.time()-t0}
 atomic_json(run/"complete.json",complete);check_source_unchanged()
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--variant",choices=["cold","warm","warm_frozen"],required=True);ap.add_argument("--smoke",action="store_true");main(ap.parse_args())
