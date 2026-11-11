"""Matched128-FIT capacity intervention; independent models on approved H100s."""
from cc_core import ROOT,atomic_json,check_source_unchanged
import argparse,json,hashlib,time,math,shutil
import numpy as np
import torch
from c21_train_r1 import setup,model,gather,digest_state
from c21_pair_r1 import loader,gpu_batch,full_dev
from c22_train import monitor
from c24_model import expand_readout
A=ROOT/"audit/c29_fit_regime_20261101"
def main(args):
    p=json.loads((A/"protocol.json").read_text())["fit128_base_protocol"];p["lr"]=args.lr;gpu=setup(p["seed"]);check_source_unchanged()
    torch.backends.cuda.matmul.allow_tf32=not args.fp32;torch.backends.cudnn.allow_tf32=not args.fp32
    run=ROOT/"runs"/("c29_fit128_"+("mid" if args.lr==.0005 else "high")+"_"+("fp32" if args.fp32 else "bf16")+("_smoke" if args.smoke else ""));run.mkdir(exist_ok=False);(run/"source").mkdir()
    net=model(False);parent=ROOT/p["parent"];ck=torch.load(parent,map_location="cpu",weights_only=False)
    net.load_state_dict(ck["model"]);del ck
    if args.variant=="wide":net=expand_readout(net)
    cfg={"protocol":p,"variant":args.variant,"gpu":gpu,"parent_sha256":hashlib.sha256(parent.read_bytes()).hexdigest(),"parameters":sum(v.numel() for v in net.parameters()),"initial_state_sha256":digest_state(net),"optimizer":"fresh matched AdamW","training_fp32":args.fp32,"smoke":args.smoke}
    for name in ["c29_fit128.py","c24_train.py","c24_model.py","c22_train.py","cc_core.py","c21_model.py","c21_model_r1.py","c21_data.py","c21_train_r1.py","c21_pair_r1.py"]:shutil.copy2(ROOT/"src"/name,run/"source"/name)
    atomic_json(run/"config.json",cfg)
    normal=[];amplified=[]
    for name,value in net.named_parameters():
        if not value.requires_grad:continue
        amplify=(args.variant=="standard_headlr" and name.startswith("readout.") and any(name.endswith(t) for t in ["2.a.weight","2.b.weight","3.a.weight","3.b.weight","4.weight"]))
        (amplified if amplify else normal).append(value)
    groups=[{"params":normal,"lr_scale":1.}]
    if amplified:groups.append({"params":amplified,"lr_scale":2.})
    opt=torch.optim.AdamW(groups,lr=p["lr"],weight_decay=1e-5)
    steps=2 if args.smoke else p["steps"];ids=np.array(p["fit_panel"]);rng=np.random.default_rng(p["seed"])
    order=np.concatenate([rng.permutation(ids) for _ in range(math.ceil(steps*8/len(ids)))])[:steps*8]
    np.save(run/"map_order.npy",order)
    pixels=np.load(ROOT/"runs/c21r1_pair_s20261123/monitor_pixels.npy")
    gen=torch.Generator().manual_seed(p["seed"]+1);t0=time.time()
    log=(run/"history.jsonl").open("a",buffering=1);qh=hashlib.sha256()
    def emit(r):
        r["seconds"]=time.time()-t0;print(json.dumps(r),flush=True)
        log.write(json.dumps(r)+"\n");atomic_json(run/"progress.json",r)
    def evaluate(step):
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        fit=monitor(net,p["fit_panel"][:4] if args.smoke else p["fit_panel"],pixels);dev=monitor(net,p["dev_panel"][:4] if args.smoke else p["dev_panel"],pixels)
        torch.backends.cuda.matmul.allow_tf32=not args.fp32;torch.backends.cudnn.allow_tf32=not args.fp32
        row={"event":"monitor","step":step,"FIT":fit,"DEV":dev}
        emit(row);atomic_json(run/"latest_eval.json",row)
        torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),"step":step,"metrics":row,"config":cfg},run/"last.pt")
    evaluate(0)
    for step,batch in enumerate(loader(order,8,6),1):
        x,tx,fs,y,m=gpu_batch(batch);pix_cpu=torch.randint(65536,(len(x),2048),generator=gen);qh.update(pix_cpu.numpy().tobytes());pix=pix_cpu.cuda()
        lr=p["lr"]*min(step/100.,1)*(.1+.9*.5*(1+math.cos(math.pi*step/p["steps"])))
        for group in opt.param_groups:group["lr"]=lr*group["lr_scale"]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not args.fp32):pred=net(x,tx,pix,512)*20+gather(fs,pix)
        mm=gather(m,pix);loss=(((pred-gather(y,pix)).square()*mm).sum((0,1))/mm.sum((0,1)).clamp_min(1)).mean()/400
        if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError("nonfinite gradient")
        opt.step()
        if step<=2 or step%100==0:emit({"event":"train","step":step,"sampled_rmse":float(torch.sqrt(loss*400)),"grad_norm":float(grad),"lr":lr,"peak_GB":torch.cuda.max_memory_allocated()/1e9})
        if step%1000==0 or step==steps:evaluate(step)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    fit=None if args.smoke else full_dev(net,p["fit_panel"],run/"fit_complete",scope="C29 complete valid128FIT; learning-rate/precision diagnostic, not officialVAL")
    atomic_json(run/"complete.json",{"status":"fit_regime_complete","steps":steps,"full_FIT":fit,"query_order_sha256":qh.hexdigest(),"map_order_sha256":hashlib.sha256(order.tobytes()).hexdigest(),"gpu":gpu,"seconds":time.time()-t0})
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--fp32",action="store_true");ap.add_argument("--smoke",action="store_true");ap.add_argument("--lr",type=float,default=.001)
    args=ap.parse_args();args.variant="standard";main(args)
