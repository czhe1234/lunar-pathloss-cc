"""C25: reduce variation across terrain batches by averaging gradients at fixed data exposure."""
from cc_core import ROOT, atomic_json, check_source_unchanged
import argparse, hashlib, json, math, shutil, time
from pathlib import Path
import numpy as np
import torch
from c21_data import Data
from c21_train_r1 import setup, model, gather, digest_state
from c21_pair_r1 import loader, gpu_batch, combine, full_dev

AUDIT=ROOT/"audit/c25_map_batch_20261101"
PARENT=ROOT/"runs/c21r1_pair_s20261123"
@torch.inference_mode()
def monitor(net,ids,pixels):
    net.eval();rows=[]
    for batch in loader(ids,4,4):
        idx=batch[0].tolist();x,tx,fs,y,m=gpu_batch(batch)
        pix=torch.as_tensor(pixels,device="cuda")[None].expand(len(x),-1)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=net(x,tx,pix,512)*20+gather(fs,pix)
        mm=gather(m,pix)
        ss=((p.double()-gather(y,pix).double()).square()*mm).sum(1).cpu().numpy()
        cc=mm.sum(1,dtype=torch.float64).cpu().numpy()
        for i,s,c in zip(idx,ss,cc):rows.append({"id":i,"sse":s.tolist(),"count":c.tolist()})
    net.train()
    return combine(rows)

def grad_groups(net):
    groups={}
    for name,p in net.named_parameters():
        if p.grad is None:continue
        key=name.split(".")[0]
        groups[key]=groups.get(key,0.)+float(p.grad.float().square().sum())
    return {k:math.sqrt(v) for k,v in groups.items()}

def main(args):
    protocol=json.loads((AUDIT/"protocol.json").read_text())
    variant=next(v for v in protocol["slices"] if v["name"]==args.variant)
    accum=variant["accumulation"];assert args.steps%accum==0 and args.eval_every%accum==0
    gpu=setup(protocol["seed"]);check_source_unchanged()
    run=ROOT/"runs"/(args.variant+args.suffix);run.mkdir(exist_ok=False)
    cfg={"variant":variant,"seed":protocol["seed"],"steps":args.steps,
         "eval_every":args.eval_every,"batch":8,"queries":2048,"gpu":gpu,"effective_batch":8*accum,"accumulation":accum,
         "parent_sha256":protocol["parent_sha256"],"parent_step":18000,
         "officialVAL_used":False,"memory_fraction":.15,"sources":{}}
    (run/"source").mkdir()
    for name in ["c25_train.py","cc_core.py","c21_model.py","c21_model_r1.py","c21_train_r1.py","c21_data.py","c21_pair_r1.py"]:
        cfg["sources"][name]=hashlib.sha256((ROOT/"src"/name).read_bytes()).hexdigest()
        shutil.copy2(ROOT/"src"/name,run/"source"/name)
    data=Data();ids=data.ids["fit"] if variant["fit"]=="all" else np.array(protocol["fit_panel"])
    rng=np.random.default_rng(protocol["seed"])
    order=np.concatenate([rng.permutation(ids) for _ in range(math.ceil(args.steps*8/len(ids)))])[:args.steps*8]
    np.save(run/"map_order.npy",order)
    cfg["map_order_sha256"]=hashlib.sha256(order.tobytes()).hexdigest()
    cfg["training_maps"]=len(ids);cfg["mean_map_visits"]=len(order)/len(ids)
    atomic_json(run/"config.json",cfg)
    net=model(False)
    ck=torch.load(PARENT/"last_A.pt",map_location="cpu",weights_only=False)
    net.load_state_dict(ck["model"])
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=variant["lr"],weight_decay=1e-5)
    opt.load_state_dict(ck["optimizer"])
    initial=digest_state(net);del ck
    gen=torch.Generator(device="cpu").manual_seed(protocol["seed"]+1)
    pixels=np.load(PARENT/"monitor_pixels.npy")
    t0=time.time();log=(run/"history.jsonl").open("a",buffering=1)
    best=1e9;best_step=0;query_hash=hashlib.sha256()
    def emit(row):
        row["seconds"]=time.time()-t0
        print(json.dumps(row),flush=True);log.write(json.dumps(row)+"\n")
        atomic_json(run/"progress.json",row)
    def evaluate(step):
        nonlocal best,best_step
        fit=monitor(net,protocol["fit_panel"],pixels)
        dev=monitor(net,protocol["dev_panel"],pixels)
        row={"event":"monitor","step":step,"FIT":fit,"DEV":dev}
        emit(row);atomic_json(run/"latest_eval.json",row)
        checkpoint={"model":net.state_dict(),"optimizer":opt.state_dict(),"step":step,
                    "parent_step":18000,"metrics":row,"config":cfg}
        torch.save(checkpoint,run/"last.pt")
        if dev["score"]<best:
            best=dev["score"];best_step=step
            shutil.copy2(run/"last.pt",run/"best.pt");atomic_json(run/"best.json",row)
        if step==args.steps:
            atomic_json(run/"final_metrics.json",row)
    atomic_json(run/"initialization.json",{"model_sha256":initial,"parent_optimizer_restored":True})
    evaluate(0)
    clipped=0;count=0;grad=0.;groups=None
    opt.zero_grad(set_to_none=True)
    for step,batch in enumerate(loader(order,8,args.workers),1):
        x,tx,fs,y,m=gpu_batch(batch)
        pix_cpu=torch.randint(65536,(len(x),2048),generator=gen)
        query_hash.update(pix_cpu.numpy().tobytes());pix=pix_cpu.cuda(non_blocking=True)
        for pg in opt.param_groups:pg["lr"]=variant["lr"]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred=net(x,tx,pix,512)*20+gather(fs,pix)
        mm=gather(m,pix);ss=((pred-gather(y,pix)).square()*mm).sum((0,1));cc=mm.sum((0,1))
        loss=(ss/cc.clamp_min(1)).mean()/400
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
        (loss/accum).backward()
        groups=None
        if step%accum==0:
            groups=grad_groups(net) if step%1000==0 else None
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
            if not torch.isfinite(grad):raise RuntimeError("Nonfinite gradient")
            clipped+=int(grad>5);count+=1;opt.step();opt.zero_grad(set_to_none=True)
        if step<=2 or step%100==0:
            emit({"event":"train","step":step,"sampled_rmse":torch.sqrt(ss/cc).tolist(),
                  "grad_norm":float(grad),"clipped_steps":clipped,"optimizer_updates":count,
                  "grad_groups":groups,"lr":variant["lr"],"peak_GB":torch.cuda.max_memory_allocated()/1e9})
        if step%args.eval_every==0 or step==args.steps:evaluate(step)
    atomic_json(run/"complete.json",{"status":"training_complete","steps":args.steps,"best_step":best_step,
        "seconds":time.time()-t0,"query_order_sha256":query_hash.hexdigest(),"clipped_steps":clipped,
        "optimizer_updates":count,"accumulation":accum,"final_model_sha256":digest_state(net),"gpu":gpu})
    log.close()

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--variant",required=True)
    ap.add_argument("--steps",type=int,default=6000);ap.add_argument("--eval-every",type=int,default=1000)
    ap.add_argument("--suffix",default="");ap.add_argument("--workers",type=int,default=6)
    main(ap.parse_args())
