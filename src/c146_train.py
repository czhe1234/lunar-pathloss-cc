"""C146 paired solver-feature ablation. No C82 reads; no old weights loaded."""
import argparse, copy, gc, json, math, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard,check_source_unchanged
from src.c146_data import AUDIT,RUN,ARMS,Reader,Panel,create,model_inputs,digest,atomic,sha
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

def save(path,obj):
    temp=path.with_suffix(".partial.pt");torch.save(obj,temp);temp.replace(path)

@torch.inference_mode()
def evaluate(net,panel,arm,batch,label,weights=None):
    net.eval();ss=torch.zeros(2,dtype=torch.float64,device="cuda");cc=torch.zeros_like(ss)
    ws=torch.zeros_like(ss);wc=torch.zeros_like(ss)
    for start in tqdm(range(0,len(panel.ids),batch),desc=label,mininterval=10):
        sl=slice(start,start+batch);basic,phys,fs,y,w=panel.batch(sl)
        with torch.autocast("cuda",dtype=torch.bfloat16):p=fs+net(model_inputs(basic,phys,arm)).float()
        assert torch.isfinite(p).all()
        se=torch.where(w,(p.double()-y.double()).square(),0)
        ss+=se.sum((0,2,3));cc+=w.sum((0,2,3))
        if weights is not None:
            ww=weights[sl,None,None,None].double();ws+=(se*ww).sum((0,2,3));wc+=(w*ww).sum((0,2,3))
    result={"rmse":torch.sqrt(ss/cc).cpu().tolist(),"sse":ss.cpu().tolist(),"count":cc.cpu().tolist()}
    if weights is not None:result["weighted_rmse"]=torch.sqrt(ws/wc).cpu().tolist()
    net.train();return result

def smoke(plan,gpu):
    path=AUDIT/"smoke.json"
    if path.exists():
        rec=json.loads(path.read_text())
        if rec.get("protocol_sha256")==sha(AUDIT/"protocol.json") and rec.get("status")=="passed":return
    reader=Reader();panel=Panel(reader,plan["panels"]["fit"][:plan["batch"]])
    basic,phys,fs,y,w=panel.batch(slice(None));nets={a:create(plan["initial_seed"]).cuda().to(memory_format=torch.channels_last) for a in ARMS}
    assert all(digest(n)==plan["initial_state_sha256"] for n in nets.values())
    opts={a:torch.optim.AdamW(n.parameters(),lr=plan["learning_rate"],weight_decay=plan["weight_decay"],fused=True) for a,n in nets.items()}
    result={a:[] for a in ARMS};t=time.monotonic()
    for step in range(3):
        for arm in ARMS:
            n=nets[arm];opts[arm].zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):p=fs+n(model_inputs(basic,phys,arm)).float()
            loss=(torch.where(w,p-y,0).square().sum((0,2,3))/w.sum((0,2,3))).mean()/25
            loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),2)
            if arm=="without_solver":
                assert torch.count_nonzero(n.local[0].weight.grad[:,11:])==0
                assert torch.count_nonzero(n.film.weight.grad[:,11:])==0
            assert torch.isfinite(loss)
            opts[arm].step();result[arm].append(float(loss.detach()))
    assert digest(nets[ARMS[0]])!=digest(nets[ARMS[1]])
    assert nets["with_solver"].local[0].weight.grad[:,11:56].abs().sum()>0
    atomic(path,{"status":"passed","protocol_sha256":sha(AUDIT/"protocol.json"),"GPU":gpu,"batch":plan["batch"],
                 "three_update_loss":result,"without_solver_grad_paths_zero":True,"with_solver_local_path_active":True,
                 "peak_GiB":torch.cuda.max_memory_allocated()/2**30,"seconds":time.monotonic()-t,
                 "smoke_updates_discarded":True})
    del nets,opts,panel,reader,basic,phys,fs,y,w,n,p,loss
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    print("SMOKE_PASSED",flush=True)

def train(plan,gpu):
    resume=RUN/"resume.pt"
    configpath=RUN/"config.json"
    identity=sha(AUDIT/"protocol.json")
    if configpath.exists():assert json.loads(configpath.read_text())["protocol_sha256"]==identity
    else:atomic(configpath,{"protocol_sha256":identity,"protocol":plan,"GPU":gpu})
    data=Reader();panels={k:Panel(data,plan["panels"][k]) for k in ("fit","monitor")}
    fit=panels["fit"];bs=plan["batch"];steps=math.ceil(len(fit.ids)/bs)
    full=Counter(data.rows[int(i)]["terrain_id"] for i in np.load(ROOT/"features/c38_dense/fit_map_indices.npy"))
    sel=Counter(fit.terrain)
    weights=torch.tensor([full[t]/sel[t] for t in fit.terrain],device="cuda",dtype=torch.float32);weights/=weights.mean()
    den=torch.zeros(2,device="cuda",dtype=torch.float64)
    for start in range(0,len(fit.ids),32):
        den+=(fit.w[start:start+32]*weights[start:start+32,None,None,None]).sum((0,2,3),dtype=torch.float64)
    den=(den/steps).float()
    nets={a:create(plan["initial_seed"]).cuda().to(memory_format=torch.channels_last) for a in ARMS}
    opts={a:torch.optim.AdamW(n.parameters(),lr=plan["learning_rate"],weight_decay=plan["weight_decay"],fused=True) for a,n in nets.items()}
    gen=torch.Generator().manual_seed(plan["data_seed"])
    history=[];best={a:{"epoch":0,"mean_rmse":1e20} for a in ARMS};last_epoch=0
    if resume.exists():
        ck=torch.load(resume,map_location="cpu",weights_only=False);assert ck["protocol_sha256"]==identity
        for a in ARMS:nets[a].load_state_dict(ck["nets"][a]);opts[a].load_state_dict(ck["optimizers"][a])
        gen.set_state(ck["generator"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
        history=ck["history"];best=ck["best"];last_epoch=ck["epoch"];del ck
        for arm in ARMS:
            selected=RUN/f"{arm}_selected.pt"
            selected_ck=torch.load(selected,map_location="cpu",weights_only=False) if selected.exists() else None
            selected_epoch=selected_ck["epoch"] if selected_ck else None
            del selected_ck
            if selected_epoch!=best[arm]["epoch"]:
                if best[arm]["epoch"]!=last_epoch:raise RuntimeError("Selected checkpoint missing before last commit")
                save(selected,{"state":nets[arm].state_dict(),"epoch":last_epoch,"arm":arm,"protocol_sha256":identity})
        atomic(RUN/"history.json",history)
    else:
        assert all(digest(n)==plan["initial_state_sha256"] for n in nets.values())
        # Initial head is zero. Both complete models equal the common analytic FSPL.
        initial=evaluate(nets[ARMS[0]],panels["monitor"],ARMS[0],bs,"C146 initial")
        for a in ARMS:
            best[a]={"epoch":0,"mean_rmse":float(np.mean(initial["rmse"]))}
            save(RUN/f"{a}_selected.pt",{"state":nets[a].state_dict(),"epoch":0,"arm":a,"protocol_sha256":identity})
        atomic(RUN/"initial.json",{"both_arms":initial,"state_sha256":plan["initial_state_sha256"]})
    t=time.monotonic()
    for epoch in range(last_epoch+1,plan["epochs"]+1):
        order=torch.randperm(len(fit.ids),generator=gen).cuda()
        totals={a:torch.zeros(4,device="cuda",dtype=torch.float64) for a in ARMS}
        it=tqdm(list(order.split(bs)),desc=f"C146 paired {epoch}/{plan['epochs']}",mininterval=10)
        for batch_index,ix in enumerate(it,1):
            step=(epoch-1)*steps+batch_index
            warm=min(1.,step/plan["warmup_updates"])
            cosine=.5*(1+math.cos(math.pi*step/plan["updates_per_arm"]))
            lr=warm*(plan["minimum_learning_rate"]+(plan["learning_rate"]-plan["minimum_learning_rate"])*cosine)
            basic,phys,fs,y,w=fit.batch(ix);ww=weights[ix,None,None,None]
            for a in (ARMS if step%2 else ARMS[::-1]):
                n=nets[a];opt=opts[a]
                for pg in opt.param_groups:pg["lr"]=lr
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):p=fs+n(model_inputs(basic,phys,a)).float()
                err=torch.where(w,p-y,0.)
                se=(err.square()*ww).sum((0,2,3))
                loss=(se/den).mean()/25
                if not torch.isfinite(loss):raise RuntimeError(f"Nonfinite loss {a}, {step}")
                loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),plan["clip_grad_norm"])
                opt.step();totals[a]+=torch.cat((se.detach().double(),(w*ww).sum((0,2,3),dtype=torch.float64)))
            if batch_index%20==0 or batch_index==steps:
                atomic(RUN/"progress.json",{"stage":"training","epoch":epoch,"epochs":plan["epochs"],
                    "batch":batch_index,"batches":steps,"updates_per_arm":step,"lr":lr,
                    "elapsed_seconds":time.monotonic()-t,"GPU":gpu})
        record={"epoch":epoch,"updates_per_arm":epoch*steps,"lr":lr,
                "running_fit_rmse":{a:(v[:2]/v[2:]).sqrt().cpu().tolist() for a,v in totals.items()}}
        improved=[]
        if epoch%plan["evaluate_every_epochs"]==0:
            record["monitor"]={}
            for a in ARMS:
                v=evaluate(nets[a],panels["monitor"],a,bs,f"C146 {a} monitor {epoch}")
                record["monitor"][a]=v
                if float(np.mean(v["rmse"]))<best[a]["mean_rmse"]:
                    best[a]={"epoch":epoch,"mean_rmse":float(np.mean(v["rmse"]))};improved.append(a)
        if epoch%plan["fit_evaluate_every_epochs"]==0:
            record["fit"]={a:evaluate(nets[a],fit,a,bs,f"C146 {a} FIT {epoch}",weights) for a in ARMS}
        record["best"]=copy.deepcopy(best);record["seconds"]=time.monotonic()-t
        history.append(record)
        # Both arms commit together. Resume never advances one arm without the other.
        save(resume,{"epoch":epoch,"protocol_sha256":identity,"nets":{a:n.state_dict() for a,n in nets.items()},
              "optimizers":{a:o.state_dict() for a,o in opts.items()},"generator":gen.get_state(),
              "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
              "history":history,"best":best})
        for a in improved:
            save(RUN/f"{a}_selected.pt",{"state":nets[a].state_dict(),"epoch":epoch,"arm":a,"protocol_sha256":identity})
        atomic(RUN/"history.json",history);print("EPOCH",json.dumps(record),flush=True)
    for a in ARMS:
        save(RUN/f"{a}_final.pt",{"state":nets[a].state_dict(),"epoch":plan["epochs"],"arm":a,"protocol_sha256":identity})
    atomic(RUN/"training_complete.json",{"status":"training_complete_pending_export","epochs":plan["epochs"],
        "updates_per_arm":plan["updates_per_arm"],"best":best,"protocol_sha256":identity})
    del nets,opts,panels,fit,data,weights
    gc.collect();torch.cuda.empty_cache()

@torch.inference_mode()
def export(plan):
    # This Reader does not open labels or masks.
    reader=Reader(supervision=False);bs=plan["batch"]
    frozen={}
    for arm in ARMS:
        for tag in ("selected","final"):
            path=RUN/f"{arm}_{tag}.pt"
            ck=torch.load(path,map_location="cpu",weights_only=False)
            frozen[f"{arm}_{tag}"]={"sha256":sha(path),"epoch":ck["epoch"]}
            del ck
    atomic(RUN/"frozen_checkpoints.json",frozen)
    for arm in ARMS:
        for tag in ("selected","final"):
            key=f"{arm}_{tag}";ck=torch.load(RUN/f"{key}.pt",map_location="cpu",weights_only=False)
            net=create(plan["initial_seed"]);net.load_state_dict(ck["state"]);del ck
            net=net.cuda().to(memory_format=torch.channels_last).eval()
            for name,ids in plan["panels"].items():
                path=RUN/f"{key}_{name}.npy";meta=path.with_suffix(".json")
                identity={"checkpoint_sha256":frozen[key]["sha256"],"ids_sha256":hashlib_ids(ids),"shape":[len(ids),2,256,256]}
                if path.exists() and meta.exists():
                    old=json.loads(meta.read_text())
                    if all(old.get(k)==v for k,v in identity.items()) and old.get("prediction_sha256")==sha(path):continue
                tmp=path.with_suffix(".partial.npy")
                out=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.float32,shape=tuple(identity["shape"]))
                for start in tqdm(range(0,len(ids),bs),desc=f"export {key} {name}",mininterval=10):
                    basic,phys,fs=reader.inputs(ids[start:start+bs],"cuda")
                    with torch.autocast("cuda",dtype=torch.bfloat16):p=fs+net(model_inputs(basic,phys,arm)).float()
                    if not torch.isfinite(p).all():raise RuntimeError("Nonfinite export")
                    out[start:start+len(p)]=p.cpu().numpy()
                out.flush();del out;tmp.replace(path)
                atomic(meta,{**identity,"prediction_sha256":sha(path),"labels_opened":False,"C82_opened":False})
            del net;gc.collect();torch.cuda.empty_cache()
    atomic(RUN/"export_complete.json",{"status":"export_complete_pending_CPU_verification","frozen":frozen})

def hashlib_ids(ids):
    import hashlib
    return hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest()

def main():
    plan=json.loads((AUDIT/"protocol.json").read_text())
    for name,expected in plan["source_sha256"].items():assert sha(ROOT/name)==expected,name
    gpu=gpu_guard();gpu["sharing_authorized"]=True
    print("GPU_ready",json.dumps(gpu),flush=True)
    check_source_unchanged();torch.set_num_threads(6);torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    smoke(plan,gpu)
    if not (RUN/"training_complete.json").exists():train(plan,gpu)
    export(plan)
if __name__=="__main__":main()
