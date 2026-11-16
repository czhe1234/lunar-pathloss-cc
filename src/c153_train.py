"""C153: one fresh standalone C151a-aligned Field, registered FSPL, 60 epochs."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from copy import deepcopy
import argparse, hashlib, json, math, os, subprocess, sys, time
import numpy as np
import torch
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cc_core import GPU_UUID, APPROVED_GPU_UUIDS, gpu_guard, check_source_unchanged
from src.c151_field import create, digest, PARAMETERS, PERMUTATION_SEED
from src.c150_field import inputs, normalize_diffraction
from src.c150_train import Reader, Panel, EvaluationReferences, cache_identity
from src.c151_train import atomic, save, link_atomic, sha, ordered_hash, groups, GROUPS, VERIFIED_GROUPS, BANDS, C82
from src import cc_tracking
AUDIT=ROOT/"audit/c153_standalone_c151a"
RUN=ROOT/"runs/c153_standalone_c151a"
ARM="standalone"
UUID="GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
EPOCHS=60
STEPS=352
UPDATES=EPOCHS*STEPS

def learning_rate(update,plan):
    if not 1<=update<=plan["updates"]:raise ValueError("Update outside frozen budget")
    if update<=plan["warmup_updates"]:
        return plan["learning_rate"]*update/plan["warmup_updates"]
    fraction=(update-plan["warmup_updates"])/(plan["updates"]-plan["warmup_updates"])
    return plan["minimum_lr"]+.5*(plan["learning_rate"]-plan["minimum_lr"])*(1+math.cos(math.pi*fraction))

def predictor_digest(net_hash,plan):
    contract={"net_state_sha256":net_hash,"output":"registered_FSPL + Field(raw68)",
        "global_mode":"aligned_global","C82_input":False,"C82_output_prior":False,
        "feature_cache_sha256":plan["cache"]["features_sha256"]}
    return hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()

def make_protocol():
    if RUN.exists() or (AUDIT/"protocol.json").exists():raise FileExistsError("C153 already prepared; inspect rather than repeat")
    first=ROOT/"audit/c153_grazing_groups/result.json";d=json.loads(first.read_text())
    assert d["status"]=="complete" and not d["training_run"] and not d["network_inference_run"]
    parent=ROOT/"audit/c145_capacity/protocol.json"
    c151=json.loads((ROOT/"audit/c151_global/protocol.json").read_text())
    panels=json.loads(parent.read_text())["panels"]
    model=create("aligned_global",123041,PERMUTATION_SEED)
    assert digest(model)==c151["initial_state_sha256"]
    assert sum(p.numel() for p in model.parameters())==PARAMETERS
    protected=[
        "audit/c101_residual/base0of1/prediction.npy","audit/c88_mesh/base.npy",
        "runs/c145_capacity_native/best_monitor.pt","runs/c145_capacity_native/best_fit.pt","runs/c145_capacity_native/last.pt",
        "runs/c145_capacity_native/selected_export.json","runs/c145_capacity_native/selected_verified.json",
        "runs/c145_capacity_native/selected_terrain_sse.npz",
        "runs/c145_capacity_native/selected_fit_prediction.npy","runs/c145_capacity_native/selected_monitor_prediction.npy"]
    files=["src/c153_train.py","src/c153_report.py","src/c153_diagnostic.py",
        "src/c151_field.py","src/c151_train.py","src/c150_train.py","src/c150_field.py",
        "src/c146_data.py","src/c123_lossless.py","src/c99_axial.py","src/c117_factorized.py",
        "src/c3_geometry.py","src/cc_core.py","src/cc_tracking.py","src/c145_verify.py","src/c150_report.py"]
    plan={"campaign":"C153","name":"C153_standalone_c151a","created_utc":datetime.now(timezone.utc).isoformat(),
        "question":"Does one C82-free aligned C151a Field improve the frozen C82 after60epochs from scratch?",
        "stage1":{"path":str(first.relative_to(ROOT)),"sha256":sha(first),"reported_to_chat_before_training":True},
        "base_protocol":str(parent.relative_to(ROOT)),"base_protocol_sha256":sha(parent),"panels":panels,
        "scope":{"fit_maps":len(panels["fit"]),"fit_terrains":1984,"monitor_maps":len(panels["monitor"]),"monitor_terrains":128},
        "arms":{ARM:"registered_FSPL + Field(raw68)"},"parameters":PARAMETERS,"channels":68,
        "channel_contract":"11 original basic +45 c73 +2 exactly zero C82 slots +10 normalized C150 diffraction",
        "global_input":"Exact existing C151a aligned_global: derived13 basic +55 physics, pixel_unshuffle4,1088to384",
        "global_input_channels":68,"global_mode":"aligned_global","permutation_seed":PERMUTATION_SEED,
        "initial_seed":123041,"data_seed":123073,"initial_state_sha256":digest(model),
        "initialization":"Fresh exact C151a random initial state; zero output head; no trained checkpoint loaded",
        "C82_role":"Scoring reference only; not training input, not output prior, no distillation or anchoring",
        "prior":"Existing registered3D FSPL Float32 cache planes10:12; no extra Ld in output",
        "cache":cache_identity(panels,require=True),
        "epochs":EPOCHS,"batch":8,"steps_per_epoch":STEPS,"updates":UPDATES,"updates_per_arm":UPDATES,
        "learning_rate":3e-4,"minimum_lr":1e-5,"warmup_updates":200,
        "schedule":"Single stage: linear0to3e-4 over updates1..200, then cosine ending exactly1e-5 at21120",
        "optimizer":"AdamW fused, single optimizer, no reset","betas":[.9,.999],"eps":1e-8,
        "weight_decay":.01,"clip_grad_norm":2.0,
        "loss":"C145 fullFIT-normalized map-weighted masked per-band SSE: mean_b(batch_weighted_SSE_b/(FIT_weighted_valid_count_b/352))/25",
        "image_weights":"full_C38_FIT_mapcount(terrain)/selected_FIT_mapcount(terrain), normalized mean1; same C145",
        "validity":"Original frequency-specific bitpacked masks for loss/scoring only, never network inputs",
        "evaluate_epochs":[0,*range(4,EPOCHS+1,4)],
        "selection":"One same checkpoint at minimum joint mean monitor RMSE among fixed evaluations; first minimum wins",
        "paired_CI":{"unit":"terrain","terrains":128,"replicates":2000,"seed":145073,
            "difference":"C153_RMSE-C82_RMSE; negative favors candidate","interval":"percentile95"},
        "GPU_UUID":UUID,"allow_shared_GPU":True,"forward_precision":"BF16 autocast / Float32 output",
        "C82":C82,"official_VAL_used":False,"repetition":"One fresh60epoch candidate; no old experiments rerun; no automatic extension",
        "source_sha256":{f:sha(ROOT/f) for f in files},
        "protected_readonly_sha256":{f:sha(ROOT/f) for f in protected},
        "comparison_limit":"Repeated development monitor, not blind; CI not corrected for checkpoint selection. One60epoch seed does not establish a theoretical convergence upper bound."}
    atomic(AUDIT/"protocol.json",plan)
    return plan

def read_protocol(check_cache=True):
    p=json.loads((AUDIT/"protocol.json").read_text())
    assert p["campaign"]=="C153" and p["epochs"]==60 and p["updates"]==21120
    assert p["stage1"]["reported_to_chat_before_training"]
    assert sha(ROOT/p["stage1"]["path"])==p["stage1"]["sha256"]
    assert sha(ROOT/p["base_protocol"])==p["base_protocol_sha256"]
    for name,h in p["source_sha256"].items():assert sha(ROOT/name)==h,name
    for name,h in p["protected_readonly_sha256"].items():assert sha(ROOT/name)==h,name
    if check_cache:assert cache_identity(p["panels"],require=True)==p["cache"]
    return p

def preflight(plan):
    torch.set_num_threads(4)
    net=create("aligned_global",plan["initial_seed"],PERMUTATION_SEED)
    assert digest(net)==plan["initial_state_sha256"]
    raw=torch.randn(1,10,256,256);raw[:,[2,3,4,7,8,9]]=raw[:,[2,3,4,7,8,9]].abs()
    x=inputs(torch.randn(1,11,256,256)*.1,torch.randn(1,45,256,256),normalize_diffraction(raw))
    assert x.shape==(1,68,256,256) and torch.count_nonzero(x[:,56:58])==0
    fields=net.global_fields(x,[314])
    physics=torch.cat((x[:,11:56].float(),x[:,58:68].float()),1).clamp(-12,12)
    assert fields.shape==(1,68,256,256) and torch.equal(fields[:,13:],physics)
    assert torch.equal(torch.nn.functional.pixel_shuffle(net.pack(x,[314]),4),fields)
    y=net(x,[314]);assert y.shape==(1,2,256,256) and torch.count_nonzero(y)==0
    with torch.no_grad():net.head.weight.normal_(0,.001)
    net(x,[314]).square().mean().backward()
    grad=net.stem.weight.grad
    assert torch.isfinite(grad).all() and (grad[:,13*16:].abs().sum((0,2,3))>0).all()
    probe=[learning_rate(u,plan) for u in (1,200,201,UPDATES)]
    assert probe[1]==3e-4 and probe[-1]==1e-5
    assert all(learning_rate(u+1,plan)<=learning_rate(u,plan)+1e-16 for u in range(200,UPDATES))
    assert not torch.cuda.is_initialized()
    return {"status":"CPU_preflight_passed","parameters":PARAMETERS,"fresh_C151a_initial_state_verified":True,
        "C82_slots_exact_zero":True,"aligned_all55_physics_preserved":True,"pixel_unshuffle_exactly_reversible":True,
        "zero_head_registered_FSPL_output":True,"all_global_physics_columns_receive_gradient":True,
        "schedule_probe":probe,"CUDA_initialized":False}

@torch.inference_mode()
def evaluate(net,panel,name,references,batch,prediction_path,weights=None,zero_head=False):
    net.eval();n=len(panel.ids)
    mm=np.lib.format.open_memmap(prediction_path,mode="w+",dtype=np.float32,shape=(n,2,256,256))
    sse=np.empty((6,n,2),np.float64);counts=np.empty((6,n,2),np.int64)
    for start in tqdm(range(0,n,batch),desc=f"C153 {name}",mininterval=5):
        stop=min(start+batch,n);sl=slice(start,stop);x,fs,_ld,y,w=panel.batch(sl)
        assert torch.count_nonzero(x[:,56:58])==0
        if zero_head:p=fs
        else:
            with torch.autocast("cuda",dtype=torch.bfloat16):p=fs+net(x,panel.ids[start:stop]).float()
        if not torch.isfinite(p).all():raise RuntimeError("Nonfinite C153 prediction")
        mm[sl]=p.cpu().numpy();error=p.double()-y.double()
        for gi,m in enumerate(groups(w,y,references.baseline[name][sl])):
            sse[gi,start:stop]=torch.where(m,error.square(),0).sum((2,3)).cpu().numpy()
            counts[gi,start:stop]=m.sum((2,3)).cpu().numpy()
    mm.flush();del mm
    assert np.array_equal(counts,references.counts[name])
    cs=counts.sum(1);ss=sse.sum(1);bs=references.sse[name].sum(1)
    result={"rmse":np.sqrt(ss[0]/cs[0]).tolist(),"groups":{}}
    for j,g in enumerate(GROUPS):
        result["groups"][g]={"RMSE":np.sqrt(ss[j]/cs[j]).tolist(),"SSE":ss[j].tolist(),
            "count":cs[j].tolist(),"C82_RMSE":np.sqrt(bs[j]/cs[j]).tolist(),"C82_SSE":bs[j].tolist()}
    if weights is not None:
        ws=(sse[0]*weights[:,None]).sum(0);wc=(counts[0]*weights[:,None]).sum(0)
        result.update(weighted_MSE_objective=float(np.mean(ws/wc)),weighted_RMSE=np.sqrt(ws/wc).tolist())
    terrains,inv=np.unique(np.asarray(panel.terrain),return_inverse=True)
    ts=np.zeros((len(terrains),2));tc=np.zeros((len(terrains),2),np.int64)
    np.add.at(ts,inv,sse[0]);np.add.at(tc,inv,counts[0])
    stats={name+"_map_ids":np.asarray(panel.ids,np.int64),name+"_map_group_sse":sse,name+"_map_group_count":counts,
        name+"_terrain_names":terrains,name+"_map_terrain_index":inv,name+"_terrain_sse":ts,name+"_terrain_count":tc}
    net.train();return result,stats

def evaluation_metrics(row):
    out={"progress/epoch":row["epoch"],"progress/total_epochs":EPOCHS,"progress/updates":row["updates"],
        "time/elapsed_seconds":row["elapsed_seconds"],"time/eta_seconds":row["eta_seconds"],
        "optim/learning_rate":row["learning_rate"],"fit/weighted_loss":row["evaluation"]["fit"]["weighted_MSE_objective"]/25}
    for name,v in row["evaluation"].items():
        for j,b in enumerate(BANDS):
            out[f"{name}/RMSE_{b}_dB"]=v["rmse"][j]
            out[f"{name}/tail_RMSE_{b}_dB"]=v["groups"]["tail"]["RMSE"][j]
    for j,b in enumerate(BANDS):
        out[f"monitor/C82_delta_RMSE_{b}_dB"]=row["evaluation"]["monitor"]["rmse"][j]-C82["monitor"][j]
    return out

def run(resume=False):
    plan=read_protocol();identity=sha(AUDIT/"protocol.json")
    if GPU_UUID!=UUID or os.environ.get("CUDA_VISIBLE_DEVICES")!=UUID or UUID not in APPROVED_GPU_UUIDS:
        raise ValueError("C153 requires the frozen approved full GPU UUID in both environment variables")
    if RUN.exists() and not resume:raise FileExistsError("C153 exists; inspect, do not repeat")
    if resume and (RUN/"result.json").exists():raise ValueError("Completed C153 cannot be resumed")
    RUN.mkdir(parents=True,exist_ok=True);lock=RUN/"active.json";tracking=False
    if lock.exists():
        old=json.loads(lock.read_text())
        try:os.kill(old["pid"],0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError("Active C153 process already exists")
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.write(fd,json.dumps({"pid":os.getpid()}).encode());os.close(fd)
    try:
        snapshot={"device":subprocess.check_output(["nvidia-smi","-i",UUID,"--query-gpu=uuid,name,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip(),
            "co_resident_processes":[v for v in subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True).splitlines() if UUID in v],
            "other_processes_terminated":False}
        check_source_unchanged();gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision("high")
        load_start=time.monotonic();reader=Reader()
        panels={name:Panel(reader,plan["panels"][name]) for name in ("fit","monitor")}
        refs=EvaluationReferences(plan,reader);fit=panels["fit"]
        full=Counter(reader.rows[int(i)]["terrain_id"] for i in np.load(ROOT/"features/c38_dense/fit_map_indices.npy"))
        chosen=Counter(fit.terrain)
        weights=torch.tensor([full[t]/chosen[t] for t in fit.terrain],device="cuda",dtype=torch.float32)
        weights/=weights.mean();weights_np=weights.cpu().numpy().astype(np.float64)
        den=torch.zeros(2,device="cuda",dtype=torch.float64)
        for start in range(0,len(fit.ids),32):
            den+=(fit.w[start:start+32]*weights[start:start+32,None,None,None]).sum((0,2,3),dtype=torch.float64)
        den=(den/STEPS).float()
        net=create("aligned_global",plan["initial_seed"],PERMUTATION_SEED).cuda().to(memory_format=torch.channels_last)
        assert digest(net)==plan["initial_state_sha256"]
        opt=torch.optim.AdamW(net.parameters(),lr=plan["learning_rate"],weight_decay=plan["weight_decay"],
            betas=tuple(plan["betas"]),eps=plan["eps"],fused=True)
        order_rng=torch.Generator().manual_seed(plan["data_seed"]);torch.cuda.manual_seed(plan["initial_seed"])
        last_epoch=0;updates=0;best_epoch=0;best_mean=float("inf");history=[];elapsed_before=0.
        if resume:
            cp=torch.load(RUN/"last.pt",map_location="cpu",weights_only=False)
            assert cp["protocol_sha256"]==identity and cp["arm"]==ARM
            net.load_state_dict(cp["state"]);opt.load_state_dict(cp["optimizer"])
            assert digest(net)==cp["net_state_sha256"]
            last_epoch=cp["epoch"];updates=cp["updates"];best_epoch=cp["best_epoch"];best_mean=cp["best_mean"]
            history=cp["history"];elapsed_before=cp["elapsed_seconds"]
            order_rng.set_state(cp["order_generator"]);torch.set_rng_state(cp["torch_rng"]);torch.cuda.set_rng_state(cp["cuda_rng"])
            assert updates==last_epoch*STEPS and cp["scheduler"]=={"kind":"linear_warmup_cosine_formula","updates":updates,"total_updates":UPDATES}
            if updates:
                assert {int(v["step"].item()) for v in opt.state.values()}=={updates}
                assert len(opt.state)==len(list(net.parameters()))
                assert math.isclose(opt.param_groups[0]["lr"],learning_rate(updates,plan),rel_tol=1e-12)
            archive=RUN/f"selected_epoch_{best_epoch:04d}"
            saved=torch.load(Path(str(archive)+".pt"),map_location="cpu",weights_only=False)
            assert saved["epoch"]==best_epoch and saved["protocol_sha256"]==identity
            export=json.loads(Path(str(archive)+"_export.json").read_text())
            link_atomic(Path(str(archive)+".pt"),RUN/"selected.pt")
            for name in panels:
                dest=RUN/f"selected_{name}_prediction.npy"
                link_atomic(Path(str(archive)+f"_{name}_prediction.npy"),dest)
                atomic(dest.with_suffix(".json"),{k:v for k,v in export["panels"][name].items() if k!="file"})
            atomic(RUN/"selected_export.json",export)
        config={"campaign":"C153","arm":ARM,"arm_label":"C153_standalone_c151a","protocol_sha256":identity,
            "protocol":plan,"panels":plan["panels"],"parameters":PARAMETERS,"GPU":gpu,
            "launch_device_snapshot":snapshot,"prior":"registered_FSPL","official_VAL_used":False,
            "_cc_dashboard":{"step_unit":"epoch","total_epochs":EPOCHS,"scope":"固定开发monitor628图/128地形；C82仅作计分参照",
                "baseline_source":"audit/c145_capacity/comparison_summary.json","C82":C82,
                "method_note":"独立C151a aligned_global：registered FSPL + Field，C82槽位为零；60轮单阶段训练"}}
        if resume:
            assert json.loads((RUN/"config.json").read_text())["protocol_sha256"]==identity
        else:atomic(RUN/"config.json",config)
        oldtrack=json.loads((RUN/"tracking.json").read_text()) if resume else {}
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();data_rng=order_rng.get_state()
        kw={"parent_run_id":oldtrack["run_id"],"parent_checkpoint":str(RUN/"last.pt"),"start_step":last_epoch} if resume else {}
        swan=cc_tracking.init(name="C153_standalone_c151a"+(" resume" if resume else ""),config=config,campaign="C153",
            description="One fresh60epoch aligned C151a Field, registered FSPL output prior, no C82 inputs; server-local only.",
            tags=["standalone","aligned-global","registered-FSPL","60epochs"],**kw)
        tracking=True;atomic(RUN/"tracking.json",{"run_id":swan.id,"dir":str(swan.dir),"step_unit":"epoch",
            "start_step":last_epoch,"parent_run_id":oldtrack.get("run_id")})
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng);order_rng.set_state(data_rng)
        assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),cuda_rng) and torch.equal(order_rng.get_state(),data_rng)
        start=time.monotonic();load_seconds=start-load_start;start_epoch=last_epoch;eta_start=time.monotonic()
        atomic(RUN/"restoration.json",{"resume":resume,"epoch":last_epoch,"updates":updates,"model_state_sha256":digest(net),
            "optimizer_state_entries":len(opt.state),"scheduler_kind":"linear_warmup_cosine_formula",
            "torch_rng_sha256":hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest(),
            "CUDA_rng_sha256":hashlib.sha256(cuda_rng.cpu().numpy().tobytes()).hexdigest(),
            "order_rng_sha256":hashlib.sha256(data_rng.numpy().tobytes()).hexdigest(),
            "RNG_restored_after_SwanLab_init":True,"C82_model_checkpoint_loaded":False,"data_load_seconds":load_seconds})
        def elapsed():return elapsed_before+time.monotonic()-start
        def eta(epoch_fraction):
            done=epoch_fraction-start_epoch
            return max(0.,(time.monotonic()-eta_start)/done*(EPOCHS-epoch_fraction)) if done>0 else 0.
        def state():
            h=digest(net)
            return {"state":net.state_dict(),"optimizer":opt.state_dict(),
                "scheduler":{"kind":"linear_warmup_cosine_formula","updates":updates,"total_updates":UPDATES},
                "order_generator":order_rng.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
                "arm":ARM,"epoch":last_epoch,"absolute_epoch":last_epoch,"updates":updates,"phase":1,
                "best_epoch":best_epoch,"best_mean":best_mean,"history":deepcopy(history),"elapsed_seconds":elapsed(),
                "protocol_sha256":identity,"net_state_sha256":h,"predictor_sha256":predictor_digest(h,plan)}
        def score_epoch(epoch,zero_head=False):
            nonlocal best_epoch,best_mean
            paths={name:RUN/f"epoch_{epoch:04d}_{name}.partial.npy" for name in panels}
            values={};stats={"group_names":np.asarray(VERIFIED_GROUPS)}
            for name,panel in panels.items():
                values[name],st=evaluate(net,panel,name,refs,plan["batch"],paths[name],
                    weights=weights_np if name=="fit" else None,zero_head=zero_head);stats.update(st)
            mean=float(np.mean(values["monitor"]["rmse"]));improved=mean<best_mean
            if improved:best_mean=mean;best_epoch=epoch
            h=digest(net)
            row={"epoch":epoch,"updates":updates,"phase":1,"evaluation":values,"best_epoch":best_epoch,
                "learning_rate":opt.param_groups[0]["lr"],"elapsed_seconds":elapsed(),"eta_seconds":eta(epoch),
                "net_state_sha256":h,"predictor_sha256":predictor_digest(h,plan)}
            history.append(row);atomic(RUN/"history.json",history)
            np.savez_compressed(RUN/f"epoch_{epoch:04d}_stats.npz",**stats)
            if epoch==0:save(RUN/"initial.pt",state())
            if improved:
                save(RUN/"selected.pt",state())
                export={"tag":"selected","checkpoint":"selected.pt","checkpoint_sha256":sha(RUN/"selected.pt"),
                    "model_sha256":row["predictor_sha256"],"net_state_sha256":h,"prediction_dtype":"float32",
                    "forward_precision":"BF16 autocast","training_state":{"absolute_epoch":epoch,"epoch":epoch},
                    "prior":"registered_FSPL","C82_in_model":False,"panels":{}}
                for name,path in paths.items():
                    dest=RUN/f"selected_{name}_prediction.npy";path.replace(dest)
                    info={"model_sha256":row["predictor_sha256"],"panel_ids_sha256":ordered_hash(plan["panels"][name]),
                        "shape":[len(plan["panels"][name]),2,256,256],"prediction_dtype":"float32","forward_precision":"BF16 autocast",
                        "file_size_bytes":dest.stat().st_size,"prediction_sha256":sha(dest)}
                    atomic(dest.with_suffix(".json"),info);export["panels"][name]={"file":dest.name,**info}
                atomic(RUN/"selected_export.json",export)
                archive=RUN/f"selected_epoch_{epoch:04d}";link_atomic(RUN/"selected.pt",Path(str(archive)+".pt"))
                for name in panels:link_atomic(RUN/f"selected_{name}_prediction.npy",Path(str(archive)+f"_{name}_prediction.npy"))
                atomic(Path(str(archive)+"_export.json"),export)
            else:
                for path in paths.values():path.unlink(missing_ok=True)
            cc_tracking.log(evaluation_metrics(row),step=epoch)
            print(json.dumps({"event":"evaluation","epoch":epoch,"fit":values["fit"]["rmse"],"monitor":values["monitor"]["rmse"],"best_epoch":best_epoch}),flush=True)
        if not resume:score_epoch(0,zero_head=True);save(RUN/"last.pt",state())
        eta_start=time.monotonic()
        for epoch in range(last_epoch+1,EPOCHS+1):
            order=torch.randperm(len(fit.ids),generator=order_rng)
            sums=torch.zeros(4,device="cuda",dtype=torch.float64);net.train()
            progress=tqdm(enumerate(order.split(plan["batch"]),1),total=STEPS,desc=f"C153 Epoch{epoch}/{EPOCHS}",mininterval=5)
            for bi,positions in progress:
                ix=positions.cuda();x,fs,_ld,y,w=fit.batch(ix)
                assert torch.count_nonzero(x[:,56:58])==0
                rate=learning_rate(updates+1,plan)
                for group in opt.param_groups:group["lr"]=rate
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    prediction=fs+net(x,[fit.ids[int(j)] for j in positions.tolist()]).float()
                if updates==0:assert torch.equal(prediction,fs)
                err=torch.where(w,prediction-y,0.);ww=weights[ix,None,None,None]
                numerator=(err.square()*ww).sum((0,2,3));count=(w*ww).sum((0,2,3))
                loss=(numerator/den).mean()/25
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite C153 loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),plan["clip_grad_norm"])
                if not torch.isfinite(norm):raise RuntimeError("Nonfinite C153 gradient")
                opt.step();updates+=1;sums+=torch.cat((numerator.detach().double(),count.double()))
                if bi%40==0 or bi==STEPS:
                    atomic(RUN/"progress.json",{"stage":"training","arm":ARM,"epoch":epoch,"epochs":EPOCHS,
                        "batch":bi,"batches":STEPS,"updates":updates,"total_updates":UPDATES,"learning_rate":rate,
                        "elapsed_seconds":elapsed(),"eta_seconds":eta(epoch-1+bi/STEPS)})
                    progress.set_postfix(loss=f"{float(loss):.4f}")
            last_epoch=epoch
            online=float((sums[:2]/sums[2:]).mean()/25)
            cc_tracking.log({"train/optimization_loss":online,"optim/learning_rate":rate,
                "progress/epoch":epoch,"progress/total_epochs":EPOCHS,"progress/updates":updates,
                "time/elapsed_seconds":elapsed(),"time/eta_seconds":eta(epoch)},step=epoch)
            if epoch%4==0:score_epoch(epoch)
            save(RUN/"last.pt",state())
        assert updates==UPDATES and math.isclose(rate,1e-5,rel_tol=1e-12)
        selected=torch.load(RUN/"selected.pt",map_location="cpu",weights_only=False)
        result={"status":"complete","arm":ARM,"epochs":EPOCHS,"updates":updates,"selected_epoch":selected["epoch"],
            "selected_predictor_sha256":selected["predictor_sha256"],"last_predictor_sha256":predictor_digest(digest(net),plan),
            "selected_monitor_RMSE_dB":next(r["evaluation"]["monitor"]["rmse"] for r in history if r["epoch"]==selected["epoch"]),
            "last_monitor_RMSE_dB":history[-1]["evaluation"]["monitor"]["rmse"],"elapsed_seconds":elapsed(),
            "data_load_seconds":load_seconds,"protocol_sha256":identity,"independent_prediction_verification":"pending",
            "automatic_budget_extension":False,"official_VAL_used":False}
        atomic(RUN/"result.json",result);atomic(RUN/"progress.json",{"stage":"complete",**result})
        print(json.dumps({"event":"complete",**result}),flush=True)
    finally:
        if tracking:cc_tracking.finish()
        lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        p=make_protocol();print(json.dumps({"status":"prepared","epochs":p["epochs"],"updates":p["updates"],"parameters":PARAMETERS}));return
    if args.preflight:
        p=read_protocol(check_cache=False);v=preflight(p);atomic(AUDIT/"preflight.json",v);print(json.dumps(v));return
    run(args.resume)
if __name__=="__main__":main()
