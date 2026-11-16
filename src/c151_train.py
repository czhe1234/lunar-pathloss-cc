"""C151a single-arm runner: active global spatial-alignment A/B, fixed12epochs.

C150 data/schedule/reference helpers are imported read-only. No C150 globals,
sources or runtime records are mutated; every forward has explicit map IDs.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cc_core import GPU_UUID, APPROVED_GPU_UUIDS, gpu_guard, check_source_unchanged
from src.c151_field import create, digest, PERMUTATION_SEED, PARAMETERS, CHANNELS
from src.c150_train import Reader, Panel, EvaluationReferences, cache_identity, schedule_lr
from src import cc_tracking
import numpy as np
import torch
from tqdm import tqdm

AUDIT = ROOT / "audit/c151_global"
RUN_ROOT = ROOT / "runs/c151_global"
CACHE = ROOT / "features/c150_diffraction/part0of1"
GROUPS = ("all", "tail", "under_tail", "over_tail", "non_tail", "C82_good")
VERIFIED_GROUPS = ("all", "C82_abs_error_ge_5dB", "C82_underestimated_loss_ge_5dB",
                   "C82_overestimated_loss_ge_5dB", "C82_abs_error_lt_5dB", "C82_abs_error_le_1dB")
CHANNEL_NAMES = ["nu415", "t415", "I415", "Ld415", "Lb415",
                 "nu5775", "t5775", "I5775", "Ld5775", "Lb5775", "FSPL415", "FSPL5775"]
UUIDS = {"A": "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
         "B": "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0"}
BANDS = ("415MHz", "5p775GHz")
C82 = {"fit": [2.184470152566454, 2.471133742110456],
       "monitor": [2.1463905073492664, 2.41801448492553]}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path, value):
    temporary = path.with_suffix(".partial.pt")
    torch.save(value, temporary); temporary.replace(path)


def link_atomic(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".link.partial")
    temporary.unlink(missing_ok=True)
    os.link(source,temporary);temporary.replace(destination)


def ordered_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def predictor_digest(net_hash, arm, plan):
    value={"net_state_sha256":net_hash,"arm":arm,"global_mode":plan["arm_modes"][arm],
           "permutation_seed":plan["permutation_seed"],"base_prior_arm":plan["base_prior_arm"],
           "prior":plan["prior_description"],"feature_cache_sha256":plan["cache"]["features_sha256"]}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def make_protocol(bind_c150=False):
    if RUN_ROOT.exists():
        raise FileExistsError("Do not rewrite a protocol after any C151 arm directory exists")
    c150_path=ROOT/"audit/c150_diffraction_training/protocol.json"
    base=json.loads(c150_path.read_text())
    plan=deepcopy(base)
    plan.update(campaign="C151a",created_utc=datetime.now(timezone.utc).isoformat(),
        question="Does spatially aligned global physics help versus an active spatially shuffled global control?",
        ready=False,base_prior_arm=None,prior_description="PENDING verified C150 joint-mean winner",
        historical_control=None,C150_protocol=str(c150_path.relative_to(ROOT)),C150_protocol_sha256=sha(c150_path),
        arm_modes={"A":"shuffled_global","B":"aligned_global"},
        arms={"A":"active spatially shuffled global physics","B":"spatially aligned global physics"},
        channel_contract="raw11 basic +45 solver +2 fixed zero +10 diffraction; global=13 derived basic+55 physics",
        parameters=PARAMETERS,permutation_seed=PERMUTATION_SEED,
        global_permutation="PCG64 complete65536pixel permutation seeded by SHA256(v1|seed|map_id), same order across55 channels",
        global_input_channels=68,
        FFN_hidden={"global":[715]*8,"local":[119,118],"refine":[222,221]},
        attribution="Both arms have the same active dense stem/rebalanced FFNs; the A/B effect is global spatial alignment, not literal blind-versus-informed capacity.",
        controlled_difference="Only the55globalphysicsfields are jointly spatially permuted in A and aligned in B; derived13/global architecture/local/TX-FiLM/commonprior/initialweights/order/budget are identical.",
        C150_reference_role="Existing C150 winner is the historical blind-global reference; do not rerun it or load its trained weights.",
        repetition="One fresh fixed12epoch active alignment pair; no old ordered-ray replay, seedsweep or automatic extension")
    model=create(plan["arm_modes"]["A"],plan["initial_seed"],plan["permutation_seed"])
    plan["initial_state_sha256"]=digest(model)
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:
        raise ValueError("C151 active model is not exactly the C145 parameter budget")
    sources=["src/c151_field.py","src/c151_train.py",*base["source_sha256"]]
    sources=list(dict.fromkeys(sources+["src/c150_report.py"]))
    plan["source_sha256"]={name:sha(ROOT/name) for name in sources}
    report=ROOT/"src/c151_report.py"
    plan["postprocess_source_sha256_at_prepare"]=sha(report) if report.exists() else None
    if bind_c150:
        summary_path=ROOT/"audit/c150_diffraction_training/comparison_summary.json"
        summary=json.loads(summary_path.read_text())
        if summary.get("status")!="verified" or summary["protocol_sha256"]!=sha(c150_path):
            raise ValueError("C150 paired independent comparison must be verified before selecting the common prior")
        winner=summary["lowest_development_joint_mean_arm"]
        if winner not in ("A","B"):raise ValueError("C150 prior winner is invalid")
        item=summary["arms"][winner]
        historical=ROOT/"runs/c150_diffraction_prior"/winner
        evidence=item["evidence"]
        for field,file in [("verified_json_sha256","selected_verified.json"),("statistic_sha256","selected_terrain_sse.npz")]:
            if sha(historical/file)!=evidence[field]:
                raise ValueError("C150 historical reference evidence changed")
        plan.update(ready=True,base_prior_arm=winner,prior_description=base["arms"][winner],
            historical_control=str(historical.relative_to(ROOT)),
            prior_binding={"rule":"minimum joint mean monitor RMSE of independently verified C150 selected checkpoints",
                "summary_source":str(summary_path.relative_to(ROOT)),"summary_sha256_at_binding":sha(summary_path),
                "C150_selected_epoch":item["selected_epoch"],"C150_predictor_sha256":item["predictor_sha256"],
                "verified_json_sha256":evidence["verified_json_sha256"],"statistic_sha256":evidence["statistic_sha256"],
                "weights_loaded_for_initialization":False})
    atomic(AUDIT/"protocol.json",plan)
    return plan


def read_protocol(require_cache=True):
    plan=json.loads((AUDIT/"protocol.json").read_text())
    if sha(ROOT/plan["directive"])!=plan["directive_sha256"]:raise ValueError("User directive identity changed")
    if sha(ROOT/plan["base_protocol"])!=plan["base_protocol_sha256"]:raise ValueError("Fixed split anchor changed")
    if sha(ROOT/plan["C150_protocol"])!=plan["C150_protocol_sha256"]:raise ValueError("C150 common input/schedule anchor changed")
    for name,expected in plan["source_sha256"].items():
        if sha(ROOT/name)!=expected:raise ValueError("Frozen C151 dependency changed: "+name)
    if plan["epochs"]!=12 or plan["updates_per_arm"]!=4224 or plan["parameters"]!=PARAMETERS:
        raise ValueError("Fixed C151 budget differs")
    if require_cache:
        if plan.get("ready") is not True or plan.get("base_prior_arm") not in ("A","B"):
            raise ValueError("Bind the independently verified C150 prior before launching C151")
        if cache_identity(plan["panels"],require=True)!=plan["cache"]:raise ValueError("C150 raw feature identity changed")
        historical=ROOT/plan["historical_control"]
        for field,file in [("verified_json_sha256","selected_verified.json"),("statistic_sha256","selected_terrain_sse.npz")]:
            if sha(historical/file)!=plan["prior_binding"][field]:raise ValueError("Historical C150 evidence changed")
    return plan


def prior(fspl, deygout, base_prior_arm):
    return fspl if base_prior_arm=="A" else fspl+deygout


def groups(valid, target, baseline):
    residual=target.double()-baseline.double();tail=residual.abs()>=5
    return (valid,valid&tail,valid&(residual>=5),valid&(residual<=-5),
            valid&~tail,valid&(residual.abs()<=1))


@torch.inference_mode()
def evaluate(net, panel, name, references, arm, base_prior_arm, batch, prediction_path, weights=None, zero_head=False):
    net.eval();n=len(panel.ids)
    mm = np.lib.format.open_memmap(prediction_path,mode="w+",dtype=np.float32,shape=(n,2,256,256))
    sse=np.empty((6,n,2),np.float64);counts=np.empty((6,n,2),np.int64)
    for start in tqdm(range(0,n,batch),desc=f"C151a{arm} {name}",mininterval=5):
        stop=min(start+batch,n);sl=slice(start,stop);x,fs,ld,y,w=panel.batch(sl)
        if zero_head:p=prior(fs,ld,base_prior_arm)
        else:
            with torch.autocast("cuda",dtype=torch.bfloat16):p=prior(fs,ld,base_prior_arm)+net(x,panel.ids[start:stop]).float()
        if not torch.isfinite(p).all():raise RuntimeError("Nonfinite C151a prediction")
        mm[sl]=p.cpu().numpy();error=p.double()-y.double()
        for gi,mask in enumerate(groups(w,y,references.baseline[name][sl])):
            sse[gi,start:stop]=torch.where(mask,error.square(),0).sum((2,3)).cpu().numpy()
            counts[gi,start:stop]=mask.sum((2,3)).cpu().numpy()
    mm.flush();del mm
    if not np.array_equal(counts,references.counts[name]):
        raise ValueError("Candidate group support differs from frozen C145 score")
    cs=counts.sum(1);ss=sse.sum(1);bs=references.sse[name].sum(1)
    result={"rmse":np.sqrt(ss[0]/cs[0]).tolist(),"groups":{}}
    for i,g in enumerate(GROUPS):
        result["groups"][g]={"RMSE":np.sqrt(ss[i]/cs[i]).tolist(),"SSE":ss[i].tolist(),"count":cs[i].tolist(),
                              "C82_RMSE":np.sqrt(bs[i]/cs[i]).tolist(),"C82_SSE":bs[i].tolist()}
    if weights is not None:
        ws=(sse[0]*weights[:,None]).sum(0);wc=(counts[0]*weights[:,None]).sum(0)
        result["weighted_MSE_objective"]=float(np.mean(ws/wc))
        result["weighted_RMSE"]=np.sqrt(ws/wc).tolist()
    terrains,inverse=np.unique(np.asarray(panel.terrain),return_inverse=True)
    ts=np.zeros((len(terrains),2),np.float64);tc=np.zeros((len(terrains),2),np.int64)
    np.add.at(ts,inverse,sse[0]);np.add.at(tc,inverse,counts[0])
    statistics={name+"_map_ids":np.asarray(panel.ids,np.int64),name+"_map_group_sse":sse,
                name+"_map_group_count":counts,name+"_terrain_names":terrains,
                name+"_map_terrain_index":inverse,name+"_terrain_sse":ts,name+"_terrain_count":tc}
    net.train();return result,statistics


def metric_log(row):
    value={"progress/epoch":row["epoch"],"progress/total_epochs":12,
           "progress/updates":row["updates"],"time/elapsed_seconds":row["elapsed_seconds"],
           "time/eta_seconds":row["eta_seconds"],"optim/learning_rate":row["learning_rate"],
           "optim/phase":row["phase"],"train/data_loss":row["evaluation"]["fit"]["weighted_MSE_objective"]/25}
    for panel,metric in row["evaluation"].items():
        for index,band in enumerate(BANDS):
            value[f"{panel}/RMSE_{band}_dB"]=metric["rmse"][index]
            value[f"{panel}/tail_RMSE_{band}_dB"]=metric["groups"]["tail"]["RMSE"][index]
            value[f"{panel}/C82_good_RMSE_{band}_dB"]=metric["groups"]["C82_good"]["RMSE"][index]
    return value


def optimizer_for(net,plan):
    return torch.optim.AdamW(net.parameters(),lr=plan["learning_rate"],weight_decay=plan["weight_decay"],fused=True)


def second_scheduler(optimizer,plan):
    p=plan["phase2"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=p["factor"],patience=p["patience"],
        threshold=p["threshold"],threshold_mode="rel",min_lr=p["minimum_lr"])


def preflight(plan):
    a=create(plan["arm_modes"]["A"],plan["initial_seed"],plan["permutation_seed"])
    b=create(plan["arm_modes"]["B"],plan["initial_seed"],plan["permutation_seed"])
    if digest(a)!=digest(b) or digest(a)!=plan["initial_state_sha256"]:raise ValueError("C151 A/B initialization differs")
    if sum(p.numel() for p in a.parameters())!=15214786:raise ValueError("C151 parameter budget differs")
    count=65536
    spatial=torch.linspace(.1,.9,count).reshape(1,1,256,256)
    x=spatial.expand(1,68,256,256).clone()+torch.arange(68)[None,:,None,None]*.01
    x[:,56:58]=0
    ga=a.global_fields(x,[314]);gb=b.global_fields(x,[314])
    assert ga.shape==gb.shape==(1,68,256,256)
    assert torch.equal(ga[:,:13],gb[:,:13])
    order=a.permutation(314)
    expected=gb[:,13:].flatten(2).index_select(2,order).reshape(1,55,256,256)
    assert torch.equal(ga[:,13:],expected)
    assert torch.equal(a.global_fields(x,[314]),ga)
    assert torch.equal(torch.sort(order).values,torch.arange(count))
    assert not torch.equal(order,a.permutation(315))
    assert torch.equal(torch.sort(ga[0,13].flatten()).values,torch.sort(gb[0,13].flatten()).values)
    for model,fields in ((a,ga),(b,gb)):
        model.zero_grad(set_to_none=True)
        packed=torch.nn.functional.pixel_unshuffle(fields,4)[:,:,:1,:1]
        model.stem(packed).sum().backward()
        assert (model.stem.weight.grad[:,13*16:].abs().sum((0,2,3))>0).all()
    assert not torch.cuda.is_initialized()
    return {"status":"CPU_preflight_passed","parameters":15214786,"initial_state_sha256":digest(a),
            "A_B_weights_identical":True,"global_channels":68,"physics_channels":55,
            "joint_permutation_bijective_deterministic":True,"terrain_global_channels_identical":True,
            "global_physics_stem_columns_active_both_arms":True,"prior_ready":plan["ready"],
            "base_prior_arm":plan["base_prior_arm"],"CUDA_initialized":False}


def run(arm,resume=False):
    plan=read_protocol();run_dir=RUN_ROOT/arm;identity=sha(AUDIT/"protocol.json")
    if GPU_UUID!=UUIDS[arm] or os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU_UUID or GPU_UUID not in APPROVED_GPU_UUIDS:
        raise ValueError("Arm launch must use its frozen approved full GPU UUID")
    if run_dir.exists() and not resume:raise FileExistsError("Inspect C151a arm; use explicit --resume, never repeat")
    if resume and (run_dir/"result.json").exists() and json.loads((run_dir/"result.json").read_text()).get("status")=="complete":
        raise ValueError("This C151a arm already completed")
    run_dir.mkdir(parents=True,exist_ok=True);lock=run_dir/"active.json"
    if lock.exists():
        old=json.loads(lock.read_text())
        try:os.kill(old["pid"],0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError("An arm process/lock is active; inspect before resuming")
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,json.dumps({"pid":os.getpid()}).encode());os.close(fd)
    tracking=False
    try:
        gpu_snapshot={"device":subprocess.check_output(["nvidia-smi","-i",GPU_UUID,"--query-gpu=uuid,name,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip(),
            "co_resident_processes":[v for v in subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True).splitlines() if GPU_UUID in v],
            "allow_shared":True,"other_processes_terminated":False}
        check_source_unchanged()
        gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision("high")
        load_started=time.monotonic();reader=Reader()
        panels={name:Panel(reader,plan["panels"][name]) for name in ("fit","monitor")}
        refs=EvaluationReferences(plan,reader);fit=panels["fit"]
        full=Counter(reader.rows[int(i)]["terrain_id"] for i in np.load(ROOT/"features/c38_dense/fit_map_indices.npy"))
        chosen=Counter(fit.terrain)
        weights=torch.tensor([full[t]/chosen[t] for t in fit.terrain],device="cuda",dtype=torch.float32)
        weights/=weights.mean();weights_np=weights.cpu().numpy().astype(np.float64)
        den=torch.zeros(2,device="cuda",dtype=torch.float64)
        for start in range(0,len(fit.ids),32):
            den+=(fit.w[start:start+32]*weights[start:start+32,None,None,None]).sum((0,2,3),dtype=torch.float64)
        den=(den/352).float()
        net=create(plan["arm_modes"][arm],plan["initial_seed"],plan["permutation_seed"]).cuda().to(memory_format=torch.channels_last)
        assert digest(net)==plan["initial_state_sha256"]
        opt=optimizer_for(net,plan);scheduler=None;generator=torch.Generator().manual_seed(plan["data_seed"])
        torch.cuda.manual_seed(plan["initial_seed"])
        phase=1;last_epoch=0;updates=0;best_epoch=0;best_mean=float("inf");history=[];elapsed_before=0.
        if resume:
            checkpoint=torch.load(run_dir/"last.pt",map_location="cpu",weights_only=False)
            if checkpoint["protocol_sha256"]!=identity or checkpoint["arm"]!=arm:raise ValueError("Resume protocol/arm differs")
            net.load_state_dict(checkpoint["state"],strict=True);opt.load_state_dict(checkpoint["optimizer"])
            if digest(net)!=checkpoint["net_state_sha256"]:raise ValueError("Restored model state identity differs")
            phase=checkpoint["phase"]
            if phase==2:
                scheduler=second_scheduler(opt,plan);scheduler.load_state_dict(checkpoint["scheduler"])
            last_epoch=checkpoint["epoch"];updates=checkpoint["updates"]
            best_epoch=checkpoint["best_epoch"];best_mean=checkpoint["best_mean"];history=checkpoint["history"]
            generator.set_state(checkpoint["order_generator"]);torch.set_rng_state(checkpoint["torch_rng"])
            torch.cuda.set_rng_state(checkpoint["cuda_rng"]);elapsed_before=checkpoint["elapsed_seconds"]
            if updates!=last_epoch*352:raise ValueError("Recovered update budget differs")
            selected_path=run_dir/"selected.pt"
            selected_epoch=torch.load(selected_path,map_location="cpu",weights_only=False)["epoch"] if selected_path.exists() else None
            if selected_epoch!=best_epoch:
                archive=run_dir/f"selected_epoch_{best_epoch:04d}"
                archive_checkpoint=Path(str(archive)+".pt")
                archived_export=json.loads(Path(str(archive)+"_export.json").read_text())
                restore=torch.load(archive_checkpoint,map_location="cpu",weights_only=False)
                if restore["epoch"]!=best_epoch or restore["protocol_sha256"]!=identity:raise ValueError("Best checkpoint recovery identity differs")
                link_atomic(archive_checkpoint,selected_path)
                for name in ("fit","monitor"):
                    destination=run_dir/f"selected_{name}_prediction.npy"
                    link_atomic(Path(str(archive)+f"_{name}_prediction.npy"),destination)
                    atomic(destination.with_suffix(".json"),{k:v for k,v in archived_export["panels"][name].items() if k!="file"})
                atomic(run_dir/"selected_export.json",archived_export)
            expected_opt_steps=(last_epoch if phase==1 else last_epoch-8)*352
            steps={int(v["step"].item()) for v in opt.state.values()}
            if expected_opt_steps>0 and (steps!={expected_opt_steps} or len(opt.state)!=len(list(net.parameters()))):
                raise ValueError("Recovered Adam state count/step differs")
        config={"campaign":"C151a","arm":arm,"arm_label":plan["arm_modes"][arm],"protocol_sha256":identity,"protocol":plan,"panels":plan["panels"],
                "GPU":gpu,"launch_device_snapshot":gpu_snapshot,"parameters":plan["parameters"],"initial_state_sha256":plan["initial_state_sha256"],
                "prior":plan["prior_description"],"base_prior_arm":plan["base_prior_arm"],"official_VAL_used":False,
                "_cc_dashboard":{"step_unit":"epoch","total_epochs":12,
                    "scope":"固定开发monitor628图/128地形；C82 beta-fit80和selection48均已用于开发",
                    "baseline_source":"audit/c145_capacity/comparison_summary.json","C82":C82}}
        if resume:
            old_config=json.loads((run_dir/"config.json").read_text())
            if old_config["protocol_sha256"]!=identity or old_config["arm"]!=arm or old_config["panels"]!=plan["panels"]:
                raise ValueError("Run config identity differs")
        else:atomic(run_dir/"config.json",config)
        rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state();rng_order=generator.get_state()
        previous=json.loads((run_dir/"tracking.json").read_text()) if resume else {}
        kwargs={"parent_run_id":previous["run_id"],"parent_checkpoint":str(run_dir/"last.pt"),"start_step":last_epoch} if resume else {}
        swan=cc_tracking.init(name=f"C151a{arm} "+plan["arm_modes"][arm]+(" resume" if resume else ""),
            config=config,campaign="C151a",description=plan["arm_modes"][arm]+"; common "+plan["prior_description"]+"; fresh12epochs; local only.",
            tags=["teacher-free","global-physics-alignment",f"arm-{arm}"],**kwargs)
        tracking=True;atomic(run_dir/"tracking.json",{"run_id":swan.id,"dir":str(swan.dir),"step_unit":"epoch","start_step":last_epoch,
                                                   "parent_run_id":previous.get("run_id")})
        torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda);generator.set_state(rng_order)
        if not (torch.equal(torch.get_rng_state(),rng_cpu) and torch.equal(torch.cuda.get_rng_state(),rng_cuda)
                and torch.equal(generator.get_state(),rng_order)):
            raise ValueError("RNG restore after tracker init failed")
        started=time.monotonic();load_seconds=started-load_started
        atomic(run_dir/"restoration.json",{"resume":resume,"epoch":last_epoch,"updates":updates,"phase":phase,
             "model_state_sha256":digest(net),"initial_state_expected":plan["initial_state_sha256"],
             "torch_rng_sha256":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
             "order_rng_sha256":hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
             "CUDA_rng_sha256":hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
             "data_load_seconds":load_seconds,"GPU":gpu,"launch_device_snapshot":gpu_snapshot,"RNG_restored_after_SwanLab_init":True})
        def elapsed():return elapsed_before+time.monotonic()-started
        def state():
            return {"state":net.state_dict(),"optimizer":opt.state_dict(),
                "scheduler":scheduler.state_dict() if scheduler else {"kind":"C123_formula","cosine_horizon_epochs":8},
                "order_generator":generator.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
                "arm":arm,"global_mode":plan["arm_modes"][arm],"permutation_seed":plan["permutation_seed"],
                "base_prior_arm":plan["base_prior_arm"],"epoch":last_epoch,"updates":updates,"phase":phase,"best_epoch":best_epoch,"best_mean":best_mean,
                "history":deepcopy(history),"elapsed_seconds":elapsed(),"protocol_sha256":identity,
                "net_state_sha256":digest(net),"predictor_sha256":predictor_digest(digest(net),arm,plan)}
        def score_epoch(epoch,zero_head=False):
            nonlocal best_epoch,best_mean,history
            pred_paths={name:run_dir/f"epoch_{epoch:04d}_{name}.partial.npy" for name in panels}
            evaluation={};stats={"group_names":np.asarray(VERIFIED_GROUPS)}
            for name,panel in panels.items():
                evaluation[name],st=evaluate(net,panel,name,refs,arm,plan["base_prior_arm"],plan["batch"],pred_paths[name],
                    weights=weights_np if name=="fit" else None,zero_head=zero_head)
                stats.update(st)
            mean=float(np.mean(evaluation["monitor"]["rmse"]));improved=mean<best_mean
            if improved:best_mean=mean;best_epoch=epoch
            if phase==2 and scheduler is not None:scheduler.step(evaluation["fit"]["weighted_MSE_objective"])
            row={"epoch":epoch,"updates":updates,"phase":phase,"evaluation":evaluation,"best_epoch":best_epoch,
                 "learning_rate":opt.param_groups[0]["lr"],"elapsed_seconds":elapsed(),
                 "eta_seconds":0. if epoch==12 else ((time.monotonic()-eta_clock_start)/(epoch-last_epoch_start)*(12-epoch) if epoch>last_epoch_start else 0.),
                 "net_state_sha256":digest(net),"predictor_sha256":predictor_digest(digest(net),arm,plan)}
            history.append(row);atomic(run_dir/"history.json",history)
            np.savez_compressed(run_dir/f"epoch_{epoch:04d}_stats.npz",**stats)
            if epoch==0:save(run_dir/"initial.pt",state())
            if improved:
                save(run_dir/"selected.pt",state())
                exported={"tag":"selected","checkpoint":"selected.pt","checkpoint_sha256":sha(run_dir/"selected.pt"),
                    "model_sha256":row["predictor_sha256"],"net_state_sha256":row["net_state_sha256"],
                    "prediction_dtype":"float32","forward_precision":"BF16 autocast","training_state":{"absolute_epoch":epoch,"epoch":epoch},
                    "prior":plan["prior_description"],"base_prior_arm":plan["base_prior_arm"],
                    "global_mode":plan["arm_modes"][arm],"permutation_seed":plan["permutation_seed"],"panels":{}}
                for name,path in pred_paths.items():
                    target=run_dir/f"selected_{name}_prediction.npy";path.replace(target)
                    info={"model_sha256":row["predictor_sha256"],"panel_ids_sha256":ordered_hash(plan["panels"][name]),
                          "shape":[len(plan["panels"][name]),2,256,256],"prediction_dtype":"float32","forward_precision":"BF16 autocast",
                          "file_size_bytes":target.stat().st_size,"prediction_sha256":sha(target)}
                    atomic(target.with_suffix(".json"),info);exported["panels"][name]={"file":target.name,**info}
                atomic(run_dir/"selected_export.json",exported)
                # Hard links preserve prior completed best artifacts across an interrupted epoch commit.
                archive=run_dir/f"selected_epoch_{epoch:04d}"
                link_atomic(run_dir/"selected.pt",Path(str(archive)+".pt"))
                for name in panels:
                    link_atomic(run_dir/f"selected_{name}_prediction.npy",Path(str(archive)+f"_{name}_prediction.npy"))
                atomic(Path(str(archive)+"_export.json"),exported)
            else:
                for path in pred_paths.values():path.unlink(missing_ok=True)
            cc_tracking.log(metric_log(row),step=epoch)
            print(json.dumps({"event":"evaluation","arm":arm,"epoch":epoch,"fit":evaluation["fit"]["rmse"],
                             "monitor":evaluation["monitor"]["rmse"],"best_epoch":best_epoch}),flush=True)
            return row
        last_epoch_start=last_epoch;eta_clock_start=time.monotonic()
        if not resume:
            score_epoch(0,zero_head=True);save(run_dir/"last.pt",state())
        eta_clock_start=time.monotonic()
        for epoch in range(last_epoch+1,13):
            if epoch==9 and phase==1:
                previous_steps={int(v["step"].item()) for v in opt.state.values()}
                if previous_steps!={2816}:raise ValueError("Epoch8 Adam step must be2816 before declared reset")
                opt=optimizer_for(net,plan);scheduler=second_scheduler(opt,plan)
                previous_fit=next(r["evaluation"]["fit"]["weighted_MSE_objective"] for r in history if r["epoch"]==8)
                scheduler.step(previous_fit);phase=2
                atomic(run_dir/"optimizer_reset.json",{"epoch":9,"after_total_updates":updates,"previous_optimizer_steps":2816,
                                                      "new_optimizer_steps":0,"FIT_scheduler_initial_objective":previous_fit})
            order=torch.randperm(len(fit.ids),generator=generator)
            sums=torch.zeros(4,device="cuda",dtype=torch.float64);running_loss=0.
            net.train();progress=tqdm(enumerate(order.split(plan["batch"]),1),total=352,desc=f"C151a{arm} Epoch{epoch}/12",mininterval=5)
            for batch_index,positions in progress:
                ix=positions.cuda();x,fs,ld,y,w=fit.batch(ix)
                rate=schedule_lr(epoch,batch_index,plan)
                if rate is not None:
                    for group in opt.param_groups:group["lr"]=rate
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    map_ids=[fit.ids[int(j)] for j in positions.tolist()]
                    prediction=prior(fs,ld,plan["base_prior_arm"])+net(x,map_ids).float()
                error=torch.where(w,prediction-y,0.);ww=weights[ix,None,None,None]
                numerator=(error.square()*ww).sum((0,2,3));count=(w*ww).sum((0,2,3))
                loss=(numerator/(count if phase==1 else den)).mean()/25
                if updates<3 and torch.count_nonzero(x[:,56:58])!=0:raise ValueError("C82 slots are not zero")
                if updates==0 and not torch.equal(prediction,prior(fs,ld,plan["base_prior_arm"])):
                    raise ValueError("Fresh zero head does not equal the analytical prior")
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite C151a data loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
                if not torch.isfinite(norm):raise RuntimeError("Nonfinite C151a gradient")
                opt.step();updates+=1;running_loss+=float(loss.detach())
                sums+=torch.cat((numerator.detach().double(),count.double()))
                if batch_index%40==0 or batch_index==352:
                    done=epoch-last_epoch_start-1+batch_index/352
                    eta=(time.monotonic()-eta_clock_start)/done*(12-last_epoch_start-done)
                    atomic(run_dir/"progress.json",{"stage":"training","arm":arm,"epoch":epoch,"epochs":12,"phase":phase,
                         "batch":batch_index,"batches":352,"updates":updates,"learning_rate":opt.param_groups[0]["lr"],
                         "elapsed_seconds":elapsed(),"eta_seconds":max(0.,eta)})
                    progress.set_postfix(loss=f"{float(loss):.4f}")
            last_epoch=epoch
            if epoch in (4,8,12):score_epoch(epoch)
            save(run_dir/"last.pt",state())
        selected=torch.load(run_dir/"selected.pt",map_location="cpu",weights_only=False)
        if updates!=4224:raise ValueError("Final fixed budget not met")
        result={"status":"complete","arm":arm,"epochs":12,"updates":updates,"selected_epoch":selected["epoch"],
                "selected_predictor_sha256":selected["predictor_sha256"],"last_predictor_sha256":predictor_digest(digest(net),arm,plan),
                "selected_monitor_RMSE_dB":next(r["evaluation"]["monitor"]["rmse"] for r in history if r["epoch"]==selected["epoch"]),
                "last_monitor_RMSE_dB":history[-1]["evaluation"]["monitor"]["rmse"],
                "elapsed_seconds":elapsed(),"data_load_seconds":load_seconds,"protocol_sha256":identity,
                "independent_prediction_verification":"pending","automatic_budget_extension":False,"official_VAL_used":False}
        atomic(run_dir/"result.json",result);atomic(run_dir/"progress.json",{"stage":"complete",**result})
        print(json.dumps({"event":"complete",**result}),flush=True)
    finally:
        if tracking:cc_tracking.finish()
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-protocol",action="store_true")
    parser.add_argument("--bind-c150",action="store_true")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--arm",choices=("A","B"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.prepare_protocol:
        plan=make_protocol(bind_c150=args.bind_c150);print(json.dumps({"protocol":str((AUDIT/"protocol.json").relative_to(ROOT)),
              "cache_ready":plan["cache"]["ready"],"prior_ready":plan["ready"],"base_prior_arm":plan["base_prior_arm"],"parameters":plan["parameters"],"initial_state_sha256":plan["initial_state_sha256"]}));return
    if args.preflight:
        plan=read_protocol(require_cache=False);print(json.dumps(preflight(plan)));return
    if args.arm is None:parser.error("--arm A|B is required for training")
    run(args.arm,args.resume)


if __name__=="__main__":main()
