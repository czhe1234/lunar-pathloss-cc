"""C151b single-arm runner: identical active Cartesian/TX-polar branch, fixed12epochs.

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
from src.c151b_field import create, digest, BRANCH_CONFIG, PARAMETERS_BY_TRUNK, TRUNKS
from src.c151_polar import PAYLOAD_CHANNEL_NAMES
FROZEN_BRANCH_SHA256 = "13aba7b2770fe53a11cd4a07c004ef0430064ed4cb183afa8f8040943bf9cceb"
from src.c150_train import Reader, Panel, EvaluationReferences, cache_identity, schedule_lr
from src import cc_tracking
import numpy as np
import torch
from tqdm import tqdm

AUDIT = ROOT / "audit/c151_polar"
RUN_ROOT = ROOT / "runs/c151_polar"
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
    value={"net_state_sha256":net_hash,"arm":arm,"branch_mode":plan["arm_modes"][arm],
           "trunk_kind":plan["trunk_kind"],"branch_config":plan["branch_config"],
           "base_prior_arm":plan["base_prior_arm"],"prior":plan["prior_description"],
           "feature_cache_sha256":plan["cache"]["features_sha256"]}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def checked_summary(summary_path, protocol_path, campaign):
    value=json.loads(summary_path.read_text())
    if (value.get("status")!="verified" or value.get("campaign")!=campaign
            or value["protocol_sha256"]!=sha(protocol_path)):
        raise ValueError(campaign+" requires its independently verified final paired report")
    return value


def historical_evidence(summary, arm, directory):
    item=summary["arms"][arm];evidence=item["evidence"]
    expected={"verified_json_sha256":evidence["verified_json_sha256"],
              "statistics_sha256":evidence.get("statistics_sha256",evidence.get("statistic_sha256"))}
    for key,file in (("verified_json_sha256","selected_verified.json"),("statistics_sha256","selected_terrain_sse.npz")):
        if sha(directory/file)!=expected[key]:raise ValueError("Historical independently verified evidence changed")
    result=json.loads((directory/"result.json").read_text())
    if (result.get("status")!="complete" or result["updates"]!=4224
            or result["selected_predictor_sha256"]!=item["predictor_sha256"]
            or result["selected_epoch"]!=item["selected_epoch"]):
        raise ValueError("Historical selected predictor/result differs")
    rmse=item["selected"]["monitor"]["all"]["RMSE_dB"]
    return {"run_directory":str(directory.relative_to(ROOT)),"selected_epoch":item["selected_epoch"],
            "predictor_sha256":item["predictor_sha256"],"monitor_RMSE_dB":rmse,
            "joint_mean_monitor_RMSE_dB":float(np.mean(rmse)),**expected}


def make_protocol(bind_results=False):
    if RUN_ROOT.exists():
        raise FileExistsError("Do not rewrite a protocol after any C151b arm directory exists")
    c150_path=ROOT/"audit/c150_diffraction_training/protocol.json"
    c151a_path=ROOT/"audit/c151_global/protocol.json"
    base=json.loads(c150_path.read_text())
    if sha(ROOT/"src/c151_polar.py")!=FROZEN_BRANCH_SHA256:
        raise ValueError("The independently reviewed coordinate branch source changed")
    prior_summary_path=ROOT/"audit/c150_diffraction_training/comparison_summary.json"
    prior_summary=checked_summary(prior_summary_path,c150_path,"C150")
    prior_arm=prior_summary["lowest_development_joint_mean_arm"]
    if prior_arm not in ("A","B"):raise ValueError("Invalid C150 common analytical prior")
    prior_dir=ROOT/"runs/c150_diffraction_prior"/prior_arm
    prior_evidence=historical_evidence(prior_summary,prior_arm,prior_dir)
    plan=deepcopy(base)
    plan.update(campaign="C151b",created_utc=datetime.now(timezone.utc).isoformat(),
        question="Does TX-centered polar routing improve an identical active Cartesian sequence branch?",
        ready=False,trunk_kind=None,parameters=None,initial_state_sha256=None,
        base_prior_arm=prior_arm,prior_description=base["arms"][prior_arm],
        historical_control=None,
        C150_protocol=str(c150_path.relative_to(ROOT)),C150_protocol_sha256=sha(c150_path),
        C151a_protocol=str(c151a_path.relative_to(ROOT)),C151a_protocol_sha256=sha(c151a_path),
        arm_modes={"A":"cartesian","B":"polar"},
        arms={"A":"active Cartesian sequence branch","B":"active TX-centered polar sequence branch"},
        channel_contract="raw11 basic +45 solver +2 fixed zero +10 diffraction; no C82 context or trained initialization",
        branch_config=BRANCH_CONFIG,branch_parameters=20752,
        parameters_by_trunk=PARAMETERS_BY_TRUNK,
        initial_state_sha256_by_trunk={},
        global_inputs_by_trunk={"blind":"unchanged derived13", "aligned":"derived13 + spatially aligned solver45 + diffraction10"},
        fusion="Global96 + Local64 + active coordinate branch16; common 176-to-128 pointwise mix",
        controlled_difference="Only branch coordinate routing differs: Cartesian rectangle versus TX-centered fixed-Rmax polar sampling/inverse routing. Shared trunk, all active branch weights, parameter shapes, sequence grid256x192, support rule, prior, initialization, data order and fixed update budget are identical.",
        branch_payload=list(PAYLOAD_CHANNEL_NAMES),
        branch_geometry={"theta":256,"radial_points":192,"radius":"uniform0..per-map maximum TX-to-query-corner distance",
            "TX":"registered published TX converted to query-image rc; no edge clamping",
            "support":"coordinates only; inverse support threshold0.25; max normalization gain4",
            "periodic_theta":True,"cartesian_bidirectional":True,"polar_bidirectional":True,
            "sampling_limitation":"Polar angular spacing grows with radius; at corner TX far arc spacing about8.8m and radial spacing about1.9m. Full-resolution trunk remains active."},
        prior_binding={"rule":"C150 independently verified minimum joint mean monitor RMSE of a single selected checkpoint",
            "summary_source":str(prior_summary_path.relative_to(ROOT)),"summary_sha256_at_binding":sha(prior_summary_path),
            "weights_loaded_for_initialization":False,**prior_evidence},
        trunk_binding={"status":"PENDING independently verified C151a report",
            "rule":"Minimum joint-mean monitor RMSE between C150 prior winner and C151a aligned B; ties retain C150. One common fresh architecture for both arms; no historical weights loaded.",
            "candidates":{"blind":prior_evidence,"aligned":{"run_directory":"runs/c151_global/B","status":"PENDING"}},
            "weights_loaded_for_initialization":False},
        attribution="The matched A/B estimates whole coordinate routing, jointly including sequence orientation, sampling density, interpolation and support coverage. It cannot isolate radial ordering alone. Historical models differ in architecture/capacity and serve as performance references.",
        repetition="One fresh12epoch Cartesian/polar pair; no C109 ordered-ray feature replay, extra seed or automatic extension")
    for trunk in TRUNKS:
        model=create(trunk,"cartesian",plan["initial_seed"],plan["branch_config"])
        if sum(p.numel() for p in model.parameters())!=PARAMETERS_BY_TRUNK[trunk]:
            raise ValueError("Unexpected C151b active parameter budget")
        plan["initial_state_sha256_by_trunk"][trunk]=digest(model)
    sources=["src/c151b_field.py","src/c151b_train.py","src/c151_polar.py","src/c151_field.py",
             *base["source_sha256"],"src/c150_report.py"]
    plan["source_sha256"]={name:sha(ROOT/name) for name in dict.fromkeys(sources)}
    plan["postprocess_source_sha256_at_prepare"]=sha(ROOT/"src/c151_report.py")
    if bind_results:
        a_summary_path=ROOT/"audit/c151_global/comparison_summary.json"
        a_summary=checked_summary(a_summary_path,c151a_path,"C151a")
        a_plan=json.loads(c151a_path.read_text())
        if (a_plan["base_prior_arm"]!=prior_arm or a_plan["panels"]!=plan["panels"]
                or a_plan["arm_modes"]["B"]!="aligned_global"):
            raise ValueError("C151a aligned candidate does not share the common prior/data")
        aligned=historical_evidence(a_summary,"B",ROOT/"runs/c151_global/B")
        candidates={"blind":prior_evidence,"aligned":aligned}
        trunk=min(TRUNKS,key=lambda name:candidates[name]["joint_mean_monitor_RMSE_dB"])
        plan.update(ready=True,trunk_kind=trunk,parameters=PARAMETERS_BY_TRUNK[trunk],
            initial_state_sha256=plan["initial_state_sha256_by_trunk"][trunk],
            historical_control=candidates[trunk]["run_directory"],
            historical_controls=[prior_evidence["run_directory"],aligned["run_directory"]],
            trunk_binding={"status":"verified_bound","selected_trunk":trunk,
                "rule":plan["trunk_binding"]["rule"],"candidates":candidates,
                "C151a_summary_source":str(a_summary_path.relative_to(ROOT)),
                "C151a_summary_sha256_at_binding":sha(a_summary_path),"weights_loaded_for_initialization":False})
    atomic(AUDIT/"protocol.json",plan)
    return plan


def read_protocol(require_cache=True):
    plan=json.loads((AUDIT/"protocol.json").read_text())
    for field,identity in (("directive","directive_sha256"),("base_protocol","base_protocol_sha256"),
                           ("C150_protocol","C150_protocol_sha256"),("C151a_protocol","C151a_protocol_sha256")):
        if sha(ROOT/plan[field])!=plan[identity]:raise ValueError("Frozen C151b anchor changed: "+field)
    for name,expected in plan["source_sha256"].items():
        if sha(ROOT/name)!=expected:raise ValueError("Frozen C151b dependency changed: "+name)
    if (plan["epochs"],plan["updates_per_arm"],plan["branch_config"],plan["parameters_by_trunk"])!=(12,4224,BRANCH_CONFIG,PARAMETERS_BY_TRUNK):
        raise ValueError("Fixed C151b budget differs")
    if require_cache:
        trunk=plan.get("trunk_kind")
        if (plan.get("ready") is not True or trunk not in TRUNKS or plan.get("base_prior_arm") not in ("A","B")
                or plan["parameters"]!=PARAMETERS_BY_TRUNK[trunk]
                or plan["initial_state_sha256"]!=plan["initial_state_sha256_by_trunk"][trunk]):
            raise ValueError("Bind independently verified C150 prior/C151a common trunk before C151b launch")
        if cache_identity(plan["panels"],require=True)!=plan["cache"]:raise ValueError("C150 raw feature identity changed")
        binding=plan["trunk_binding"]
        if (sha(ROOT/binding["C151a_summary_source"])!=binding["C151a_summary_sha256_at_binding"]
                or sha(ROOT/plan["prior_binding"]["summary_source"])!=plan["prior_binding"]["summary_sha256_at_binding"]):
            raise ValueError("Bound independently verified report changed")
        for item in binding["candidates"].values():
            directory=ROOT/item["run_directory"]
            for key,file in (("verified_json_sha256","selected_verified.json"),("statistics_sha256","selected_terrain_sse.npz")):
                if sha(directory/file)!=item[key]:raise ValueError("Historical frozen evidence changed")
    return plan


def model_for(plan,arm):
    return create(plan["trunk_kind"],plan["arm_modes"][arm],plan["initial_seed"],plan["branch_config"])


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
    for start in tqdm(range(0,n,batch),desc=f"C151b{arm} {name}",mininterval=5):
        stop=min(start+batch,n);sl=slice(start,stop);x,fs,ld,y,w=panel.batch(sl)
        if zero_head:p=prior(fs,ld,base_prior_arm)
        else:
            with torch.autocast("cuda",dtype=torch.bfloat16):p=prior(fs,ld,base_prior_arm)+net(x,panel.ids[start:stop]).float()
        if not torch.isfinite(p).all():raise RuntimeError("Nonfinite C151b prediction")
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
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise AssertionError("CPU preflight may not initialize CUDA")
    yy,xx=torch.meshgrid(torch.arange(256).float(),torch.arange(256).float(),indexing="ij")
    x=torch.empty(1,68,256,256)
    for channel in range(68):x[:,channel]=.1*torch.sin(xx/(5+channel%7))+.08*torch.cos(yy/(7+channel%5))+.001*channel
    x[:,0]=.5+.001*xx+.0005*yy;x[:,3]=.02;x[:,4]=-.03
    x[:,5]=(xx-131)/128;x[:,6]=(yy-97)/128;x[:,56:58]=0
    checks={}
    for trunk in TRUNKS:
        initial_hash=None;outputs=[];mode_checks={}
        for mode in ("cartesian","polar"):
            net=create(trunk,mode,plan["initial_seed"],plan["branch_config"])
            count=sum(p.numel() for p in net.parameters());net_hash=digest(net)
            if count!=PARAMETERS_BY_TRUNK[trunk]:raise ValueError("C151b parameter mismatch")
            if initial_hash is None:initial_hash=net_hash
            if initial_hash!=net_hash or net_hash!=plan["initial_state_sha256_by_trunk"][trunk]:
                raise ValueError("Cartesian/polar initial weights differ")
            with torch.inference_mode():
                zero=net(x,[314])
                assert zero.shape==(1,2,256,256) and torch.count_nonzero(zero)==0
            # Disposable in-memory probe exposes body/branch gradients beyond the shared zero output head.
            with torch.no_grad():
                torch.manual_seed(151029);net.trunk.head.weight.normal_(std=.01)
            net.zero_grad(set_to_none=True)
            output=net(x,[314]);assert torch.isfinite(output).all()
            output.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            branch_gradient=float(sum(p.grad.abs().sum() for p in net.branch.parameters()))
            branch_mix=net.trunk.mix.weight.grad[:,160:]
            assert branch_gradient>0 and torch.isfinite(branch_mix).all() and branch_mix.abs().sum()>0
            for module in (net.branch.stem,net.branch.blocks,net.branch.head):
                assert sum(float(p.grad.abs().sum()) for p in module.parameters())>0
            outputs.append(output.detach().clone())
            with torch.inference_mode(),torch.autocast("cpu",dtype=torch.bfloat16):
                mixed=net(x,[314])
                assert mixed.shape==(1,2,256,256) and torch.isfinite(mixed).all()
            mode_checks[mode]={"parameters":count,"initial_state_sha256":net_hash,
                "zero_head_exact_analytic_residual":True,"all_parameter_gradients_finite":True,
                "CPU_BF16_autocast_forward_finite":True,
                "active_branch_gradient_L1":branch_gradient,"active_branch_fusion_gradient_L1":float(branch_mix.abs().sum())}
            del net,zero,output
        if torch.equal(outputs[0],outputs[1]):raise ValueError("Coordinate modes did not change routed predictions")
        checks[trunk]={"A_B_weights_identical":True,"modes":mode_checks,
                       "coordinate_output_difference_RMS":float((outputs[1]-outputs[0]).square().mean().sqrt())}
    assert not torch.cuda.is_initialized()
    return {"status":"CPU_preflight_passed","device":"cpu","CUDA_initialized":False,
        "labels_read":False,"masks_read":False,"historical_weights_loaded":False,
        "fixed_branch_parameters":20752,"sequence_grid":[256,192],
        "trunk_ready":plan["ready"],"trunk_kind":plan["trunk_kind"],"base_prior_arm":plan["base_prior_arm"],"checks":checks}



def run(arm,resume=False):
    plan=read_protocol();run_dir=RUN_ROOT/arm;identity=sha(AUDIT/"protocol.json")
    if GPU_UUID!=UUIDS[arm] or os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU_UUID or GPU_UUID not in APPROVED_GPU_UUIDS:
        raise ValueError("Arm launch must use its frozen approved full GPU UUID")
    if run_dir.exists() and not resume:raise FileExistsError("Inspect C151b arm; use explicit --resume, never repeat")
    if resume and (run_dir/"result.json").exists() and json.loads((run_dir/"result.json").read_text()).get("status")=="complete":
        raise ValueError("This C151b arm already completed")
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
        net=model_for(plan,arm).cuda().to(memory_format=torch.channels_last)
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
        config={"campaign":"C151b","arm":arm,"arm_label":plan["arm_modes"][arm],"protocol_sha256":identity,"protocol":plan,"panels":plan["panels"],
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
        swan=cc_tracking.init(name=f"C151b{arm} "+plan["arm_modes"][arm]+(" resume" if resume else ""),
            config=config,campaign="C151b",description=plan["arm_modes"][arm]+"; common "+plan["prior_description"]+"; fresh12epochs; local only.",
            tags=["teacher-free","TX-coordinate-routing",f"arm-{arm}"],**kwargs)
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
                "arm":arm,"branch_mode":plan["arm_modes"][arm],"trunk_kind":plan["trunk_kind"],"branch_config":plan["branch_config"],
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
                    "branch_mode":plan["arm_modes"][arm],"trunk_kind":plan["trunk_kind"],"branch_config":plan["branch_config"],"panels":{}}
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
            net.train();progress=tqdm(enumerate(order.split(plan["batch"]),1),total=352,desc=f"C151b{arm} Epoch{epoch}/12",mininterval=5)
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
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite C151b data loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
                if not torch.isfinite(norm):raise RuntimeError("Nonfinite C151b gradient")
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
    parser.add_argument("--bind-results",action="store_true")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--arm",choices=("A","B"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.bind_results and not args.prepare_protocol:parser.error("--bind-results requires --prepare-protocol")
    if args.prepare_protocol:
        plan=make_protocol(bind_results=args.bind_results);print(json.dumps({"protocol":str((AUDIT/"protocol.json").relative_to(ROOT)),
              "cache_ready":plan["cache"]["ready"],"trunk_ready":plan["ready"],"trunk_kind":plan["trunk_kind"],"base_prior_arm":plan["base_prior_arm"],"parameters":plan["parameters"],"initial_state_sha256":plan["initial_state_sha256"],"parameters_by_trunk":plan["parameters_by_trunk"]}));return
    if args.preflight:
        plan=read_protocol(require_cache=False);print(json.dumps(preflight(plan)));return
    if args.arm is None:parser.error("--arm A|B is required for training")
    run(args.arm,args.resume)


if __name__=="__main__":main()
