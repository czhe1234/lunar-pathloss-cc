"""C150 paired-design single-arm training; fresh initialization, no C82 input.

Run A and B as separate processes with identical frozen protocol/cache/order/state.
Their ONLY different computation is the fixed analytic output prior.
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
from src.c150_field import create, digest, inputs, normalize_diffraction, CHANNELS
from src.c146_data import Reader as SolverReader, Panel as SolverPanel
from src import cc_tracking
import numpy as np
import torch
from tqdm import tqdm

AUDIT = ROOT / "audit/c150_diffraction_training"
RUN_ROOT = ROOT / "runs/c150_diffraction_prior"
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
    value = {"net_state_sha256": net_hash, "arm": arm, "prior": plan["arms"][arm],
             "feature_cache_sha256": plan["cache"]["features_sha256"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cache_identity(panels, require=False):
    expected = panels["fit"] + panels["monitor"]
    identity = {"root": str(CACHE.relative_to(ROOT)), "ready": False,
                "shape": [len(expected), 12, 256, 256], "dtype": "float32",
                "ordered_ids_sha256": ordered_hash(expected), "channel_names": CHANNEL_NAMES}
    files = [CACHE / name for name in ("features.npy", "indices.json", "complete.json")]
    if not all(path.exists() for path in files):
        if require:
            raise FileNotFoundError("C150 input-only feature export is not complete")
        return identity
    meta = json.loads(files[2].read_text())
    if meta.get("status") != "complete":
        raise ValueError("C150 feature marker is not complete")
    for key in ("labels_read", "masks_read", "C82_read"):
        if meta.get(key) is not False:
            raise ValueError(f"Input-only feature provenance missing/violated: {key}")
    if meta.get("channel_names") != CHANNEL_NAMES:
        raise ValueError("Raw feature channel names differ")
    if not isinstance(meta.get("source_sha256"),dict) or not meta["source_sha256"]:
        raise ValueError("Generator source identities are missing")
    for name,expected_hash in meta["source_sha256"].items():
        source=ROOT/name
        if not source.is_file() or sha(source)!=expected_hash:
            raise ValueError("Feature generator source identity changed: "+name)
    ids = json.loads(files[1].read_text())
    if ids != expected:
        raise ValueError("C150 cache must preserve exact FIT then monitor ordered IDs")
    array = np.load(files[0], mmap_mode="r", allow_pickle=False)
    if list(array.shape) != identity["shape"] or array.dtype != np.float32:
        raise ValueError("C150 raw feature layout differs")
    del array
    hashed = sha(files[0])
    if meta.get("features_sha256") != hashed:
        raise ValueError("C150 feature file digest differs from exporter")
    return {**identity, "ready": True, "features_sha256": hashed,
            "indices_file_sha256": sha(files[1]), "complete_sha256": sha(files[2]),
            "geometry_convention": meta.get("geometry_convention"),
            "generator_protocol_sha256": meta.get("protocol_sha256"),
            "generator_source_sha256": meta.get("source_sha256")}


def make_protocol():
    if RUN_ROOT.exists():
        raise FileExistsError("Cannot rewrite a protocol after a C150 arm directory exists")
    old_path = ROOT / "audit/c145_capacity/protocol.json"
    original = json.loads(old_path.read_text())
    first_path = ROOT / "audit/c123_lossless/protocol.json"
    first = json.loads(first_path.read_text())
    if first["epochs"] != 8 or original["parent_epoch"] != 8 or original["warmup_updates"] != 100:
        raise ValueError("Historical schedule anchor differs; inspect before changing")
    model = create(123041)
    source_files = ["src/c150_field.py", "src/c150_train.py",
                    "src/c146_data.py", "src/c123_lossless.py", "src/c99_axial.py",
                    "src/c117_factorized.py", "src/c3_geometry.py", "src/cc_core.py",
                    "src/cc_tracking.py"]
    optional = ["src/c150_diffraction.py", "src/c150_diffraction.cu", "src/c150_reference.py"]
    source_files += [name for name in optional if (ROOT/name).exists()]
    cache = cache_identity(original["panels"])
    directive = ROOT / "audit/c149_c151/directive.md"
    plan = {"campaign": "C150", "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "At identical fresh Field/input/order/updates, does analytical Deygout output prior help?",
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                        "SwanLab": importlib.metadata.version("swanlab"), "forward": "BF16 autocast / Float32 outputs"},
        "directive": str(directive.relative_to(ROOT)), "directive_sha256": sha(directive),
        "base_protocol": str(old_path.relative_to(ROOT)), "base_protocol_sha256": sha(old_path),
        "c123_protocol_sha256": sha(first_path), "panels": original["panels"],
        "scope": {"fit_maps": 2815, "fit_terrains": 1984, "monitor_maps": 628, "monitor_terrains": 128,
                  "monitor_is_fresh_blind": False, "C82_beta_fit_terrains": 80, "C82_selection_terrains": 48},
        "arms": {"A": "registered_FSPL + Net(x)", "B": "registered_FSPL + raw_L_Deygout + Net(x)"},
        "channels": 68, "channel_contract": "11 basic +45 solver +2 literal zero +10 deterministic diffraction",
        "diffraction_normalization": "Each band: asinh(nu)/3,2*t-1,log1p(max(I,0))/3,Ld/20,Lb/20",
        "raw_priors": "Registered 3D FSPL and raw Float32 Ld; never reconstruct priors from quantized normalized inputs",
        "C82_role": "Evaluation-only references/regions; never input, target prior, weight initialization or feature generator",
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_seed": 123041, "initial_state_sha256": digest(model), "data_seed": 123073,
        "epochs": 12, "batch": 8, "steps_per_epoch": 352, "updates_per_arm": 4224,
        "learning_rate": .0003, "weight_decay": .01, "clip_grad_norm": 2.,
        "evaluate_epochs": [0,4,8,12], "optimizer": "AdamW fused; reset at epoch9 in BOTH arms",
        "phase1": {"epochs": 8, "cosine_horizon_epochs": 8, "warmup_updates": 100,
                   "warmup_start_fraction": .1, "cosine_min_fraction": .15,
                   "loss": "C123 per-batch map-weighted per-band SSE/count, average bands/25"},
        "phase2": {"epochs": 4, "optimizer_reset_epoch": 9, "warmup_updates": 100,
                   "loss": "C145 exact fullFIT weighted per-band count divided by352 denominator; mean bands/25",
                   "scheduler": "ReduceLROnPlateau FIT-only; initialized from epoch8 weighted FIT objective; step at epoch12",
                   "factor": original["lr_factor"], "patience": original["lr_patience_evaluations"],
                   "threshold": original["minimum_relative_fit_improvement"],
                   "minimum_lr": original["minimum_learning_rate"]},
        "schedule_scope": "Same12 fullFIT exposure and historical8+reset+4 schedule/loss path; no C145 weights or C82 context",
        "selection": "One checkpoint per arm: minimum joint mean monitor RMSE among0,4,8,12; report final12 too",
        "gate": {"difference": "candidate_RMSE-reference_RMSE; negative favors candidate",
                 "references": ["C82", "C145_selected12"], "CI": "paired terrain bootstrap95%;2000rep;seed145073",
                 "promotion": "Both band upper95CI<0 versus BOTH references; one same checkpoint; only then scale"},
        "cache": cache, "GPU_UUID_by_arm": UUIDS, "allow_shared_GPU": True,
        "repetition": "One seed/one fixed12epoch pair; no hyperparameter or seed sweeps, no automatic extension",
        "source_sha256": {name: sha(ROOT/name) for name in source_files},
        "postprocess_source_sha256_at_prepare": sha(ROOT/"src/c150_report.py") if (ROOT/"src/c150_report.py").exists() else None,
        "official_VAL_used": False}
    if plan["parameters"] != 15223106:
        raise ValueError("Unexpected C150 parameter count")
    atomic(AUDIT/"protocol.json", plan)
    return plan


def read_protocol(require_cache=True):
    plan = json.loads((AUDIT/"protocol.json").read_text())
    if sha(ROOT/plan["directive"]) != plan["directive_sha256"]:
        raise ValueError("User directive identity changed")
    if sha(ROOT/plan["base_protocol"]) != plan["base_protocol_sha256"]:
        raise ValueError("Frozen C145 panel/schedule anchor changed")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Frozen C150 implementation changed: {name}")
    if plan["epochs"] != 12 or plan["updates_per_arm"] != 4224:
        raise ValueError("Fixed C150 budget changed")
    if require_cache:
        actual = cache_identity(plan["panels"], require=True)
        if actual != plan["cache"] or not actual["ready"]:
            raise ValueError("Freeze the complete feature cache in protocol before training")
    return plan


class Reader(SolverReader):
    """Training/prediction reader never opens C82 predictions or weights."""
    def __init__(self, supervision=True):
        super().__init__(supervision=supervision)
        self.diffraction = np.load(CACHE/"features.npy", mmap_mode="r", allow_pickle=False)
        self.diff_ids = json.loads((CACHE/"indices.json").read_text())
        self.diff_lookup = {int(i): j for j,i in enumerate(self.diff_ids)}

    def raw_diffraction(self, ids):
        positions = [self.diff_lookup[int(i)] for i in ids]
        if positions == list(range(positions[0], positions[0]+len(positions))):
            return np.array(self.diffraction[positions[0]:positions[0]+len(positions)])
        return np.array(self.diffraction[positions])


class Panel(SolverPanel):
    def __init__(self, reader, ids, device="cuda"):
        super().__init__(reader, ids, device=device)
        n = len(ids)
        self.diff = torch.empty((n,10,256,256), dtype=torch.float16, device=device)
        self.fspl = torch.empty((n,2,256,256), dtype=torch.float32, device=device)
        self.deygout = torch.empty_like(self.fspl)
        for start in range(0,n,32):
            stop = min(start+32,n)
            raw = torch.as_tensor(reader.raw_diffraction(ids[start:stop]), device=device)
            if not torch.isfinite(raw).all():
                raise ValueError("Nonfinite raw diffraction cache")
            self.diff[start:stop] = normalize_diffraction(raw[:,:10]).half()
            self.fspl[start:stop] = raw[:,10:12]
            self.deygout[start:stop] = raw[:,[3,8]]
            if (raw[:,[2,3,4,7,8,9]] < -1e-6).any():
                raise ValueError("Negative integral or analytic loss")
        print(json.dumps({"event":"C150_panel", "maps":n, "channels":68}),flush=True)

    def batch(self, ix):
        basic, physical, _legacy_fspl, y, valid = super().batch(ix)
        x = inputs(basic, physical, self.diff[ix].float())
        return x, self.fspl[ix], self.deygout[ix], y, valid


class EvaluationReferences:
    """Evaluation-only C82 fields form masks; sufficient statistics are reused."""
    def __init__(self, plan, reader):
        parent = ROOT/"runs/c145_capacity_native"
        self.baseline = {}; self.counts = {}; self.sse = {}
        for panel in ("fit","monitor"):
            ids = plan["panels"][panel]
            path = ROOT/("audit/c101_residual/base0of1/prediction.npy" if panel=="fit" else "audit/c88_mesh/base.npy")
            source_ids = json.loads((path.parent/("indices.json" if panel=="fit" else "base_index.json")).read_text())
            if isinstance(source_ids,dict): source_ids=source_ids["ids"]
            lookup = {int(i):j for j,i in enumerate(source_ids)}
            array = np.load(path,mmap_mode="r",allow_pickle=False)
            self.baseline[panel] = torch.empty((len(ids),2,256,256), device="cuda",dtype=torch.float32)
            for start in range(0,len(ids),32):
                self.baseline[panel][start:start+32] = torch.as_tensor(np.stack([array[lookup[int(i)]] for i in ids[start:start+32]]),device="cuda")
            with np.load(parent/"selected_terrain_sse.npz",allow_pickle=False) as stats:
                if stats["group_names"].tolist()!=list(VERIFIED_GROUPS) or stats[panel+"_map_ids"].tolist()!=ids:
                    raise ValueError("Frozen reference group/panel identity changed")
                self.counts[panel]=stats[panel+"_map_group_count"].copy()
                self.sse[panel]=stats[panel+"_map_group_baseline_sse"].copy()


def prior(fspl, deygout, arm):
    return fspl if arm=="A" else fspl+deygout


def groups(valid, target, baseline):
    residual=target.double()-baseline.double();tail=residual.abs()>=5
    return (valid,valid&tail,valid&(residual>=5),valid&(residual<=-5),
            valid&~tail,valid&(residual.abs()<=1))


@torch.inference_mode()
def evaluate(net, panel, name, references, arm, batch, prediction_path, weights=None, zero_head=False):
    net.eval();n=len(panel.ids)
    mm = np.lib.format.open_memmap(prediction_path,mode="w+",dtype=np.float32,shape=(n,2,256,256))
    sse=np.empty((6,n,2),np.float64);counts=np.empty((6,n,2),np.int64)
    for start in tqdm(range(0,n,batch),desc=f"C150{arm} {name}",mininterval=5):
        stop=min(start+batch,n);sl=slice(start,stop);x,fs,ld,y,w=panel.batch(sl)
        if zero_head:p=prior(fs,ld,arm)
        else:
            with torch.autocast("cuda",dtype=torch.bfloat16):p=prior(fs,ld,arm)+net(x).float()
        if not torch.isfinite(p).all():raise RuntimeError("Nonfinite C150 prediction")
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


def schedule_lr(epoch,batch_index,plan):
    steps=plan["steps_per_epoch"];step=(epoch-1)*steps+batch_index
    if epoch<=8:
        warm=min(1.,.1+.9*step/100)
        return plan["learning_rate"]*warm*(.15+.85*.5*(1+math.cos(math.pi*(step-1)/(steps*8))))
    reset_step=(epoch-9)*steps+batch_index
    return plan["learning_rate"]*(.1+.9*reset_step/100) if reset_step<=100 else None


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
    a=create(plan["initial_seed"]);b=create(plan["initial_seed"])
    assert digest(a)==digest(b)==plan["initial_state_sha256"]
    assert sum(p.numel() for p in a.parameters())==plan["parameters"]==15223106
    basic=torch.zeros((2,11,8,8));physical=torch.ones((2,45,8,8))
    raw=torch.zeros((2,10,8,8));raw[:,[1,6]]=.25;raw[:,[2,7]]=1;raw[:,[3,8]]=20
    x=inputs(basic,physical,normalize_diffraction(raw))
    assert x.shape==(2,68,8,8) and torch.count_nonzero(x[:,56:58])==0
    assert torch.all(x[:,59]==-.5) and torch.all(x[:,61]==1)
    assert a.stem.in_channels==208 and a.local[0].in_channels==68 and a.film.in_features==68
    rates=[schedule_lr(e,j,plan) for e,j in [(1,1),(8,352),(9,1),(9,100),(9,101)]]
    assert abs(rates[0]-.0003*.109)<1e-15 and rates[-1] is None
    from src.c123_lossless import create as original_create
    old=original_create("native").state_dict(); shared=[]
    for name,value in a.state_dict().items():
        if name.startswith(("film.","local.0.")):continue
        if not torch.equal(value,old[name]):raise ValueError("Shared C123 initialization changed: "+name)
        shared.append(name)
    assert torch.cuda.is_initialized() is False
    return {"status":"CPU_preflight_passed","parameters":plan["parameters"],
            "initial_state_sha256":digest(a),"two_arms_initial_states_identical":True,"shared_initial_tensors_equal_C123":len(shared),
            "C82_context_slots_exact_zero":True,"global_backbone_input_channels":13,
            "raw_input_channels":68,"cache_ready":plan["cache"]["ready"],
            "schedule_probe_LR":rates,"CUDA_initialized":False}


def run(arm,resume=False):
    plan=read_protocol();run_dir=RUN_ROOT/arm;identity=sha(AUDIT/"protocol.json")
    if GPU_UUID!=UUIDS[arm] or os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU_UUID or GPU_UUID not in APPROVED_GPU_UUIDS:
        raise ValueError("Arm launch must use its frozen approved full GPU UUID")
    if run_dir.exists() and not resume:raise FileExistsError("Inspect C150 arm; use explicit --resume, never repeat")
    if resume and (run_dir/"result.json").exists() and json.loads((run_dir/"result.json").read_text()).get("status")=="complete":
        raise ValueError("This C150 arm already completed")
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
        net=create(plan["initial_seed"]).cuda().to(memory_format=torch.channels_last)
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
        config={"campaign":"C150","arm":arm,"protocol_sha256":identity,"protocol":plan,"panels":plan["panels"],
                "GPU":gpu,"launch_device_snapshot":gpu_snapshot,"parameters":plan["parameters"],"initial_state_sha256":plan["initial_state_sha256"],
                "prior":plan["arms"][arm],"official_VAL_used":False,
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
        swan=cc_tracking.init(name=f"C150{arm} registered diffraction prior"+(" resume" if resume else ""),
            config=config,campaign="C150",description=plan["arms"][arm]+"; fresh12epochs; no C82 inputs/weights; local only.",
            tags=["teacher-free","diffraction",f"arm-{arm}"],**kwargs)
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
                "arm":arm,"epoch":last_epoch,"updates":updates,"phase":phase,"best_epoch":best_epoch,"best_mean":best_mean,
                "history":deepcopy(history),"elapsed_seconds":elapsed(),"protocol_sha256":identity,
                "net_state_sha256":digest(net),"predictor_sha256":predictor_digest(digest(net),arm,plan)}
        def score_epoch(epoch,zero_head=False):
            nonlocal best_epoch,best_mean,history
            pred_paths={name:run_dir/f"epoch_{epoch:04d}_{name}.partial.npy" for name in panels}
            evaluation={};stats={"group_names":np.asarray(VERIFIED_GROUPS)}
            for name,panel in panels.items():
                evaluation[name],st=evaluate(net,panel,name,refs,arm,plan["batch"],pred_paths[name],
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
                    "prior":plan["arms"][arm],"panels":{}}
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
            net.train();progress=tqdm(enumerate(order.split(plan["batch"]),1),total=352,desc=f"C150{arm} Epoch{epoch}/12",mininterval=5)
            for batch_index,positions in progress:
                ix=positions.cuda();x,fs,ld,y,w=fit.batch(ix)
                rate=schedule_lr(epoch,batch_index,plan)
                if rate is not None:
                    for group in opt.param_groups:group["lr"]=rate
                with torch.autocast("cuda",dtype=torch.bfloat16):prediction=prior(fs,ld,arm)+net(x).float()
                error=torch.where(w,prediction-y,0.);ww=weights[ix,None,None,None]
                numerator=(error.square()*ww).sum((0,2,3));count=(w*ww).sum((0,2,3))
                loss=(numerator/(count if phase==1 else den)).mean()/25
                if updates<3 and torch.count_nonzero(x[:,56:58])!=0:raise ValueError("C82 slots are not zero")
                if updates==0 and not torch.equal(prediction,prior(fs,ld,arm)):
                    raise ValueError("Fresh zero head does not equal the analytical prior")
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite C150 data loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
                if not torch.isfinite(norm):raise RuntimeError("Nonfinite C150 gradient")
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
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--arm",choices=("A","B"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.prepare_protocol:
        plan=make_protocol();print(json.dumps({"protocol":str((AUDIT/"protocol.json").relative_to(ROOT)),
              "cache_ready":plan["cache"]["ready"],"parameters":plan["parameters"],"initial_state_sha256":plan["initial_state_sha256"]}));return
    if args.preflight:
        plan=read_protocol(require_cache=False);print(json.dumps(preflight(plan)));return
    if args.arm is None:parser.error("--arm A|B is required for training")
    run(args.arm,args.resume)


if __name__=="__main__":main()
