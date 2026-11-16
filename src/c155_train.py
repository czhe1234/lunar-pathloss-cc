"""C155 architecture-only ConvNeXt U-Net comparison to frozen C153;60epochs."""
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
from src.c155_field import create, digest, PARAMETERS
from src.c155_grazing import support_path, identity as file_identity, box, finish_stats, KEYS as GRAZING_KEYS
from src.c150_field import inputs, normalize_diffraction
from src.c150_train import Reader, Panel, EvaluationReferences, cache_identity
from src.c151_train import atomic, save, link_atomic, sha, ordered_hash, groups, GROUPS, VERIFIED_GROUPS, BANDS, C82
from src import cc_tracking
AUDIT=ROOT/"audit/c155_unet"
RUN=ROOT/"runs/c155_unet"
ARM="unet"
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
    contract={"net_state_sha256":net_hash,"output":"registered_FSPL + ConvNeXt_UNet(raw68)",
        "C82_input":False,"C82_output_prior":False,"feature_cache_sha256":plan["cache"]["features_sha256"]}
    return hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()

def make_protocol():
    if RUN.exists() or (AUDIT/"protocol.json").exists():raise FileExistsError("C155 already prepared; inspect, do not repeat")
    parent=ROOT/"audit/c153_standalone_c151a/protocol.json"
    old=json.loads(parent.read_text());support=json.loads((AUDIT/"grazing_support.json").read_text())
    assert support["status"]=="complete" and not support["training_run"] and not support["network_inference_run"]
    assert sha(parent)==support["control_protocol_sha256"]
    keep=("panels","scope","channels","channel_contract","initial_seed","data_seed","C82_role","prior","cache",
        "epochs","batch","steps_per_epoch","updates","updates_per_arm","learning_rate","minimum_lr","warmup_updates",
        "schedule","optimizer","betas","eps","weight_decay","clip_grad_norm","loss","image_weights","validity",
        "evaluate_epochs","selection","forward_precision","C82","official_VAL_used","comparison_limit")
    plan={k:deepcopy(old[k]) for k in keep}
    net=create(old["initial_seed"]);assert sum(p.numel() for p in net.parameters())==PARAMETERS
    files=["src/c155_field.py","src/c155_train.py","src/c155_grazing.py","src/c155_report.py",
        "src/c153_train.py","src/c151_train.py","src/c150_train.py","src/c150_field.py","src/c146_data.py",
        "src/c3_geometry.py","src/cc_core.py","src/cc_tracking.py","src/c145_verify.py","src/c150_report.py","src/c152_los_diagnostic.py"]
    for n,h in old["source_sha256"].items():assert sha(ROOT/n)==h,n
    protected=support["protected_readonly_sha256_from_C154"]
    identities=support["protected_file_identities_before_after_unchanged"]
    assert all(file_identity(ROOT/n)==v for n,v in identities.items())
    for n,v in support["cache_file_identities"].items():assert file_identity(ROOT/n)==v,n
    assert json.loads((ROOT/"features/c150_diffraction/part0of1/complete.json").read_text())["features_sha256"]==old["cache"]["features_sha256"]
    for name in ("fit","monitor"):assert sha(support_path(name))==support["supports"][name]["sha256"]
    plan.update(campaign="C155",name="C155_convnext_unet",created_utc=datetime.now(timezone.utc).isoformat(),
        question="With exact C153 inputs/training budget, does a3.19M four-level full-resolution ConvNeXt U-Net improve monitor generalization and Grazing highpass error?",
        arms={ARM:"registered_FSPL + ConvNeXt_UNet(raw68)"},parameters=PARAMETERS,
        architecture={"widths":[48,96,192,256],"resolutions":[256,128,64,32],"blocks_per_encoder_level":2,
            "blocks_per_decoder_level":2,"depthwise_kernel":7,"MLP_expansion":4,"LayerNorm_epsilon":1e-6,
            "LayerScale_initial":1e-6,"downsample":"LayerNorm plus2x2 stride2 convolution",
            "upsample":"bilinear align_corners=False plus1x1 conv; concatenate skip;1x1 merge",
            "head":"zero initialized1x1 to2channels, output multiplier4 (same C153)",
            "stem":"68to48 channels,3x3 stride1 padding1","dropout":0,"stochastic_depth":0,
            "attention":False,"input_clamp":[-12,12],"translation_limit":"Stride2 and finite padding are not strictly equivariant to every1pixel shift"},
        initialization="Fresh random model,seed123041,zero output head; no trained weights loaded",
        initial_state_sha256=digest(net),augmentation="none",D4=False,
        control_protocol=str(parent.relative_to(ROOT)),control_protocol_sha256=sha(parent),
        base_protocol=old["base_protocol"],base_protocol_sha256=old["base_protocol_sha256"],
        control_selected={"epoch":12,"monitor":[2.301096794983892,2.7937590733328324],
            "FIT":[2.3135863640094168,2.7907659787835604]},
        control_last={"epoch":60,"monitor":[2.564185965510261,2.965352399137775],
            "FIT":[1.88417975269379,2.0565496667122383]},
        only_intervention="Network architecture; same data,features,FSPL,loss,imageweights,batch,optimizer,schedule,evaluation/selection and60epochs",
        grazing={"support_marker":"audit/c155_unet/grazing_support.json","support_marker_sha256":sha(AUDIT/"grazing_support.json"),
            "definition":support["definition"],"primary":"highpass_error_RMS_dB","secondary":["GT and Pred minus FSPL highpass RMS","localmean error RMS"],
            "evaluation":"Original unaugmented full FIT and monitor every4epochs including0",
            "control":"Existing C153 selected12 saved predictions only; historical per-epoch highpass predictions were not retained"},
        paired_CI={"unit":"terrain","terrains":128,"replicates":2000,"seed":145073,"difference":"C155 minus C153 selected12 (primary); C82 secondary","interval":"percentile95"},
        GPU_UUID=UUID,allow_shared_GPU=True,source_sha256={n:sha(ROOT/n) for n in files},
        protected_readonly_sha256=protected,protected_file_identities=identities,cache_file_identities=support["cache_file_identities"],
        protected_verification="C154 end-of-audit SHA reused with unchanged file identities; identities checked each launch; full protected SHA rechecked after new training",
        repetition="One fresh60epoch candidate, no old experiment rerun or automatic extension",
        extra_measurements="Fixed Grazing diagnostics and retaining final predictions; neither changes optimizer/data/loss or adds model passes")
    # Exact common training contract and source loss/schedule assertions.
    for k in keep:assert plan[k]==old[k],k
    body=(ROOT/"src/c153_train.py").read_text()
    for expression in ("order=torch.randperm(len(fit.ids),generator=order_rng)",
        "err=torch.where(w,prediction-y,0.);ww=weights[ix,None,None,None]",
        "numerator=(err.square()*ww).sum((0,2,3));count=(w*ww).sum((0,2,3))",
        "loss=(numerator/den).mean()/25","den=(den/STEPS).float()"):
        assert expression in body and expression in Path(__file__).read_text(),expression
    plan["unchanged_contract_fields"]=list(keep)
    atomic(AUDIT/"protocol.json",plan);return plan

def read_protocol(check_cache=True):
    plan=json.loads((AUDIT/"protocol.json").read_text())
    assert (plan["campaign"],plan["epochs"],plan["updates"],plan["batch"])==("C155",60,21120,8)
    assert plan["augmentation"]=="none" and not plan["D4"]
    assert sha(ROOT/plan["control_protocol"])==plan["control_protocol_sha256"]
    old=json.loads((ROOT/plan["control_protocol"]).read_text())
    for k in plan["unchanged_contract_fields"]:assert plan[k]==old[k],k
    for n,h in plan["source_sha256"].items():assert sha(ROOT/n)==h,n
    for n,v in plan["protected_file_identities"].items():assert file_identity(ROOT/n)==v,n
    for n,v in plan["cache_file_identities"].items():assert file_identity(ROOT/n)==v,n
    assert sha(ROOT/plan["grazing"]["support_marker"])==plan["grazing"]["support_marker_sha256"]
    return plan

def preflight(plan):
    torch.set_num_threads(2);net=create(plan["initial_seed"])
    assert digest(net)==plan["initial_state_sha256"]
    x=torch.randn(1,68,256,256)*.1;x[:,56:58]=0;x.requires_grad_(True)
    shapes=[];handles=[]
    for module in net.encoder:handles.append(module.register_forward_hook(lambda m,i,o:shapes.append(list(o.shape[-2:]))))
    y=net(x,[314]);assert y.shape==(1,2,256,256) and torch.count_nonzero(y)==0
    assert shapes==[[256,256],[128,128],[64,64],[32,32]]
    for h in handles:h.remove()
    with torch.no_grad():net.head.weight.normal_(0,.001)
    net(x,[314]).square().mean().backward()
    norms=x.grad.abs().sum((0,2,3))
    assert torch.isfinite(norms).all() and (norms>0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert not any(isinstance(m,(torch.nn.MultiheadAttention,torch.nn.Dropout)) for m in net.modules())
    yy=torch.randn(2,2,256,256,dtype=torch.float64);pp=yy+torch.randn_like(yy);ff=torch.randn_like(yy)
    mask=torch.ones(2,2,254,254,dtype=torch.bool)
    sums,counts=grazing_sufficient(yy,pp,ff,mask)
    from src.c152_los_diagnostic import Rough
    ref=Rough()
    for j in range(2):ref.add((yy-ff)[j].numpy(),(pp-ff)[j].numpy(),mask[j].numpy())
    actual=finish_stats(counts.sum(0).numpy(),{k:sums[j].sum(0).numpy() for j,k in enumerate(GRAZING_KEYS)})
    expected=ref.result()
    for k,v in actual.items():assert np.allclose(v,expected[k],rtol=0,atol=1e-12),(k,v,expected[k])
    assert learning_rate(200,plan)==3e-4 and learning_rate(21120,plan)==1e-5
    assert not torch.cuda.is_initialized()
    return {"status":"CPU_preflight_passed","parameters":PARAMETERS,"encoder_resolutions":shapes,
        "C82_slots_zero":True,"zero_head_FSPL_output":True,"all_raw68_gradient_paths":True,
        "all_parameters_have_finite_gradients":True,"Grazing_Torch_vs_independent_NumPy_exact_1e12":True,
        "same_C153_training_contract":True,"CUDA_initialized":False}

def gpu_smoke(plan):
    assert GPU_UUID==UUID and os.environ.get("CUDA_VISIBLE_DEVICES")==UUID
    gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision("high")
    net=create(plan["initial_seed"]).cuda().to(memory_format=torch.channels_last)
    before=digest(net);x=torch.randn(8,68,256,256,device="cuda").to(memory_format=torch.channels_last);x[:,56:58]=0
    times=[]
    for j in range(4):
        torch.cuda.synchronize();start=time.monotonic()
        with torch.autocast("cuda",dtype=torch.bfloat16):p=net(x,list(range(8))).float()
        p.square().mean().backward();net.zero_grad(set_to_none=True);torch.cuda.synchronize()
        if j:times.append(time.monotonic()-start)
    assert digest(net)==before==plan["initial_state_sha256"]
    return {"status":"GPU_forward_backward_smoke_passed","GPU":gpu,"optimizer_updates":0,
        "model_state_unchanged":True,"batch":8,"mean_forward_backward_seconds":float(np.mean(times)),
        "three_timed_iterations_seconds":times,"peak_memory_GiB":torch.cuda.max_memory_allocated()/2**30}

def grazing_sufficient(y,p,fs,mask):
    gy=y.double()-fs.double();pr=p.double()-fs.double()
    hg=gy[...,1:-1,1:-1]-box(gy);hp=pr[...,1:-1,1:-1]-box(pr)
    he=hp-hg;lo=box(p.double()-y.double())
    vals=(hg.square(),hp.square(),he.square(),lo.square())
    sums=torch.stack([torch.where(mask,v,0).sum((2,3)) for v in vals])
    return sums,mask.sum((2,3))

@torch.inference_mode()
def evaluate(net,panel,name,references,batch,prediction_path,weights=None,zero_head=False):
    net.eval();n=len(panel.ids)
    mm=np.lib.format.open_memmap(prediction_path,mode="w+",dtype=np.float32,shape=(n,2,256,256))
    sse=np.empty((6,n,2),np.float64);counts=np.empty((6,n,2),np.int64)
    grazing_sums=np.empty((4,n,2),np.float64);grazing_counts=np.empty((n,2),np.int64)
    for start in tqdm(range(0,n,batch),desc=f"C155 {name}",mininterval=5):
        stop=min(start+batch,n);sl=slice(start,stop);x,fs,_ld,y,w=panel.batch(sl)
        assert torch.count_nonzero(x[:,56:58])==0
        if zero_head:p=fs
        else:
            with torch.autocast("cuda",dtype=torch.bfloat16):p=fs+net(x,panel.ids[start:stop]).float()
        if not torch.isfinite(p).all():raise RuntimeError("Nonfinite C155 prediction")
        mm[sl]=p.cpu().numpy();error=p.double()-y.double()
        for gi,m in enumerate(groups(w,y,references.baseline[name][sl])):
            sse[gi,start:stop]=torch.where(m,error.square(),0).sum((2,3)).cpu().numpy()
            counts[gi,start:stop]=m.sum((2,3)).cpu().numpy()

        gs,gc=grazing_sufficient(y,p,fs,panel.grazing9[sl])
        grazing_sums[:,start:stop]=gs.cpu().numpy();grazing_counts[start:stop]=gc.cpu().numpy()

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

    result["grazing"]=finish_stats(grazing_counts.sum(0),{k:grazing_sums[j].sum(0) for j,k in enumerate(GRAZING_KEYS)})
    for j,k in enumerate(GRAZING_KEYS):stats[name+"_grazing_"+k]=grazing_sums[j]
    stats[name+"_grazing_count"]=grazing_counts

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

    for name,v in row["evaluation"].items():
        for j,b in enumerate(BANDS):
            g=v["grazing"]
            for metric,key in (("Grazing_HP_error_RMS","highpass_error_RMS_dB"),
                ("Grazing_GT_HP_RMS","GT_minus_FSPL_highpass_RMS_dB"),
                ("Grazing_Pred_HP_RMS","Pred_minus_FSPL_highpass_RMS_dB"),
                ("Grazing_localmean_error_RMS","localmean_error_RMS_dB")):
                out[f"{name}/{metric}_{b}_dB"]=g[key][j]
            out[f"{name}/Grazing_valid9_count_{b}"]=g["count"][j]

    return out

def run(resume=False):
    plan=read_protocol();identity=sha(AUDIT/"protocol.json")
    if GPU_UUID!=UUID or os.environ.get("CUDA_VISIBLE_DEVICES")!=UUID or UUID not in APPROVED_GPU_UUIDS:
        raise ValueError("C155 requires the frozen approved full GPU UUID in both environment variables")
    if RUN.exists() and not resume:raise FileExistsError("C155 exists; inspect, do not repeat")
    if resume and (RUN/"result.json").exists():raise ValueError("Completed C155 cannot be resumed")
    RUN.mkdir(parents=True,exist_ok=True);lock=RUN/"active.json";tracking=False
    if lock.exists():
        old=json.loads(lock.read_text())
        try:os.kill(old["pid"],0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError("Active C155 process already exists")
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.write(fd,json.dumps({"pid":os.getpid()}).encode());os.close(fd)
    try:
        snapshot={"device":subprocess.check_output(["nvidia-smi","-i",UUID,"--query-gpu=uuid,name,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip(),
            "co_resident_processes":[v for v in subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True).splitlines() if UUID in v],
            "other_processes_terminated":False}
        check_source_unchanged();gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision("high")
        load_start=time.monotonic();reader=Reader()
        panels={name:Panel(reader,plan["panels"][name]) for name in ("fit","monitor")}
        for name,panel in panels.items():
            packed=np.load(support_path(name),mmap_mode="r")
            panel.grazing9=torch.as_tensor(np.unpackbits(packed,axis=-1,count=254,bitorder="big").astype(bool),device="cuda")
        refs=EvaluationReferences(plan,reader);fit=panels["fit"]
        full=Counter(reader.rows[int(i)]["terrain_id"] for i in np.load(ROOT/"features/c38_dense/fit_map_indices.npy"))
        chosen=Counter(fit.terrain)
        weights=torch.tensor([full[t]/chosen[t] for t in fit.terrain],device="cuda",dtype=torch.float32)
        weights/=weights.mean();weights_np=weights.cpu().numpy().astype(np.float64)
        den=torch.zeros(2,device="cuda",dtype=torch.float64)
        for start in range(0,len(fit.ids),32):
            den+=(fit.w[start:start+32]*weights[start:start+32,None,None,None]).sum((0,2,3),dtype=torch.float64)
        den=(den/STEPS).float()
        net=create(plan["initial_seed"]).cuda().to(memory_format=torch.channels_last)
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
        config={"campaign":"C155","arm":ARM,"arm_label":"C155_convnext_unet","protocol_sha256":identity,
            "protocol":plan,"panels":plan["panels"],"parameters":PARAMETERS,"GPU":gpu,
            "launch_device_snapshot":snapshot,"prior":"registered_FSPL","official_VAL_used":False,
            "_cc_dashboard":{"step_unit":"epoch","total_epochs":EPOCHS,"scope":"固定开发monitor628图/128地形；C82仅作计分参照",
                "baseline_source":"audit/c145_capacity/comparison_summary.json","C82":C82,
                "method_note":"架构单变量ConvNeXt-UNet319万参数；其余C153协议不变；Grazing高频误差同评估记录"}}
        if resume:
            assert json.loads((RUN/"config.json").read_text())["protocol_sha256"]==identity
        else:atomic(RUN/"config.json",config)
        oldtrack=json.loads((RUN/"tracking.json").read_text()) if resume else {}
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();data_rng=order_rng.get_state()
        kw={"parent_run_id":oldtrack["run_id"],"parent_checkpoint":str(RUN/"last.pt"),"start_step":last_epoch} if resume else {}
        swan=cc_tracking.init(name="C155_convnext_unet"+(" resume" if resume else ""),config=config,campaign="C155",
            description="Architecture-only four-level ConvNeXt U-Net versus C153; same raw68/FSPL/training60epochs; local Grazing diagnostics.",
            tags=["unet","architecture-only","registered-FSPL","60epochs","Grazing"],**kw)
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
            if epoch==0:
                control0=json.loads((ROOT/"runs/c153_standalone_c151a/history.json").read_text())[0]
                for name in panels:assert np.allclose(values[name]["rmse"],control0["evaluation"][name]["rmse"],rtol=0,atol=1e-10)
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
                for name,path in paths.items():
                    if epoch==EPOCHS:path.replace(RUN/f"last_{name}_prediction.npy")
                    else:path.unlink(missing_ok=True)
            if epoch==EPOCHS and improved:
                for name in panels:link_atomic(RUN/f"selected_{name}_prediction.npy",RUN/f"last_{name}_prediction.npy")
            cc_tracking.log(evaluation_metrics(row),step=epoch)
            print(json.dumps({"event":"evaluation","epoch":epoch,"fit":values["fit"]["rmse"],"monitor":values["monitor"]["rmse"],"best_epoch":best_epoch}),flush=True)
        if not resume:score_epoch(0,zero_head=True);save(RUN/"last.pt",state())
        eta_start=time.monotonic()
        for epoch in range(last_epoch+1,EPOCHS+1):
            order=torch.randperm(len(fit.ids),generator=order_rng)
            sums=torch.zeros(4,device="cuda",dtype=torch.float64);net.train()
            progress=tqdm(enumerate(order.split(plan["batch"]),1),total=STEPS,desc=f"C155 Epoch{epoch}/{EPOCHS}",mininterval=5)
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
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite C155 loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),plan["clip_grad_norm"])
                if not torch.isfinite(norm):raise RuntimeError("Nonfinite C155 gradient")
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

        last_h=digest(net)
        last_export={"tag":"last","checkpoint":"last.pt","checkpoint_sha256":sha(RUN/"last.pt"),
            "model_sha256":predictor_digest(last_h,plan),"net_state_sha256":last_h,"prediction_dtype":"float32",
            "forward_precision":"BF16 autocast","training_state":{"absolute_epoch":60,"epoch":60},
            "prior":"registered_FSPL","C82_in_model":False,"panels":{}}
        for name in panels:
            dest=RUN/f"last_{name}_prediction.npy"
            info={"model_sha256":last_export["model_sha256"],"panel_ids_sha256":ordered_hash(plan["panels"][name]),
                "shape":[len(plan["panels"][name]),2,256,256],"prediction_dtype":"float32","forward_precision":"BF16 autocast",
                "file_size_bytes":dest.stat().st_size,"prediction_sha256":sha(dest)}
            atomic(dest.with_suffix(".json"),info);last_export["panels"][name]={"file":dest.name,**info}
        atomic(RUN/"last_export.json",last_export)

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
    parser.add_argument("--prepare",action="store_true");parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--gpu-smoke",action="store_true");parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        p=make_protocol();print(json.dumps({"status":"prepared","parameters":PARAMETERS,"epochs":p["epochs"]}));return
    if args.preflight:
        p=read_protocol(False);v=preflight(p);atomic(AUDIT/"preflight.json",v);print(json.dumps(v));return
    if args.gpu_smoke:
        p=read_protocol(False);v=gpu_smoke(p);atomic(AUDIT/"gpu_smoke.json",v);print(json.dumps(v));return
    run(args.resume)
if __name__=="__main__":main()
