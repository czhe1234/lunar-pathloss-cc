"""C152 one candidate: C82 prior/context + diffraction10 + aligned raw68 Global.
Fresh12 full-FIT, matching C1238 + C1454 prefix. No old control is rerun.
"""
from pathlib import Path
import argparse,os,sys,json,math,time,hashlib,subprocess
from datetime import datetime,timezone
from collections import Counter
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import GPU_UUID,APPROVED_GPU_UUIDS,gpu_guard,check_source_unchanged
from src.c152_field import create,digest,inputs,normalize_diffraction,PARAMETERS
from src.c150_train import Reader,EvaluationReferences,cache_identity,schedule_lr,optimizer_for,second_scheduler
from src.c150_train import atomic,save,link_atomic,sha,ordered_hash,groups,GROUPS,VERIFIED_GROUPS,C82,BANDS
from src.c146_data import Panel as SolverPanel
from src import cc_tracking
import numpy as np
import torch
from tqdm import tqdm
AUDIT=ROOT/"audit/c152_fair_global"
RUN_ROOT=ROOT/"runs/c152_c82_diffraction_global"
ARM="candidate"
UUIDS={ARM:"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"}
REGIONS=("NLoS","C145_NLoS_underprediction_ge5dB","c73_total_floor_coded")
class Panel(SolverPanel):
 def __init__(self,reader,ids,device="cuda"):
  super().__init__(reader,ids,device)
  self.diff=torch.empty((len(ids),10,256,256),device=device,dtype=torch.float16)
  self.reference=None
  for start in range(0,len(ids),32):
   end=min(start+32,len(ids));raw=torch.as_tensor(reader.raw_diffraction(ids[start:end]),device=device)
   assert torch.isfinite(raw).all()
   self.diff[start:end]=normalize_diffraction(raw[:,:10]).half()
 def batch(self,ix):
  if self.reference is None:raise ValueError("C82 reference is not bound")
  basic,physical,legacy_fspl,y,w=super().batch(ix)
  baseline=self.reference[ix]
  return inputs(basic,physical,baseline,legacy_fspl,self.diff[ix].float()),baseline,y,w

class AuxRegions:
 """Scoring-only C145/geometry/floor masks. Never supplied to model inputs."""
 def __init__(self,panel,plan):
  ids=plan["panels"]["monitor"];assert ids==panel.ids
  prof=np.load(ROOT/"audit/c149_error_audit/profiles.npy",mmap_mode="r")
  self.ref145=torch.as_tensor(np.array(np.load(ROOT/"runs/c145_capacity_native/selected_monitor_prediction.npy",mmap_mode="r")),device="cuda")
  h=torch.as_tensor(np.array(prof[:,0]),device="cuda")
  nlos=(h>=0)[:,None].expand(-1,2,-1,-1)
  self.masks={"NLoS":nlos,
   "C145_NLoS_underprediction_ge5dB":nlos&(self.ref145-panel.y<=-5),
   "c73_total_floor_coded":panel.physical[:,[41,42]].float()<=-3}
  self.reference=panel.reference
 def batch(self,sl,y,w,pred):
  masks=torch.stack([v[sl]&w for v in self.masks.values()])
  out={}
  for name,value in (("candidate",pred),("C145",self.ref145[sl]),("C82",self.reference[sl])):
   e=value.double()-y.double()
   out[name+"_sse"]=torch.where(masks,e[None].square(),0).sum((3,4)).cpu().numpy()
   out[name+"_error_sum"]=torch.where(masks,e[None],0).sum((3,4)).cpu().numpy()
  out["count"]=masks.sum((3,4)).cpu().numpy()
  return out

def predictor_digest(net_hash,arm,plan):
 value={"net_state_sha256":net_hash,"arm":arm,"prior":plan["arms"][arm],"feature_cache_sha256":plan["cache"]["features_sha256"],
  "C82_prior_identity":plan["C82_prediction_sources"],"global_pack":"original derived13 + physical57 pixel_unshuffle4"}
 return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def make_protocol():
 if RUN_ROOT.exists():raise FileExistsError("Do not rewrite protocol after a candidate exists")
 diag=ROOT/"audit/c152_los_diagnostic/result.json";d=json.loads(diag.read_text())
 assert d["status"]=="complete" and not d["training_run"] and not d["network_inference_run"]
 old=ROOT/"audit/c150_diffraction_training/protocol.json";base=json.loads(old.read_text());plan=deepcopy(base)
 parent=ROOT/"runs/c145_capacity_native"
 sources={
  "fit":ROOT/"audit/c101_residual/base0of1/prediction.npy",
  "monitor":ROOT/"audit/c88_mesh/base.npy"}
 identities={name:{"path":str(path.relative_to(ROOT)),"sha256":sha(path)} for name,path in sources.items()}
 model=create(123041)
 plan.update(campaign="C152",created_utc=datetime.now(timezone.utc).isoformat(),
  question="At preserved C82 prior/context and matched12epoch schedule, do diffraction10 plus aligned full-input Global70 improve selected12?",
  arms={ARM:"C82_cached + Field(raw68)"},C82_role="Frozen external output prior AND two context inputs (C82-legacyFSPL)/20; evaluation reference also retained",
  channel_contract="11 original basic +45 unchanged c73 +2 real C82 context +10 normalized C150 diffraction",
  parameters=PARAMETERS,initial_state_sha256=digest(model),initialization="Fresh same seed123041; all shape-compatible C123 random tensors retained, local old58 copied and extra10 zero; Global original13-derived prefix copied and new57 columns zero; no trained weights loaded",
  global_input_channels=70,global_input="Preserve original13 derived terrain/TX fields; append all45 c73+2 C82 context+10 diffraction; pixel_unshuffle4 -> Conv1120to384; raw model input68",
  FFN_hidden={"global":[768]*8,"local":[128,128],"refine":[256,256]},
  parameter_change_vs_C145=PARAMETERS-15214786,parameter_scope="Trainable Field only; requested larger input stem increases2.356% without extra FFN redistribution",
  prior="C82_cached, unchanged; registered FSPL/Ld NOT added to output",
  C82_prediction_sources=identities,step1={"path":str(diag.relative_to(ROOT)),"sha256":sha(diag)},
  GPU_UUID_by_arm=UUIDS,selection="One joint-mean monitor minimum among0,4,8,12, including exactC82 at0; no per-band selections",
  comparison_limit="C152 jointly adds diffraction and raw68 Global wiring; not separate causal effects. Original13 derived Global fields and their initial weights preserved; physical57 appended with zero initial connections. Raw input68/internal Global70. C145 full search764 versus this fixed12 prefix; development monitor reused.",
  excluded_implementation_attempt="audit/c152_fair_global/invalid_global_input_status.json",no_control_rerun=True,repetition="Exactly one fresh12/4224 candidate, one seed; frozen C145 selected12 reused; no extensions/sweeps",
  diagnostic_regions={"NLoS":"highest>=0, scoring only","C145_NLoS_underprediction_ge5dB":"highest>=0 and fixed(C145selected12-GT)<=-5, scoring only",
   "c73_total_floor_coded":"original compact41/42<=-3, scoring only; Q=P*dface^2 code, not exact rawP saturation"},
  raw_priors="C82 cached Float32 output; old basic FSPL only builds original C82 excess inputs",
  schedule_scope="Same C1238+freshAdamWreset+C1454 loss/LR/order prefix; same C82 prior/context; fresh from random rather than loading epoch8",
  source_sha256={})
 files=["src/c152_field.py","src/c152_train.py","src/c150_train.py","src/c150_field.py","src/c146_data.py","src/c123_lossless.py",
  "src/c99_axial.py","src/c117_factorized.py","src/c3_geometry.py","src/cc_core.py","src/cc_tracking.py",
  "src/c152_report.py","src/c145_verify.py","src/c150_report.py","src/c151_report.py","src/c152_los_diagnostic.py"]
 plan["source_sha256"]={f:sha(ROOT/f) for f in files}
 plan["score_only_source_identity"]={
  "profiles": {"path":"audit/c149_error_audit/profiles.npy","sha256":json.loads((ROOT/"audit/c149_error_audit/summary.json").read_text())["profile_cache_sha256"]},
  "C145": {"path":"runs/c145_capacity_native/selected_monitor_prediction.npy","sha256":json.loads((ROOT/"runs/c145_capacity_native/selected_export.json").read_text())["panels"]["monitor"]["prediction_sha256"]}}
 plan["cache"]=cache_identity(plan["panels"],require=True)
 assert sum(p.numel() for p in model.parameters())==PARAMETERS
 atomic(AUDIT/"protocol.json",plan);return plan
def read_protocol(require_cache=True):
 p=json.loads((AUDIT/"protocol.json").read_text())
 for name,h in p["source_sha256"].items():
  if sha(ROOT/name)!=h:raise ValueError("Frozen source changed: "+name)
 assert sha(ROOT/p["step1"]["path"])==p["step1"]["sha256"]
 assert p["panels"]==json.loads((ROOT/"audit/c145_capacity/protocol.json").read_text())["panels"]
 assert p["epochs"]==12 and p["updates_per_arm"]==4224
 for value in p["C82_prediction_sources"].values():assert sha(ROOT/value["path"])==value["sha256"]
 for value in p["score_only_source_identity"].values():assert sha(ROOT/value["path"])==value["sha256"]
 if require_cache:assert cache_identity(p["panels"],require=True)==p["cache"]
 return p

@torch.inference_mode()
def evaluate(net,panel,name,references,arm,batch,prediction_path,weights=None,zero_head=False,aux=None):
 net.eval();n=len(panel.ids);mm=np.lib.format.open_memmap(prediction_path,mode="w+",dtype=np.float32,shape=(n,2,256,256))
 sse=np.empty((6,n,2),np.float64);counts=np.empty((6,n,2),np.int64)
 diag={k:np.empty((3,n,2),np.int64 if k=="count" else np.float64) for k in ("count","candidate_sse","candidate_error_sum","C145_sse","C145_error_sum","C82_sse","C82_error_sum")} if name=="monitor" else {}
 for start in tqdm(range(0,n,batch),desc="C152 "+name,mininterval=5):
  stop=min(start+batch,n);sl=slice(start,stop);x,baseline,y,w=panel.batch(sl)
  if zero_head:p=baseline
  else:
   with torch.autocast("cuda",dtype=torch.bfloat16):p=baseline+net(x).float()
  if not torch.isfinite(p).all():raise RuntimeError("Nonfinite C152 prediction")
  mm[sl]=p.cpu().numpy();error=p.double()-y.double()
  for gi,m in enumerate(groups(w,y,references.baseline[name][sl])):
   sse[gi,start:stop]=torch.where(m,error.square(),0).sum((2,3)).cpu().numpy();counts[gi,start:stop]=m.sum((2,3)).cpu().numpy()
  if diag:
   for k,v in aux.batch(sl,y,w,p).items():diag[k][:,start:stop]=v
 mm.flush();del mm
 assert np.array_equal(counts,references.counts[name])
 cs=counts.sum(1);ss=sse.sum(1);bs=references.sse[name].sum(1)
 if zero_head:assert np.allclose(ss,bs,rtol=1e-12,atol=1e-5),"Initial output must reproduce C82 in every fixed group"
 result={"rmse":np.sqrt(ss[0]/cs[0]).tolist(),"groups":{}}
 for i,g in enumerate(GROUPS):result["groups"][g]={"RMSE":np.sqrt(ss[i]/cs[i]).tolist(),"SSE":ss[i].tolist(),"count":cs[i].tolist(),"C82_RMSE":np.sqrt(bs[i]/cs[i]).tolist(),"C82_SSE":bs[i].tolist()}
 if weights is not None:
  ws=(sse[0]*weights[:,None]).sum(0);wc=(counts[0]*weights[:,None]).sum(0)
  result.update(weighted_MSE_objective=float(np.mean(ws/wc)),weighted_RMSE=np.sqrt(ws/wc).tolist())
 terrains,inv=np.unique(np.asarray(panel.terrain),return_inverse=True);ts=np.zeros((len(terrains),2));tc=np.zeros((len(terrains),2),np.int64)
 np.add.at(ts,inv,sse[0]);np.add.at(tc,inv,counts[0])
 statistics={name+"_map_ids":np.asarray(panel.ids,np.int64),name+"_map_group_sse":sse,name+"_map_group_count":counts,
  name+"_terrain_names":terrains,name+"_map_terrain_index":inv,name+"_terrain_sse":ts,name+"_terrain_count":tc}
 if diag:
  result["diagnostic_regions"]={}
  for j,region in enumerate(REGIONS):
   count=diag["count"][j].sum(0);v={"count":count.tolist()}
   for model in ("candidate","C145","C82"):
    se=diag[model+"_sse"][j].sum(0);err=diag[model+"_error_sum"][j].sum(0)
    v[model]={"SSE":se.tolist(),"RMSE":np.sqrt(se/count).tolist(),"MeanBias":(err/count).tolist()}
   result["diagnostic_regions"][region]=v
  statistics.update({"monitor_diagnostic_region_names":np.asarray(REGIONS),**{"monitor_diagnostic_"+k:v for k,v in diag.items()}})
 net.train();return result,statistics
def metric_log(row):
 v={"progress/epoch":row["epoch"],"progress/total_epochs":12,"progress/updates":row["updates"],"time/elapsed_seconds":row["elapsed_seconds"],"time/eta_seconds":row["eta_seconds"],"optim/learning_rate":row["learning_rate"],"optim/phase":row["phase"],"train/data_loss":row["evaluation"]["fit"]["weighted_MSE_objective"]/25}
 for p in ("fit","monitor"):
  for j,b in enumerate(BANDS):v[f"{p}/RMSE_{b}_dB"]=row["evaluation"][p]["rmse"][j]
 for j,b in enumerate(BANDS):v[f"monitor/C145_delta_RMSE_{b}_dB"]=row["evaluation"]["monitor"]["rmse"][j]-[2.1412210108233403,2.4082364795626496][j]
 for region,j,key in (("NLoS",0,"NLoS415"),("C145_NLoS_underprediction_ge5dB",0,"fixed_shadow_under415"),("c73_total_floor_coded",1,"floor5775")):
  x=row["evaluation"]["monitor"]["diagnostic_regions"][region]["candidate"]
  v[f"diagnostic/{key}/RMSE_dB"]=x["RMSE"][j];v[f"diagnostic/{key}/MeanBias_dB"]=x["MeanBias"][j]
 return v

def preflight(plan):
 a=create(plan["initial_seed"]);b=create(plan["initial_seed"]);assert digest(a)==digest(b)==plan["initial_state_sha256"]
 assert sum(p.numel() for p in a.parameters())==15573314
 from src.c123_lossless import create as oldcreate
 old=oldcreate("native").state_dict();common=[]
 for k,v in a.state_dict().items():
  if k.startswith(("stem.","film.","local.0.")):continue
  assert torch.equal(v,old[k]),k;common.append(k)
 assert torch.equal(a.local[0].weight[:,:58],old["local.0.weight"]) and torch.count_nonzero(a.local[0].weight[:,58:])==0
 torch.set_num_threads(4)
 basic=torch.randn(1,11,256,256)*.1;physical=torch.randn(1,45,256,256)
 base=torch.ones(1,2,256,256)*100;fs=torch.ones_like(base)*80;raw=torch.randn(1,10,256,256);raw[:,[2,3,4,7,8,9]]=raw[:,[2,3,4,7,8,9]].abs()
 x=inputs(basic,physical,base,fs,normalize_diffraction(raw))
 assert torch.equal(x[:,11:56],physical) and torch.equal(x[:,56:58],torch.ones_like(base))
 assert a.stem.in_channels==1120 and a.film.in_features==68 and a.local[0].in_channels==68
 y=a(x);assert y.shape==(1,2,256,256) and torch.count_nonzero(y)==0
 with torch.no_grad():a.head.weight.normal_(0,.001)
 original=oldcreate("native")
 with torch.no_grad():
  original.head.weight.copy_(a.head.weight);original.head.bias.copy_(a.head.bias)
  probe_old=original(x[:,:58]);probe_new=a(x)
  torch.testing.assert_close(probe_new,probe_old,rtol=2e-5,atol=2e-6)
 a(x).square().mean().backward()
 assert torch.isfinite(a.stem.weight.grad).all()
 assert (a.stem.weight.grad.abs().sum((0,2,3))>0).all()
 assert not torch.cuda.is_initialized()
 return {"status":"CPU_preflight_passed","parameters":15573314,"initial_state_sha256":plan["initial_state_sha256"],"C82_context_formula_passed":True,
  "C73_all45_unchanged":True,"Global_all70_columns_end_to_end_active":True,"zero_head_exact_C82":True,"original_Field_function_preserved_at_initialization":True,"shared_C123_random_tensors_identical":len(common),"CUDA_initialized":False,
  "schedule_probe":[schedule_lr(e,j,plan) for e,j in ((1,1),(8,352),(9,1),(9,100),(9,101))]}


def run(resume=False):
    arm=ARM
    plan=read_protocol();run_dir=RUN_ROOT;identity=sha(AUDIT/"protocol.json")
    if GPU_UUID!=UUIDS[arm] or os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU_UUID or GPU_UUID not in APPROVED_GPU_UUIDS:
        raise ValueError("Arm launch must use its frozen approved full GPU UUID")
    if run_dir.exists() and not resume:raise FileExistsError("Inspect C152 arm; use explicit --resume, never repeat")
    if resume and (run_dir/"result.json").exists() and json.loads((run_dir/"result.json").read_text()).get("status")=="complete":
        raise ValueError("This C152 arm already completed")
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
        for name,panel in panels.items():panel.reference=refs.baseline[name]
        aux=AuxRegions(panels["monitor"],plan)
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
        config={"campaign":"C152","arm":arm,"protocol_sha256":identity,"protocol":plan,"panels":plan["panels"],
                "GPU":gpu,"launch_device_snapshot":gpu_snapshot,"parameters":plan["parameters"],"initial_state_sha256":plan["initial_state_sha256"],
                "prior":plan["arms"][arm],"official_VAL_used":False,
                "_cc_dashboard":{"step_unit":"epoch","total_epochs":12,
                    "scope":"固定开发monitor628图/128地形；C82 beta-fit80和selection48均已用于开发",
                    "baseline_source":"audit/c145_capacity/comparison_summary.json","C82":C82,"method_note":"保留C82输出基底和上下文；对照C145 selected12；Global70+diffraction。"}}
        if resume:
            old_config=json.loads((run_dir/"config.json").read_text())
            if old_config["protocol_sha256"]!=identity or old_config["arm"]!=arm or old_config["panels"]!=plan["panels"]:
                raise ValueError("Run config identity differs")
        else:atomic(run_dir/"config.json",config)
        rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state();rng_order=generator.get_state()
        previous=json.loads((run_dir/"tracking.json").read_text()) if resume else {}
        kwargs={"parent_run_id":previous["run_id"],"parent_checkpoint":str(run_dir/"last.pt"),"start_step":last_epoch} if resume else {}
        swan=cc_tracking.init(name="C152 C82 + diffraction + Global70"+(" resume" if resume else ""),
            config=config,campaign="C152",description=plan["arms"][arm]+"; fresh12epochs; preserved C82 prior/context; local only.",
            tags=["C82-prior","global68","diffraction","single-candidate"],**kwargs)
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
                    weights=weights_np if name=="fit" else None,zero_head=zero_head,aux=aux)
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
            net.train();progress=tqdm(enumerate(order.split(plan["batch"]),1),total=352,desc=f"C152{arm} Epoch{epoch}/12",mininterval=5)
            for batch_index,positions in progress:
                ix=positions.cuda();x,baseline,y,w=fit.batch(ix)
                rate=schedule_lr(epoch,batch_index,plan)
                if rate is not None:
                    for group in opt.param_groups:group["lr"]=rate
                with torch.autocast("cuda",dtype=torch.bfloat16):prediction=baseline+net(x).float()
                error=torch.where(w,prediction-y,0.);ww=weights[ix,None,None,None]
                numerator=(error.square()*ww).sum((0,2,3));count=(w*ww).sum((0,2,3))
                loss=(numerator/(count if phase==1 else den)).mean()/25
                if updates<3 and torch.count_nonzero(x[:,56:58])==0:raise ValueError("Real C82 context is missing")
                if updates==0 and not torch.equal(prediction,baseline):
                    raise ValueError("Fresh zero head does not equal frozen C82")
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite C152 data loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
                if not torch.isfinite(norm):raise RuntimeError("Nonfinite C152 gradient")
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
 parser.add_argument("--prepare-protocol",action="store_true");parser.add_argument("--preflight",action="store_true")
 parser.add_argument("--run",action="store_true");parser.add_argument("--resume",action="store_true")
 args=parser.parse_args()
 if args.prepare_protocol:
  p=make_protocol();print(json.dumps({"parameters":p["parameters"],"initial_state_sha256":p["initial_state_sha256"],"cache_ready":p["cache"]["ready"]}));return
 if args.preflight:
  p=read_protocol(require_cache=False);v=preflight(p);atomic(AUDIT/"preflight.json",v);print(json.dumps(v));return
 if not args.run:parser.error("--run required; no accidental launch")
 run(args.resume)
if __name__=="__main__":main()
