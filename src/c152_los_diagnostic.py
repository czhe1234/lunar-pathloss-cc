"""C152 step1: CPU-only LoS scales and C82 reference audit from frozen artifacts."""
from pathlib import Path
import os,sys,json,hashlib,time
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.c145_verify import validate_export
OUT=ROOT/"audit/c152_los_diagnostic"
RUN=ROOT/"runs/c145_capacity_native"
BANDS=("415MHz","5p775GHz")

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def atomic(p,x):
 tmp=p.with_suffix(p.suffix+".partial")
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
 tmp.replace(p)
def div(a,b):return np.divide(a,b,out=np.zeros_like(np.asarray(a,dtype=float)),where=np.asarray(b)>0)
class Stats:
 def __init__(self):self.n=np.zeros(2,np.int64);self.se=np.zeros(2);self.err=np.zeros(2)
 def add(self,e,m):
  self.n+=m.sum((1,2));self.se+=np.where(m,e*e,0).sum((1,2),dtype=np.float64);self.err+=np.where(m,e,0).sum((1,2),dtype=np.float64)
 def result(self,all_s,los):
  return dict(count=self.n.tolist(),pixel_percent_all=(100*div(self.n,all_s.n)).tolist(),pixel_percent_LoS=(100*div(self.n,los.n)).tolist(),
   MeanBias_dB=div(self.err,self.n).tolist(),RMSE_dB=np.sqrt(div(self.se,self.n)).tolist(),SSE_dB2=self.se.tolist(),
   SSE_percent_all=(100*div(self.se,all_s.se)).tolist(),SSE_percent_LoS=(100*div(self.se,los.se)).tolist())
def box(v):
 return sum(v[:,dy:dy+254,dx:dx+254] for dy in range(3) for dx in range(3))/9.
def nine(m):return np.logical_and.reduce([m[:,dy:dy+254,dx:dx+254] for dy in range(3) for dx in range(3)])
class Rough:
 def __init__(self):
  self.n=np.zeros(2,np.int64)
  self.acc={k:np.zeros(2) for k in ("GT_hp2","Pred_hp2","error_hp2","error_localmean2","error2","cross","GT_hp","Pred_hp","GT_Pred_hp")}
 def add(self,gt,pred,valid):
  hg=gt[:,1:-1,1:-1]-box(gt);hp=pred[:,1:-1,1:-1]-box(pred)
  he=hp-hg;lo=box(pred-gt);e=(pred-gt)[:,1:-1,1:-1]
  vals=dict(GT_hp2=hg*hg,Pred_hp2=hp*hp,error_hp2=he*he,error_localmean2=lo*lo,error2=e*e,cross=2*he*lo,GT_hp=hg,Pred_hp=hp,GT_Pred_hp=hg*hp)
  self.n+=valid.sum((1,2))
  for k,v in vals.items():self.acc[k]+=np.where(valid,v,0).sum((1,2),dtype=np.float64)
 def result(self):
  z={k:div(v,self.n) for k,v in self.acc.items()}
  cov=z["GT_Pred_hp"]-z["GT_hp"]*z["Pred_hp"]
  vg=z["GT_hp2"]-z["GT_hp"]**2;vp=z["Pred_hp2"]-z["Pred_hp"]**2
  return dict(count=self.n.tolist(),GT_minus_FSPL_highpass_RMS_dB=np.sqrt(z["GT_hp2"]).tolist(),
   Pred_minus_FSPL_highpass_RMS_dB=np.sqrt(z["Pred_hp2"]).tolist(),highpass_error_RMS_dB=np.sqrt(z["error_hp2"]).tolist(),
   localmean_error_RMS_dB=np.sqrt(z["error_localmean2"]).tolist(),center_error_RMSE_dB=np.sqrt(z["error2"]).tolist(),
   error_MSE_dB2=z["error2"].tolist(),cross_term_dB2=z["cross"].tolist(),
   decomposition_error_dB2=(z["error_hp2"]+z["error_localmean2"]+z["cross"]-z["error2"]).tolist(),
   GT_Pred_highpass_correlation=div(cov,np.sqrt(np.maximum(0,vg*vp))).tolist())
def main():
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("Pure CPU only")
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/"result.json").exists():raise FileExistsError("Diagnostic already completed; reuse it")
 start=time.monotonic()
 summary_path=ROOT/"audit/c149_error_audit/summary.json";summary=read(summary_path)
 plan=read(ROOT/"audit/c149_error_audit/protocol.json");ids=plan["ordered_panel_ids"]
 assert ids==read(ROOT/"audit/c145_capacity/protocol.json")["panels"]["monitor"]
 pp=RUN/"selected_monitor_prediction.npy"
 identity=validate_export(RUN,"selected",{"monitor":ids},{"monitor":pp})
 assert identity["model_sha256"]==summary["model_sha256"]
 profiles_path=ROOT/"audit/c149_error_audit/profiles.npy"
 assert sha(profiles_path)==summary["profile_cache_sha256"]
 frozen={str(p.relative_to(ROOT)):sha(p) for p in (summary_path,ROOT/"audit/c149_error_audit/protocol.json",ROOT/"src/c149_error_audit.py",ROOT/"src/c149_profiles.cu")}
 pred=np.load(pp,mmap_mode="r");profiles=np.load(profiles_path,mmap_mode="r")
 base_path=ROOT/"audit/c88_mesh/base.npy";base=np.load(base_path,mmap_mode="r")
 b_ids=read(base_path.parent/"base_index.json")
 if isinstance(b_ids,dict):b_ids=b_ids["ids"]
 blookup={int(i):j for j,i in enumerate(b_ids)}
 feature_dir=ROOT/"features/c150_diffraction/part0of1";fids=read(feature_dir/"indices.json");fl={int(i):j for j,i in enumerate(fids)}
 feature=np.load(feature_dir/"features.npy",mmap_mode="r")
 assert read(feature_dir/"complete.json")["status"]=="complete"
 data=ROOT.parent/"data";raw_audit=read(ROOT/"audit/raw_audit.json")
 for name in ("train_pl415.npy","train_pl58.npy","train_mask415.npy","train_mask58.npy"):
  p=data/name;st=p.stat();old=raw_audit["files"][name]
  assert (st.st_size,st.st_mtime_ns)==(old["size_bytes"],old["mtime_ns"])
 labels=[np.load(data/f"train_pl{b}.npy",mmap_mode="r") for b in ("415","58")]
 masks=[np.load(data/f"train_mask{b}.npy",mmap_mode="r") for b in ("415","58")]
 wave=np.asarray(summary["wavelength_m"])[:,None,None]
 names=("all","LoS","Clear","Grazing","NLoS")
 stats={m:{k:Stats() for k in names} for m in ("C145","C82")}
 rough={k:Rough() for k in ("center_Clear_all9_valid","all9_Clear_valid")}
 for j,i in enumerate(ids):
  y=np.stack([v[i] for v in labels]).astype(np.float64)
  w=np.stack([np.unpackbits(v[i],bitorder="big").reshape(256,256) for v in masks]).astype(bool)
  p=np.asarray(pred[j],np.float64);b=np.asarray(base[blookup[i]],np.float64)
  fs=np.asarray(feature[fl[i],10:12],np.float64)
  h=np.asarray(profiles[j,0]);nu=np.asarray(profiles[j,1],np.float64)[None]/np.sqrt(wave)
  assert np.isfinite(y[w]).all() and np.isfinite(p[w]).all() and np.isfinite(b[w]).all() and np.isfinite(fs).all()
  los=w&(h[None]<0)
  clear=los&(nu<-.7);grazing=los&(nu>=-.7)&(nu<0);nlos=w&(h[None]>=0)
  assert np.array_equal(clear|grazing,los)
  groups=dict(all=w,LoS=los,Clear=clear,Grazing=grazing,NLoS=nlos)
  for m,e in (("C145",p-y),("C82",b-y)):
   for k,mask in groups.items():stats[m][k].add(e,mask)
  valid9=nine(w)&np.isfinite(box(y))&np.isfinite(box(p))
  rough["center_Clear_all9_valid"].add(y-fs,p-fs,clear[:,1:-1,1:-1]&valid9)
  rough["all9_Clear_valid"].add(y-fs,p-fs,nine(clear)&valid9)
  if (j+1)%100==0:print(json.dumps(dict(stage="CPU_statistics",maps=j+1,total_maps=len(ids))),flush=True)
 assert np.array_equal(stats["C145"]["all"].n,summary["all"]["valid_pixels"])
 assert np.allclose(stats["C145"]["all"].se,summary["all"]["SSE_dB2"],rtol=1e-12,atol=1e-5)
 assert np.allclose(stats["C82"]["all"].se,summary["all"]["C82_SSE_dB2"],rtol=1e-12,atol=1e-5)
 for name,old in (("LoS","profile_LoS"),("Clear","Fresnel_Clear"),("Grazing","Fresnel_Grazing"),("NLoS","profile_NLoS")):
  assert np.array_equal(stats["C145"][name].n,summary["groups"][old]["valid_pixels"])
  assert np.allclose(stats["C145"][name].se,summary["groups"][old]["SSE_dB2"],rtol=1e-12,atol=1e-5)
 assert all(sha(ROOT/k)==v for k,v in frozen.items())
 result=dict(status="complete",created_utc=datetime.now(timezone.utc).isoformat(),scope=summary["scope"],
  band_order=list(BANDS),training_run=False,network_inference_run=False,geometry_run=False,model_weights_modified=False,torch_imported="torch" in sys.modules,
  definitions=dict(error="pred-GT",highest="registered-point C149 highest_m",nu="C149 nu_lambda1m/sqrt(lambda_m)",Clear="valid & highest<0 & nu<-0.7",
   Grazing="valid & highest<0 & -0.7<=nu<0",NLoS="valid & highest>=0",pixel_denominators="Both all-valid and LoS-valid denominators retained",MSE_contribution="group SSE/all-valid SSE; LoS denominator also supplied",
   FSPL="Frozen C150 raw Float32 registered3D FSPL planes10:12; no geometry rerun",
   roughness="RMS of v-center - equal mean of all nine 3x3 pixels, v=GT-FSPL or Pred-FSPL; no image padding; all9label-valid; two Clear support definitions",
   decomposition="e=highpass(e)+localmean3x3(e); MSE=HP_MSE+LP_MSE+cross; components are not orthogonal",
   interpretation_limit="High-frequency structure is not by itself proof of physical interference, label noise or unpredictability."),
  regions={m:{k:s.result(stats[m]["all"],stats[m]["LoS"]) for k,s in ss.items()} for m,ss in stats.items()},
  roughness={k:v.result() for k,v in rough.items()},
  validation=dict(original_C149_counts_and_SSE_match=True,C82_cached_counts_SSE_match=True,LoS_partition_exact=True,frozen_sources_unchanged=True),
  source_identity=dict(source_sha256=sha(Path(__file__)),frozen=frozen,prediction=identity["panels"]["monitor"],C82_prediction_sha256=sha(base_path),
   C82_index_sha256=sha(base_path.parent/"base_index.json"),feature_complete_sha256=sha(feature_dir/"complete.json"),feature_cache_sha256_recorded=read(feature_dir/"complete.json")["features_sha256"]),
  elapsed_seconds=time.monotonic()-start)
 assert all(max(abs(x) for x in v["decomposition_error_dB2"])<1e-8 for v in result["roughness"].values())
 atomic(OUT/"result.json",result)
 print(json.dumps(dict(status="complete",path=str(OUT/"result.json"),elapsed_seconds=result["elapsed_seconds"])),flush=True)
if __name__=="__main__":main()
