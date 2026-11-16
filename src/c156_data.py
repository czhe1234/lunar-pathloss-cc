"""C156 CPU-only data and frozen arithmetic replay; no neural network calls."""
from pathlib import Path
import ast,csv,ctypes,hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/"data";OUT=ROOT/"audit/c156_attribution"
BANDS=("415MHz","5p775GHz");STAGES=("base","local","graph","blend","calibration_replay","C82","C155","FSPL_Deygout")
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def identity(p):
 s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".partial")
 t.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def box(x):
 return sum(x[...,dy:dy+254,dx:dx+254] for dy in range(3) for dx in range(3))/9.
def nine(x):return np.logical_and.reduce([x[...,dy:dy+254,dx:dx+254] for dy in range(3) for dx in range(3)])
class Data:
 def __init__(self):
  self.plan=load(ROOT/"audit/c155_unet/protocol.json");self.ids=self.plan["panels"]["monitor"]
  self.rows=list(csv.DictReader((DATA/"train_index.csv").open()))
  self.h=np.load(DATA/"train_hm.npy",mmap_mode="r")
  old=ROOT/"runs/c54_combo_dev/bf16_compiled/all_shard0of1"
  self.cids=load(old/"indices.json")["index_rows"];self.cl={i:j for j,i in enumerate(self.cids)}
  assert all(i in self.cl for i in self.ids) and len(self.ids)==628
  assert load(ROOT/"runs/c72_observed/full_internal/indices.json")==self.cids
  assert load(ROOT/"runs/c73_wide_real/full_internal/indices.json")==self.cids
  assert load(ROOT/"audit/c88_mesh/base_index.json")["ids"]==self.cids
  self.base=[np.load(old/f"pl{b}.npy",mmap_mode="r") for b in ("415","58")]
  self.corr=np.load(ROOT/"runs/c72_observed/full_internal/correction.npy",mmap_mode="r")
  self.graph=np.load(ROOT/"runs/c73_wide_real/full_internal/prediction.npy",mmap_mode="r")
  self.c82=np.load(ROOT/"audit/c88_mesh/base.npy",mmap_mode="r")
  self.c155=np.load(ROOT/"runs/c155_unet/selected_monitor_prediction.npy",mmap_mode="r")
  assert load(ROOT/"runs/c155_unet/selected_export.json")["training_state"]["epoch"]==52
  self.profiles=np.load(ROOT/"audit/c149_error_audit/profiles.npy",mmap_mode="r")
  assert load(ROOT/"audit/c149_error_audit/protocol.json")["ordered_panel_ids"]==self.ids
  self.wave=np.asarray(load(ROOT/"audit/c149_error_audit/summary.json")["wavelength_m"])[:,None,None]
  d=ROOT/"features/c150_diffraction/part0of1"
  self.diff=np.load(d/"features.npy",mmap_mode="r");self.fl={i:j for j,i in enumerate(load(d/"indices.json"))}
  self.labels=[np.load(DATA/f"train_pl{b}.npy",mmap_mode="r") for b in ("415","58")]
  self.masks=[np.load(DATA/f"train_mask{b}.npy",mmap_mode="r") for b in ("415","58")]
  self.c73={}
  for folder in ("c73_compact","c73_dev_extra_compact"):
   d=ROOT/"features"/folder/"part0of1"
   a=np.load(d/"features.npy",mmap_mode="r")
   for j,i in enumerate(load(d/"indices.json")):
    assert i not in self.c73;self.c73[i]=(a,j)
  self.replay=None
 def map(self,j,components=True):
  i=self.ids[j];k=self.cl[i];row=self.rows[i]
  h=np.asarray(self.h[int(row["hm_row"])],np.float32)
  y=np.stack([v[i] for v in self.labels]).astype(np.float64)
  valid=np.stack([np.unpackbits(v[i],bitorder="big").reshape(256,256) for v in self.masks]).astype(bool)
  c82=np.asarray(self.c82[k],np.float32);c155=np.asarray(self.c155[j],np.float32)
  prof=np.asarray(self.profiles[j]);nu=prof[1][None].astype(float)/np.sqrt(self.wave)
  raw=np.asarray(self.diff[self.fl[i]])
  ca,ci=self.c73[i];f=np.asarray(ca[ci])
  result={"map_id":i,"sample_id":row["sample_id"],"terrain_id":row["terrain_id"],"row":row,"h":h,"y":y,"valid":valid,
   "C82":c82,"C155":c155,"highest":prof[0],"nu":nu,"profile_flags":prof[4].astype(np.int32),
   "FSPL":raw[10:12].astype(float),"Ld":raw[[3,8]].astype(float),
   "floor":f[[41,42]]<=-3,"vis":np.array(f[16],dtype=np.float32)}
  if components:
   base=np.stack([v[k] for v in self.base]).astype(np.float32)
   local=base+np.asarray(self.corr[k],np.float32);graph=np.asarray(self.graph[k],np.float32)
   blend=.5*local+.5*graph
   if self.replay is None:self.replay=Calibrator()
   replay,corr=self.replay(base,local,blend,h,int(row["tx_row"]),int(row["tx_col"]),result["vis"])
   result.update(base=base,local=local,graph=graph,blend=blend,calibration_replay=replay,calibration_correction=corr)
  result["FSPL_Deygout"]=result["FSPL"]+result["Ld"]
  return result
class Calibrator:
 def __init__(self):
  import torch
  from torch.nn import functional as F
  torch.set_num_threads(4)
  self.torch=torch;cfg=load(ROOT/"runs/c82_selected/spectrum.json")
  assert cfg["mode"]=="full" and len(cfg["beta"][0])==117
  self.beta=[torch.tensor(v,dtype=torch.float32) for v in cfg["beta"]];self.mode=cfg["mode"]
  source=ROOT/"src/c82_blend.py";tree=ast.parse(source.read_text())
  functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ("smooth","field_features")]
  ns={"torch":torch,"F":F};exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),"exec"),ns)
  self.function=ns["field_features"]
  self.yy,self.xx=torch.meshgrid(torch.arange(256),torch.arange(256),indexing="ij")
  self.freq=torch.tensor([415e6,5.775e9])[None,:,None,None]*(4*np.pi/299792458)
 def __call__(self,base,local,blend,h,tr,tc,vis):
  t=self.torch
  with t.no_grad():
   B=t.tensor(base[None]);L=t.tensor(local[None]);P=t.tensor(blend[None]);G=2*P-L
   H=t.tensor(np.array(h));altitude=H[min(tr+1,255),tc].floor()+3
   d=t.sqrt((self.xx-tc).square()+(self.yy-tr).square()+(H+1-altitude).square()).clamp_min(.25)
   fs=20*t.log10(d[None,None]*self.freq);v=t.tensor(vis[None,None])
   corr=t.cat([(self.function(B,L,G,fs,v,self.mode,b)*self.beta[b][None,:,None,None]).sum(1,keepdim=True)
    for b in (0,1)],1).clamp(-15,15)
   return (P+corr)[0].numpy(),corr[0].numpy()
class Counts:
 def __init__(self):
  lib=ctypes.CDLL(str(OUT/"libc156_counts.so"))
  self.fn=lib.c156_counts
  self.fn.argtypes=[np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,
   np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS"),ctypes.c_int,
   np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")]
  self.fn.restype=None
 def __call__(self,h,tr,tc,pixels):
  h=np.ascontiguousarray(h,np.float32);p=np.ascontiguousarray(pixels,np.int32)
  out=np.empty((len(p),5),np.float64)
  self.fn(h,tr,tc,p,len(p),out)
  if len(p):assert (out[:,4]==1).all()
  return out
def ray_profile(h,tr,tc,r,c):
 from src.c150_reference import registered_profile
 d,z,tx,rx=registered_profile(h,tr,tc,r,c)
 e=z-(tx+(rx-tx)*(d/d[-1]));state=e>1e-10
 return {"distance_m":d,"height_m":z,"direct_ray_m":tx+(rx-tx)*(d/d[-1]),"excess_m":e,
  "crossings":int(np.count_nonzero(state[1:]!=state[:-1])),
  "blocked_segments":int(state[0])+int(np.count_nonzero((~state[:-1])&state[1:])),
  "TX_z_m":tx,"RX_z_m":rx}
