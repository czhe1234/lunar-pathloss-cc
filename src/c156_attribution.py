"""C156 CPU attribution of frozen component fields and common large errors."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,heapq,json,math,os,time,hashlib
import numpy as np
from scipy import ndimage
from src.c156_data import *
from src.c152_los_diagnostic import Rough
RANGE=np.array([0,10,25,50,100,150,200,1e4]);S=256/255
QX,QY=np.meshgrid(np.linspace(1/3,255-1/3,256),np.linspace(1/3,255-1/3,256))
EDGE=np.minimum.reduce(np.indices((256,256)).tolist()+[(255-np.indices((256,256))[0]),(255-np.indices((256,256))[1])])<4
REGIONS=("all","C82_good","C82_tail","common_tail","both_under","both_over","opposite_sign")
TERMS=("base_instead_of_C155","local_correction_half","graph_correction_half","multi_scale_calibration","cache_replay_residual")
def ratio(a,b):return np.divide(a,b,out=np.zeros_like(np.asarray(a,dtype=float)),where=np.asarray(b)>0)
class Score:
 def __init__(self):self.n=np.zeros(2,np.int64);self.sse=np.zeros(2);self.err=np.zeros(2)
 def add(self,p,y,m):
  e=p.astype(float)-y;self.n+=m.sum((1,2));self.sse+=np.where(m,e*e,0).sum((1,2),dtype=float);self.err+=np.where(m,e,0).sum((1,2),dtype=float)
 def result(self):
  return {"count":self.n.tolist(),"RMSE_dB":np.sqrt(ratio(self.sse,self.n)).tolist(),
   "MeanBias_dB":ratio(self.err,self.n).tolist(),"SSE_dB2":self.sse.tolist()}
class Moments:
 def __init__(self):self.a=np.zeros((6,2))
 def add(self,x,y,m):
  for j in range(2):
   xx=x[j][m[j]].astype(float);yy=y[j][m[j]].astype(float)
   self.a[:,j]+=[len(xx),xx.sum(),yy.sum(),xx@xx,xx@yy,yy@yy]
 def result(self):
  n,sx,sy,sxx,sxy,syy=self.a;vx=sxx-sx*sx/n;vy=syy-sy*sy/n;cov=sxy-sx*sy/n
  a=cov/vx;b=(sy-a*sx)/n;ss=np.maximum(0,vy-cov*cov/vx)
  original=sxx-2*sxy+syy
  return {"count":n.astype(int).tolist(),"a":a.tolist(),"b":b.tolist(),"original_RMSE_dB":np.sqrt(original/n).tolist(),
   "oracle_RMSE_dB":np.sqrt(ss/n).tolist(),"MSE_reduction_percent":(100*(original-ss)/original).tolist(),
   "pred_std_over_GT_std":np.sqrt(vx/vy).tolist(),"correlation":(cov/np.sqrt(vx*vy)).tolist(),
   "warning":"GT-assisted global2scalar diagnostic; no predictions changed; not a deployable calibration"}
class Physical:
 def __init__(self):
  self.n=np.zeros(2,np.int64);self.cls=np.zeros((3,2),np.int64);self.floor=np.zeros(2,np.int64);self.near=self.floor.copy();self.far=self.floor.copy();self.edge=self.floor.copy()
  self.range=np.zeros((7,2),np.int64);self.sum={k:np.zeros(2) for k in ["distance_m","nu","slope","terrain_HP_abs_m","GT_HP_abs_dB","Ld_dB","native_LoS_fraction"]}
  self.cnt={k:np.zeros(2,np.int64) for k in self.sum};self.inf=np.zeros(2,np.int64)
  self.nsum=np.zeros(2);self.bsum=np.zeros(2);self.uSum=np.zeros(2);self.ncnt=np.zeros(2,np.int64)
  self.nhist=np.zeros((4,2),np.int64);self.bhist=np.zeros((4,2),np.int64);self.nuvals=[[],[]]
  self.gtfs=np.zeros(2);self.ep82=np.zeros(2);self.ep155=np.zeros(2);self.sse82=np.zeros(2);self.sse155=np.zeros(2)
 def add(self,m,fields,geo=None):
  count=m.sum((1,2));self.n+=count;self.cls+=np.array([(m&x).sum((1,2)) for x in fields["classes"]])
  for attr,flag in [("floor",fields["floor"]),("near",fields["dist"]<10),("far",fields["dist"]>=150),("edge",EDGE)]:
   getattr(self,attr)[:]+=(m&flag).sum((1,2))
  for j in range(7):self.range[j]+=(m&(fields["dist"]>=RANGE[j])&(fields["dist"]<RANGE[j+1])).sum((1,2))
  for k,v in fields["quantities"].items():
   value=np.broadcast_to(v,m.shape);ok=m&np.isfinite(value)
   self.sum[k]+=np.where(ok,value,0).sum((1,2));self.cnt[k]+=ok.sum((1,2))
  self.inf+=(m&np.isinf(fields["nu"])).sum((1,2))
  self.gtfs+=np.where(m,fields["y"]-fields["FSPL"],0).sum((1,2))
  self.ep82+=np.where(m,fields["C82"].astype(float)-fields["y"],0).sum((1,2))
  self.ep155+=np.where(m,fields["C155"].astype(float)-fields["y"],0).sum((1,2))
  self.sse82+=np.where(m,(fields["C82"].astype(float)-fields["y"])**2,0).sum((1,2))
  self.sse155+=np.where(m,(fields["C155"].astype(float)-fields["y"])**2,0).sum((1,2))
  for b in (0,1):
   if m[b].any():self.nuvals[b].append(fields["nu"][b][m[b]&np.isfinite(fields["nu"][b])])
  if geo is not None:
   for b in (0,1):
    ok=m[b]&np.isfinite(geo[0]);n=geo[0][ok];blocks=geo[1][ok];under=geo[2][ok]
    self.ncnt[b]+=len(n);self.nsum[b]+=n.sum();self.bsum[b]+=blocks.sum();self.uSum[b]+=under.sum()
    self.nhist[:,b]+=[(n==0).sum(),((n>0)&(n<4)).sum(),((n>=4)&(n<8)).sum(),(n>=8).sum()]
    self.bhist[:,b]+=[(blocks==0).sum(),(blocks==1).sum(),(blocks==2).sum(),(blocks>=3).sum()]
 def result(self):
  return {"count":self.n.tolist(),"class_order":["Clear","Grazing","NLoS"],"class_pixel_count":self.cls.tolist(),
   "class_percent":(100*ratio(self.cls,self.n)).tolist(),"total_floor_percent":(100*ratio(self.floor,self.n)).tolist(),
   "near_TX_lt10m_percent":(100*ratio(self.near,self.n)).tolist(),"far_ge150m_percent":(100*ratio(self.far,self.n)).tolist(),
   "outer4pixel_percent":(100*ratio(self.edge,self.n)).tolist(),"range_edges_m":RANGE.tolist(),"range_percent":(100*ratio(self.range,self.n)).tolist(),
   "mean":{k:ratio(v,self.cnt[k]).tolist() for k,v in self.sum.items()},"nu_infinite_percent":(100*ratio(self.inf,self.n)).tolist(),
   "nu_finite_percentiles_10_50_90":[np.percentile(np.concatenate(v),[10,50,90]).tolist() if v else None for v in self.nuvals],
   "crossing_count_n_mean":[float(self.nsum[b]/self.ncnt[b]) if self.ncnt[b] else None for b in (0,1)],"blocked_segments_mean":[float(self.bsum[b]/self.ncnt[b]) if self.ncnt[b] else None for b in (0,1)],
   "blocked_path_fraction_mean":[float(self.uSum[b]/self.ncnt[b]) if self.ncnt[b] else None for b in (0,1)],"n_observed_count":self.ncnt.tolist(),
   "n_hist_bins":["0","1to3","4to7","8plus"],"n_hist_percent":(100*ratio(self.nhist,self.ncnt)).tolist(),
   "blocked_segment_bins":["0","1","2","3plus"],"blocked_segment_percent":(100*ratio(self.bhist,self.ncnt)).tolist(),
   "mean_GT_minus_FSPL_dB":ratio(self.gtfs,self.n).tolist(),"mean_C82_minus_GT_dB":ratio(self.ep82,self.n).tolist(),
   "mean_C155_minus_GT_dB":ratio(self.ep155,self.n).tolist(),"C82_RMSE_dB":np.sqrt(ratio(self.sse82,self.n)).tolist(),"C155_RMSE_dB":np.sqrt(ratio(self.sse155,self.n)).tolist(),"C155_SSE_dB2":self.sse155.tolist()}
def fields(m):
 r=m["row"];dist=S*np.hypot(QX-int(r["tx_col"])*255/256,QY-(int(r["tx_row"])+1)*255/256)
 highest=m["highest"][None];nu=m["nu"]
 classes=[(highest<0)&(nu<-.7),(highest<0)&(nu>=-.7)&(nu<0),highest>=0]
 classes=[np.broadcast_to(x,m["valid"].shape) for x in classes]
 assert np.logical_or.reduce(classes)[m["valid"]].all()
 sy,sx=np.gradient(m["h"].astype(float),S,S);slope=np.hypot(sx,sy)
 terrainhp=np.full((256,256),np.nan);terrainhp[1:-1,1:-1]=np.abs(m["h"][1:-1,1:-1].astype(float)-box(m["h"].astype(float)))
 ghp=np.full((2,256,256),np.nan);v=m["y"]-m["FSPL"]
 ghp[:,1:-1,1:-1]=np.where(nine(m["valid"]),np.abs(v[:,1:-1,1:-1]-box(v)),np.nan)
 return {**m,"dist":dist,"classes":classes,"quantities":{"distance_m":dist,"nu":nu,"slope":slope,
  "terrain_HP_abs_m":terrainhp,"GT_HP_abs_dB":ghp,"Ld_dB":m["Ld"],"native_LoS_fraction":m["vis"]}}
def diagnostic_oracle_result(a):
 # a rows: count, SSE, err*HP, HP^2.
 n,ss,eh,hh=a;alpha=-eh/hh
 return {"count":n.astype(int).tolist(),"alpha_optimal":alpha.tolist(),"original_RMSE_dB":np.sqrt(ss/n).tolist(),
  "oracle_RMSE_dB":np.sqrt(np.maximum(0,ss-eh*eh/hh)/n).tolist(),
  "MSE_reduction_percent":(100*eh*eh/hh/ss).tolist(),
  "fixed_smoothing_alpha_minus1_RMSE_dB":np.sqrt(np.maximum(0,ss-2*eh+hh)/n).tolist(),
  "meaning":"positive alpha adds highpass, negative removes highpass; label-assisted statistic, no prediction changes"}
def main():
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU only")
 if (OUT/"result.json").exists():raise FileExistsError("C156 complete; reuse, do not repeat")
 plan=load(OUT/"protocol.json");assert plan["phase"]=="frozen"
 for p,h in plan["source_sha256"].items():assert sha(ROOT/p)==h,p
 for p,v in plan["input_file_identities"].items():assert identity(ROOT/p)==v,p
 start=time.monotonic();data=Data();counter=Counts()
 score={g:{s:Score() for s in STAGES} for g in REGIONS}
 rough={support:{s:Rough() for s in ("C82","C155")} for support in ("center_good_all9valid","all9_good")}
 moments={k:Moments() for k in ("absolute_pathloss","FSPL_residual")}
 allocation={g:np.zeros((len(TERMS),2)) for g in ("all","C82_good")}
 physical={g:Physical() for g in ("all","C82_tail","common_tail","common_Clear","common_Grazing","common_NLoS")}
 shifts=np.zeros((5,5,2));shiftcount=np.zeros(2,np.int64);permap_best=np.zeros(2);permap_hist=np.zeros((5,5,2),np.int64)
 sharpen={k:np.zeros((4,2)) for k in rough}
 closure={"count":0,"sse":np.zeros(2),"max_abs":np.zeros(2),"map_rmse":[]}
 heap=[];sparse=[];hashers={k:hashlib.sha256() for k in ("ordered_base","ordered_local_correction","ordered_graph","ordered_C82","ordered_C155")}
 for j,i in enumerate(data.ids):
  m=data.map(j);f=fields(m);y=m["y"];valid=m["valid"]
  e82=m["C82"].astype(float)-y;e155=m["C155"].astype(float)-y
  good=valid&(np.abs(e82)<=1);tail=valid&(np.abs(e82)>=5);common=tail&(np.abs(e155)>=5)
  masks={"all":valid,"C82_good":good,"C82_tail":tail,"common_tail":common,
   "both_under":common&(e82<0)&(e155<0),"both_over":common&(e82>0)&(e155>0),
   "opposite_sign":common&(e82*e155<0)}
  for key,a in (("ordered_base",m["base"]),("ordered_local_correction",data.corr[data.cl[i]]),
    ("ordered_graph",m["graph"]),("ordered_C82",m["C82"]),("ordered_C155",m["C155"])):
   hashers[key].update(np.ascontiguousarray(a).tobytes())
  for g,mask in masks.items():
   for s in STAGES:score[g][s].add(m[s],y,mask)
  diff=m["calibration_replay"].astype(float)-m["C82"].astype(float)
  closure["count"]+=65536;closure["sse"]+=(diff*diff).sum((1,2));closure["max_abs"]=np.maximum(closure["max_abs"],np.abs(diff).max((1,2)))
  closure["map_rmse"].append(np.sqrt((diff*diff).mean((1,2))))
  B=m["base"].astype(float);L=m["local"].astype(float);G=m["graph"].astype(float);P=m["C155"].astype(float)
  dl=.5*(L-B);dg=.5*(G-B);dc=m["calibration_replay"].astype(float)-m["blend"].astype(float)
  dr=m["C82"].astype(float)-B-dl-dg-dc
  terms=[B-P,dl,dg,dc,dr];total=m["C82"].astype(float)-P
  assert np.max(np.abs(sum(terms)-total))<1e-10
  for g,mask in [("all",valid),("C82_good",good)]:
   for k,d in enumerate(terms):allocation[g][k]+=np.where(mask,-2*e155*d-d*total,0).sum((1,2))
  moments["absolute_pathloss"].add(P,y,good);moments["FSPL_residual"].add(P-m["FSPL"],y-m["FSPL"],good)
  core=good[:,2:-2,2:-2];target=y[:,2:-2,2:-2];shiftcount+=core.sum((1,2));ms=np.zeros((5,5,2))
  for ai,dy in enumerate(range(-2,3)):
   for bi,dx in enumerate(range(-2,3)):
    delta=P[:,2+dy:254+dy,2+dx:254+dx]-target
    ms[ai,bi]=np.where(core,delta*delta,0).sum((1,2))
  shifts+=ms;permap_best+=ms.reshape(25,2).min(0)
  for b in (0,1):
   ai,bi=np.unravel_index(ms[:,:,b].argmin(),(5,5))
   if core[b].any():permap_hist[ai,bi,b]+=1
  valid9=nine(valid);good9=nine(good)
  supports={"center_good_all9valid":good[:,1:-1,1:-1]&valid9,"all9_good":good9}
  yp=y-m["FSPL"];pp=P-m["FSPL"];bp=m["C82"].astype(float)-m["FSPL"]
  hp=pp[:,1:-1,1:-1]-box(pp)
  for name,mask in supports.items():
   rough[name]["C82"].add(yp,bp,mask);rough[name]["C155"].add(yp,pp,mask)
   ec=e155[:,1:-1,1:-1]
   for b in (0,1):
    ee=ec[b][mask[b]];hh=hp[b][mask[b]]
    sharpen[name][:,b]+=[len(ee),ee@ee,ee@hh,hh@hh]
  union=common.any(0);pixels=np.flatnonzero(union).astype(np.int32)
  geoarr=counter(m["h"],int(m["row"]["tx_row"]),int(m["row"]["tx_col"]),pixels)
  ngrid=np.full((256,256),np.nan);bgrid=ngrid.copy();ugrid=ngrid.copy()
  ngrid.ravel()[pixels]=geoarr[:,0];bgrid.ravel()[pixels]=geoarr[:,1];ugrid.ravel()[pixels]=geoarr[:,3]
  if len(pixels):assert np.max(np.abs(geoarr[:,2]-m["highest"].ravel()[pixels]))<1e-5
  sparse.append((j,pixels,geoarr[:,[0,1,3]].copy()))
  physical["all"].add(valid,f)
  physical["C82_tail"].add(tail,f)
  physical["common_tail"].add(common,f,(ngrid,bgrid,ugrid))
  for k,name in enumerate(("Clear","Grazing","NLoS")):
   physical["common_"+name].add(common&f["classes"][k],f,(ngrid,bgrid,ugrid))
  labels,ncomp=ndimage.label(union,structure=np.ones((3,3),int))
  if ncomp:
   sizes=np.bincount(labels.ravel());joint=np.where(common,.5*(e82*e82+e155*e155),0).sum(0)
   sums=np.bincount(labels.ravel(),weights=joint.ravel())
   eligible=np.flatnonzero((sizes>=16)&(np.arange(len(sizes))>0))
   best=eligible[np.argsort(sums[eligible])[-3:]] if len(eligible) else []
   for lab in best:
    item=(float(sums[lab]),-j,-int(lab),np.flatnonzero(labels==lab).astype(np.int32))
    if len(heap)<3:heapq.heappush(heap,item)
    elif item[:3]>heap[0][:3]:heapq.heapreplace(heap,item)
  if (j+1)%20==0 or j+1==len(data.ids):
   prog={"stage":"CPU_attribution","maps":j+1,"total":len(data.ids),"elapsed_seconds":time.monotonic()-start,
    "eta_seconds":(time.monotonic()-start)/(j+1)*(len(data.ids)-j-1),"geometry_rays_so_far":sum(len(s[1]) for s in sparse)}
   atomic(OUT/"progress.json",prog)
   from src import cc_tracking
   cc_tracking.log({"progress/maps":j+1,"progress/total_maps":len(data.ids),"time/elapsed_seconds":prog["elapsed_seconds"],"time/eta_seconds":prog["eta_seconds"]},step=j+1)
   print(json.dumps(prog),flush=True)
 scores={g:{s:v.result() for s,v in table.items()} for g,table in score.items()}
 known=load(ROOT/"runs/c155_unet/selected_verified.json")["panels"]["monitor"]["groups"]
 for group,old in [("all","all"),("C82_good","C82_abs_error_le_1dB"),("C82_tail","C82_abs_error_ge_5dB")]:
  for model,key in [("C155","RMSE_dB"),("C82","C82_RMSE_dB")]:
   assert np.allclose(scores[group][model]["RMSE_dB"],known[old][key],rtol=0,atol=1e-9),(group,model)
  assert scores[group]["C155"]["count"]==known[old]["valid_pixels"]
 alloc={}
 for g,table in allocation.items():
  total_gain=np.asarray(scores[g]["C155"]["SSE_dB2"])-np.asarray(scores[g]["C82"]["SSE_dB2"])
  assert np.allclose(table.sum(0),total_gain,atol=1e-5,rtol=1e-12)
  alloc[g]={"terms":list(TERMS),"SSE_gain_dB2":table.tolist(),"MSE_gain_dB2":(table/score[g]["C155"].n).tolist(),
   "percent_of_C155_to_C82_SSE_gain":(100*table/total_gain).tolist(),"sum_matches_actual_gain":True,
   "meaning":"Fixed-output additive Shapley squared-error allocation; no causal retraining claim"}
 shiftres={"count":shiftcount.tolist(),"offsets_dy_dx":[-2,-1,0,1,2],"SSE_grid":shifts.tolist(),
  "identity_RMSE_dB":np.sqrt(shifts[2,2]/shiftcount).tolist(),"best_global":[],"per_image_oracle_RMSE_dB":np.sqrt(permap_best/shiftcount).tolist(),
  "per_image_optimal_shift_histogram":permap_hist.tolist()}
 for b in (0,1):
  ai,bi=np.unravel_index(shifts[:,:,b].argmin(),(5,5))
  shiftres["best_global"].append({"dy":ai-2,"dx":bi-2,"RMSE_dB":float(np.sqrt(shifts[ai,bi,b]/shiftcount[b])),
   "MSE_reduction_percent":float(100*(shifts[2,2,b]-shifts[ai,bi,b])/shifts[2,2,b])})
 physical_result={k:v.result() for k,v in physical.items()}
 # Keep sparse new CPU counts for audit/reuse, never rewrite original caches.
 offsets=[0];flat=[];geos=[]
 for j,px,g in sparse:flat.append(px);geos.append(g);offsets.append(offsets[-1]+len(px))
 np.savez_compressed(OUT/"crossing_counts_sparse.npz",ordered_ids=np.asarray(data.ids),offsets=np.asarray(offsets),
  pixels=np.concatenate(flat),counts=np.concatenate(geos),columns=np.asarray(["n","blocked_segments","blocked_fraction"]))
 cases=[]
 for rank,item in enumerate(sorted(heap,key=lambda x:x[:3],reverse=True),1):
  joint_sse,negj,neglab,px=item;j=-negj;m=data.map(j,components=False);y=m["y"]
  mask=np.zeros((256,256),bool);mask.ravel()[px]=True
  e82=m["C82"].astype(float)-y;e155=m["C155"].astype(float)-y
  common=m["valid"]&(np.abs(e82)>=5)&(np.abs(e155)>=5)&mask[None]
  rows,cols=np.where(mask);geo=sparse[j];lookup={int(p):g for p,g in zip(geo[1],geo[2])}
  driving=int(np.argmax(np.where(common,.5*(e82*e82+e155*e155),0).sum((1,2))))
  perband=[];representatives=[]
  for b in (0,1):
   mm=common[b];n=int(mm.sum());models={"GT":y[b],"C82":m["C82"][b],"C155":m["C155"][b],"FSPL_Deygout":m["FSPL_Deygout"][b],"FSPL":m["FSPL"][b]}
   values={k:{"mean_dB":float(v[mm].mean()) if n else None,"RMSE_vs_GT_dB":float(np.sqrt(np.mean((v[mm]-y[b][mm])**2))) if n else None,
    "median_difference_vs_GT_dB":float(np.median(v[mm]-y[b][mm])) if n else None,
    "difference_q10_q90_dB":np.quantile(v[mm]-y[b][mm],[.1,.9]).tolist() if n else None} for k,v in models.items()}
   cls=[mm&(m["highest"]<0)&(m["nu"][b]<-.7),mm&(m["highest"]<0)&(m["nu"][b]>=-.7)&(m["nu"][b]<0),mm&(m["highest"]>=0)]
   pts=np.flatnonzero(mm);ng=np.array([lookup[int(p)][0] for p in pts]);bg=np.array([lookup[int(p)][1] for p in pts])
   perband.append({"band":BANDS[b],"count":n,"models":values,"class_percent":[100*x.sum()/n if n else 0 for x in cls],
    "mean_n":float(ng.mean()) if n else None,"mean_blocked_segments":float(bg.mean()) if n else None,
    "mean_nu":float(m["nu"][b][mm&np.isfinite(m["nu"][b])].mean()) if (mm&np.isfinite(m["nu"][b])).any() else None,"nu_infinite_count":int((mm&np.isinf(m["nu"][b])).sum()),"floor_percent":100*float(m["floor"][b][mm].mean()) if n else None})
   if n:
    avgerr=.5*(np.abs(e82[b].ravel()[pts])+np.abs(e155[b].ravel()[pts]))
    for quantile in (.25,.5,.75):
     pix=int(pts[np.argmin(np.abs(avgerr-np.quantile(avgerr,quantile)))]);rr,cc=divmod(pix,256)
     representatives.append({"band":BANDS[b],"error_quantile":quantile,"row":rr,"col":cc,
      **{k:float(v[rr,cc]) for k,v in models.items()},"n":int(lookup[pix][0]),"blocked_segments":int(lookup[pix][1]),
      "nu_max":float(m["nu"][b,rr,cc]),"highest_m":float(m["highest"][rr,cc])})
  anchor=next(v for v in representatives if v["band"]==BANDS[driving] and v["error_quantile"]==.5)
  profile=ray_profile(m["h"],int(m["row"]["tx_row"]),int(m["row"]["tx_col"]),anchor["row"],anchor["col"])
  assert profile["crossings"]==anchor["n"] and profile["blocked_segments"]==anchor["blocked_segments"]
  from src.c150_reference import reference_array
  reference=reference_array(m["h"],int(m["row"]["tx_row"]),int(m["row"]["tx_col"]),anchor["row"],anchor["col"])
  cached=np.ravel([m["Ld"][0,anchor["row"],anchor["col"]],m["Ld"][1,anchor["row"],anchor["col"]]])
  assert np.allclose(reference[[3,8]],cached,atol=4e-4,rtol=1e-5)
  file=OUT/f"case_{rank}.npz"
  np.savez_compressed(file,mask=mask,common_tail=common,GT=y,valid=m["valid"],C82=m["C82"],C155=m["C155"],FSPL=m["FSPL"],Deygout=m["FSPL_Deygout"],
   height=m["h"],highest=m["highest"],nu=m["nu"],floor=m["floor"],**{"profile_"+k:v for k,v in profile.items()})
  case={"rank":rank,"sample_id":m["sample_id"],"map_id":m["map_id"],"terrain_id":m["terrain_id"],"driving_band":BANDS[driving],
   "joint_SSE_dB2":joint_sse,"union_spatial_pixels":len(px),"bbox_rows_cols":[int(rows.min()),int(rows.max()),int(cols.min()),int(cols.max())],
   "per_band":perband,"representative_pixels":representatives,"profile_anchor":anchor,
   "profile_crossings":profile["crossings"],"profile_blocked_segments":profile["blocked_segments"],
   "near_TX_lt10m_percent":[100*float((fields(m)["dist"][common[b]]<10).mean()) if common[b].any() else None for b in (0,1)],
   "far_ge150m_percent":[100*float((fields(m)["dist"][common[b]]>=150).mean()) if common[b].any() else None for b in (0,1)],
   "outer4pixel_percent":[100*float(EDGE[common[b]].mean()) if common[b].any() else None for b in (0,1)],
   "data":str(file.relative_to(ROOT))}
  cases.append(case)
 import torch
 assert not torch.cuda.is_initialized()
 assert all(identity(ROOT/p)==v for p,v in plan["input_file_identities"].items())
 assert all(sha(ROOT/p)==h for p,h in plan["source_sha256"].items())
 result={"status":"complete","campaign":"C156","protocol_sha256":sha(OUT/"protocol.json"),"scope":plan["scope"],"scores":scores,
  "C155_excess_MSE_attribution":alloc,"C82_good_diagnostics":{"integer_shift":shiftres,
   "affine_oracles":{k:v.result() for k,v in moments.items()},"roughness":{k:{s:v.result() for s,v in tab.items()} for k,tab in rough.items()},
   "highpass_oracle":{k:diagnostic_oracle_result(v) for k,v in sharpen.items()}},
  "physical":physical_result,"top3_regions":cases,
  "calibration_replay_residual":{"RMSE_dB":np.sqrt(closure["sse"]/closure["count"]).tolist(),"max_abs_dB":closure["max_abs"].tolist(),
   "map_RMSE_percentiles_50_90_99":np.percentile(np.asarray(closure["map_rmse"]),[50,90,99],axis=0).tolist()},
  "source_subset_sha256":{k:v.hexdigest() for k,v in hashers.items()},"source_files_and_identities_unchanged":True,
  "CPU_geometry_rays":offsets[-1],"elapsed_seconds":time.monotonic()-start,"training":False,"neural_inference":False,"CUDA_initialized":False,
  "limits":plan["limits"]+["Scalar/sharpen/shift diagnostic oracles use labels in the same region and are not new validation scores.",
   "A few top regions and line-profile counts cannot by themselves uniquely identify simulation interference or raster publication artifacts."]}
 atomic(OUT/"result.json",result);atomic(OUT/"progress.json",{"stage":"statistics_complete","elapsed_seconds":result["elapsed_seconds"],"maps":628})
 print(json.dumps({"status":"statistics_complete","seconds":result["elapsed_seconds"],"common_tail":physical_result["common_tail"]["count"],
  "replay_RMSE":result["calibration_replay_residual"]["RMSE_dB"]}),flush=True)
def entry():
 if (OUT/"result.json").exists():
  print(json.dumps({"status":"existing_statistics_reused_no_rerun"}));return
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU only")
 from src import cc_tracking
 plan=load(OUT/"protocol.json")
 swan=cc_tracking.init(name="C156 CPU component attribution",campaign="C156",
  config={"run_kind":"analysis.attribution","_cc_dashboard":{"kind":"analysis"},"protocol_sha256":sha(OUT/"protocol.json"),
   "scope":plan["scope"],"training":False,"neural_inference":False,"GPU_used":False},
  description="Frozen C82 component and C155 selected52 attribution, common-tail geometry; CPU only.",tags=["CPU","attribution","C82","C155"])
 atomic(OUT/"tracking.json",{"run_id":swan.id,"dir":str(swan.dir),"mode":"local"})
 atomic(OUT/"status.json",{"status":"running","phase":"CPU_statistics","pid":os.getpid(),"training":False,"neural_inference":False})
 try:
  main()
  d=load(OUT/"result.json");metrics={}
  for scope in ("all","C82_good"):
   for stage in STAGES:
    for b,band in enumerate(BANDS):metrics[f"{scope}/{stage}/RMSE_{band}_dB"]=d["scores"][scope][stage]["RMSE_dB"][b]
  for b,band in enumerate(BANDS):
   metrics[f"common_tail/{band}/count"]=d["physical"]["common_tail"]["count"][b]
   metrics[f"common_tail/{band}/mean_n"]=d["physical"]["common_tail"]["crossing_count_n_mean"][b]
  cc_tracking.log(metrics,step=629)
  from src.c156_report import run as report
  report()
  from src.c156_figures import render
  render()
  atomic(OUT/"status.json",{"status":"awaiting_figure_inspection","phase":"tables_and_figures_rendered","pid":os.getpid(),"training":False,"neural_inference":False})
 except BaseException as exc:
  atomic(OUT/"status.json",{"status":"failed","phase":"CPU_analysis","error":repr(exc),"pid":os.getpid(),"training":False})
  raise
 finally:cc_tracking.finish()
if __name__=="__main__":entry()

