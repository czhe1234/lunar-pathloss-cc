"""C156 CPU preflight and protocol; no trained network is executed."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ctypes,json,os,subprocess,time
import numpy as np
from src.c156_data import *
def prepare():
 if (OUT/"protocol.json").exists():raise FileExistsError("C156 protocol exists")
 d=Data();model=load(ROOT/"runs/c82_selected/model.json");assert model["graph_weight"]==.5 and model["gate_weights"] is None
 inputs=["runs/c54_combo_dev/bf16_compiled/all_shard0of1/pl415.npy","runs/c54_combo_dev/bf16_compiled/all_shard0of1/pl58.npy",
  "runs/c54_combo_dev/bf16_compiled/all_shard0of1/indices.json","runs/c72_observed/full_internal/correction.npy","runs/c72_observed/full_internal/indices.json",
  "runs/c73_wide_real/full_internal/prediction.npy","runs/c73_wide_real/full_internal/indices.json","audit/c88_mesh/base.npy","audit/c88_mesh/base_index.json",
  "runs/c82_selected/spectrum.json","runs/c82_selected/model.json","runs/c82_selected/weights/post.pt",
  "runs/c155_unet/selected.pt","runs/c155_unet/last.pt","runs/c155_unet/selected_monitor_prediction.npy","runs/c155_unet/selected_monitor_prediction.json",
  "runs/c155_unet/selected_export.json","runs/c155_unet/selected_verified.json","runs/c155_unet/history.json",
  "audit/c149_error_audit/profiles.npy","audit/c149_error_audit/protocol.json","features/c150_diffraction/part0of1/features.npy",
  "features/c150_diffraction/part0of1/indices.json","features/c150_diffraction/part0of1/complete.json",
  "features/c73_compact/part0of1/features.npy","features/c73_dev_extra_compact/part0of1/features.npy"]
 sources=["src/c156_data.py","src/c156_counts.cpp","src/c156_preflight.py","src/c82_inference.py","src/c82_blend.py",
  "src/c74_inference.py","src/c72_native.py","src/c73_compact.py","src/c70_geometry.cu","src/c149_profiles.cu",
  "src/c150_reference.py","src/c152_los_diagnostic.py","src/cc_tracking.py"]
 plan={"campaign":"C156","phase":"preflight","created_utc":datetime.now(timezone.utc).isoformat(),
  "parent":"C155 selected52 versus frozen C82","scope":{"monitor_maps":628,"terrains":128,"frequency_hz":[415e6,5.775e9]},
  "ordered_ids":d.ids,"coverage":{"base":628,"local":628,"graph":628,"C82":628,"C155":628},
  "definitions":{"valid":"Original per-frequency bitpacked masks; never shifted or used as model input",
   "error":"pred-GT","C82_good":"valid & abs(C82-GT)<=1dB","C82_tail":"valid & abs(C82-GT)>=5dB",
   "common_tail":"valid & abs(C82-GT)>=5dB & abs(C155-GT)>=5dB, per frequency",
   "common_same_sign":"common_tail & (C82-GT)*(C155-GT)>0",
   "geometry_classes":"Clear:highest<0 & nu<-.7; Grazing:highest<0 & -.7<=nu<0; NLoS:highest>=0",
   "nu":"Existing C149 nu(lambda1m)/sqrt(lambda_m); report finite mean plus infinite fraction",
   "n":"NEW CPU exact registered-point terrain/straight-line crossing count; not reflection depth or inverse averaged log count",
   "blocked_segments":"Contiguous positive terrain-minus-ray profile intervals; distinguish one obstruction from multiple intervals",
   "floor":"c73 Float16 total linear-mean channels41/42 <=-3. Q=P*d_face² encoded; not exact absolute P clamp",
   "distance":"Registered horizontal TX-RX distance in meters; near/far here are geometric range categories, not electromagnetic field regimes",
   "near_TX_m":10,"far_range_m":150,"range_edges_m":[0,10,25,50,100,150,200,10000],
   "map_edge_width_pixels":4,"slope":"raw terrain gradient divided by256/255, m/m"},
  "components":{"base":"Frozen C54 per-band Float32","local":"base + cached C72 correction(Float16 castFloat32)",
   "graph":"Cached C73 standalone field Float32","blend":".5*local+.5*graph, parallel experts",
   "calibration_replay":"Exact existing c82_blend.field_features AST with frozen117coeff/band, CPU arithmetic only",
   "replay_graph":"2*blend-local, as c82_inference",
   "C82":"Authoritative frozen audit/c88_mesh/base.npy",
   "replay_residual":"C82 - calibration_replay; reported separately so backend/cache error is not assigned to calibration",
   "component_attribution":"Exact squared-error gain allocation for frozen additive terms, interactions shared symmetrically; not a retrained causal ablation"},
  "C82good_diagnostics":{"shift_pixels":[-2,-1,0,1,2],"shift_support":"Fixed2pixel interior target C82_good, all candidates same support",
   "affine_oracles":["a*Pred+b","FSPL+a*(Pred-FSPL)+b"],
   "roughness_support":["center_good_and_all9_valid","all9_good_valid"],
   "sharpen_oracle":"Pred + alpha*HP3x3(Pred-FSPL); signed alpha; exact least-square statistic only",
   "oracle_limit":"GT-assisted in-sample diagnostics. Never modify predictions or weights, never use as a new score."},
  "top_regions":{"mask":"8-connected components of per-map spatial union of per-band common_tail",
   "minimum_spatial_pixels":16,"ranking":"sum over both bands/common_tail of (errC82²+errC155²)/2",
   "count":3,"ties":"map order then component index","no_manual_case_selection":True,
   "regional_scores":"Per-band common-tail pixels inside the chosen spatial component",
   "representative":"Pixels nearest25/50/75percentile average absolute errors; median case ray profile"},
  "resource":{"CPU_only":True,"CUDA_VISIBLE_DEVICES":"","Torch_fixed_arithmetic_threads":4,"CPU_geometry_OpenMP_max_threads":8,
   "training":False,"neural_inference":False,"GPU_inference":False,"propagation_power_solver":False,
   "new_geometry":"Counts only for common-tail union; existing highest/nu/FSPL/Ld reused"},
  "input_file_identities":{n:identity(ROOT/n) for n in inputs},"source_sha256":{n:sha(ROOT/n) for n in sources},
  "parent_verified_sha256":sha(ROOT/"audit/c155_unet/comparison_summary.json"),
  "frozen_full_model_artifacts_unchanged":True,"SwanLab":{"mode":"local","project":"Lunar_CC","port":16007},
  "limits":["C82_good is conditioned on the final C82; all-valid stage scores are also required.",
   "Profiles and floor descriptions concern the current reconstructed scene/cache. They do not uniquely prove the unknown label-generation mechanism.",
   "CPU geometry n is a new statistic, not rerunning old propagation experiments."]}
 atomic(OUT/"protocol.json",plan);print(json.dumps({"status":"prepared","coverage":plan["coverage"]}),flush=True)
def preflight():
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU only")
 p=load(OUT/"protocol.json");d=Data();count=Counts();checks=[]
 xx=np.arange(256,dtype=np.float32)[None,:];flat=np.zeros((256,256),np.float32)
 cases=[("flat",flat,0),("one_ridge",np.broadcast_to(np.maximum(0,12-np.abs(xx-128)),(256,256)).copy(),2),
  ("two_ridges",np.broadcast_to(np.maximum(np.maximum(0,8-np.abs(xx-115)),np.maximum(0,11-np.abs(xx-160))),(256,256)).copy(),4)]
 for name,h,expected in cases:
  got=count(h,100,80,[100*256+220])[0];ref=ray_profile(h,100,80,100,220)
  assert got[0]==expected==ref["crossings"],(name,got,ref["crossings"])
  assert got[1]==ref["blocked_segments"]
  checks.append({"fixture":name,"n":int(got[0]),"blocked_segments":int(got[1])})
 closure=[]
 for j in (0,100,300,500,627):
  m=d.map(j);r=m["row"];pixels=np.array([17*256+21,123*256+200,235*256+58],np.int32)
  c=count(m["h"],int(r["tx_row"]),int(r["tx_col"]),pixels)
  for k,pix in enumerate(pixels):
   rr,cc=divmod(int(pix),256);ref=ray_profile(m["h"],int(r["tx_row"]),int(r["tx_col"]),rr,cc)
   assert c[k,0]==ref["crossings"] and c[k,1]==ref["blocked_segments"],(j,pix,c[k],ref["crossings"])
   assert abs(c[k,2]-float(m["highest"][rr,cc]))<5e-6
  dif=m["calibration_replay"].astype(float)-m["C82"].astype(float)
  closure.append({"sample_id":m["sample_id"],"RMSE_dB":np.sqrt((dif*dif).mean((1,2))).tolist(),"max_abs_dB":np.abs(dif).max((1,2)).tolist()})
 import torch
 assert not torch.cuda.is_initialized()
 out={"status":"CPU_preflight_passed","geometry_checks":checks,"real_maps_checked":5,"real_pixels_checked":15,
  "calibration_replay_closure":closure,"Torch_CUDA_initialized":False,"training_run":False,"neural_inference":False}
 atomic(OUT/"preflight.json",out);print(json.dumps(out),flush=True)
if __name__=="__main__":
 import sys
 if "--prepare" in sys.argv:prepare()
 elif "--preflight" in sys.argv:preflight()
