"""Freeze the teacher-free matched pair and validate its input routes on CPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import sys, json, math, hashlib, ast, csv
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.c146_data import AUDIT,RUN,Reader,create,model_inputs,digest,atomic,sha,ARMS
torch.set_num_threads(2)
AUDIT.mkdir(exist_ok=True);RUN.mkdir(exist_ok=True)
parent=json.loads((ROOT/"audit/c145_capacity/protocol.json").read_text())
confirm=json.loads((ROOT/"audit/c99_axial/protocol.json").read_text())["confirmation"]
panels={**parent["panels"],"confirmation":confirm}
reader=Reader(supervision=False)
terrains={name:set(reader.rows[i]["terrain_id"] for i in ids) for name,ids in panels.items()}
assert all(not (terrains[a]&terrains[b]) for a,b in (("fit","monitor"),("fit","confirmation"),("monitor","confirmation")))
assert all(i in reader.lookup for ids in panels.values() for i in ids)
net=create(146041); initial=digest(net)
assert initial==digest(create(146041))
# Intervene with nonzero output head so the test cannot pass just because initialization predicts FSPL.
torch.manual_seed(146099)
torch.nn.init.normal_(net.head.weight,std=.01)
net.eval()
basic,physical,fs=reader.inputs(panels["fit"][:1],"cpu")
x=model_inputs(basic,physical,"without_solver")
other=model_inputs(basic,physical+3.17,"without_solver")
assert torch.equal(x,other) and torch.count_nonzero(x[:,11:])==0
with torch.no_grad():
 a=net(x);b=net(other);c=net(model_inputs(basic,physical,"with_solver"))
assert torch.equal(a,b) and not torch.equal(a,c)
for file in ("src/c146_data.py","src/c146_train.py"):
 text=(ROOT/file).read_text()
 tree=ast.parse(text)
 for node in ast.walk(tree):
  if isinstance(node,ast.ImportFrom):
   assert node.module not in ("src.c105_context","src.c99_data","src.c145_train")
cfg={
 "created":datetime.now(timezone.utc).isoformat(),"campaign_id":"C146_solver_feature_ablation",
 "question":"At fixed network/split/update budget, do the existing 45 fast-solver channels help or harm new-terrain and fixed-tail accuracy, with all C82 learning/inference paths removed?",
 "panels":panels,"panel_counts":{k:{"maps":len(v),"terrains":len(terrains[k])} for k,v in panels.items()},
 "arms":list(ARMS),"architecture":"C123 Field(native), 58 channels, same 15,214,786 parameters",
 "initial_seed":146041,"initial_state_sha256":initial,"data_seed":146073,
 "epochs":128,"batch":8,"updates_per_arm":math.ceil(len(panels["fit"])/8)*128,
 "learning_rate":.0003,"minimum_learning_rate":.000003,"warmup_updates":200,
 "weight_decay":.01,"clip_grad_norm":2.0,"evaluate_every_epochs":4,"fit_evaluate_every_epochs":8,
 "optimizer":"AdamW, independent equal optimizers; fixed cosine schedule, no adaptive per-arm LR or early stop",
 "shared_input":"11 raw H/TX geometry channels; no solver-derived power prior; common analytic FSPL from distance/frequency",
 "intervention":"channels11:56 are existing45 solver-derived values vs zeros; channels56:58 zero in BOTH; mask before all branches including TX FiLM",
 "output":"FSPL(H,TX)+network output; target raw TRAIN pathloss; no C82 predictions/weights/distillation/residual targets",
 "loss":"Full native valid pixels; original terrain map-inclusion weights; perband weighted SSE divided by exact fullFIT weighted valid count / batches, then averagebands/25",
 "comparison":"Paired arms alternate each batch on same GPU, exact same maps/order/targets/weights/update count and initial weights",
 "selection":"Each arm minimum mean monitor RMSE among epoch0 and every4 epochs; BOTH final128 and selected checkpoints reported. Monitor cannot stop training or change budget.",
 "evaluation":"Independent CPU FP64 on full FIT, monitor and confirmation; all, fixed |Y-C82|>=5dB, under/over tails, non-tail, C82good<=1dB; candidate own >5dB fraction and SSE share; new errors on C82good",
 "C82_role":"Evaluation-only frozen reference and region definition in separate verifier AFTER weights/predictions frozen; never opened by training/prediction Reader",
 "confirmation":"Original484terrain1934maps, frozen choices before scoring; historically used development data, not fresh blind or official VAL",
 "primary_endpoint":"Selected checkpoints on confirmation full valid pixels; matched final128 as required equal-update endpoint; tail gains and ordinary-region harm reported separately",
 "effect_interpretation":"Delta=RMSE_without-RMSE_with; positive helps. Terrain paired bootstrap95CI. Practical materiality0.05dB perband and CI excluding0; mixed bands/regions reported mixed, no universal information-limit claim",
 "repetition":"One fixed paired seed; no sweeping or repeats of old experiments; no broad seed-robustness claim",
 "resource_authorization":"User: 不需要只用空闲卡. This C146 may share approved physical1/3/4 only; never terminate other tasks. Prefer1 H100 for paired identical hardware.",
 "official_VAL_used":False,
}
sources=["src/c146_data.py","src/c146_train.py","src/c146_verify.py","scripts/c146_control.py",
         "src/c123_lossless.py","src/c99_axial.py","src/c117_factorized.py","src/c3_geometry.py"]
cfg["source_sha256"]={p:sha(ROOT/p) for p in sources}
if (AUDIT/"protocol.json").exists():
 old=json.loads((AUDIT/"protocol.json").read_text())
 for key in cfg:
  if key!="created" and old.get(key)!=cfg[key]:raise ValueError("Protocol differs: "+key)
 cfg=old
else: atomic(AUDIT/"protocol.json",cfg)
atomic(AUDIT/"preflight.json",{
 "status":"passed_CPU","parameters":sum(p.numel() for p in net.parameters()),
 "initial_state_sha256":initial,"identical_initialization":True,"terrain_disjoint":True,
 "all_solver_cache_maps_present":True,"without_solver_invariant_to_arbitrary_solver_change":True,
 "with_solver_changes_output_nonzero_head":float((a-c).abs().max()),
 "old_C82_channels_zero_both":True,"C82_training_reader_absent":True,
 "nonzero_head_only_for_preflight":"discarded; training recreates frozen random initial state",
 "panels":cfg["panel_counts"],"GPU_initialized":torch.cuda.is_initialized()
})
assert not torch.cuda.is_initialized()
print(json.dumps({"preflight":"passed_CPU","panels":cfg["panel_counts"],"updates_per_arm":cfg["updates_per_arm"],"parameters":sum(p.numel() for p in net.parameters())}),flush=True)
