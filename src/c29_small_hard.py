"""Reuse the verified original four-map trainer with only FIT identities changed."""
from pathlib import Path
import json,argparse,hashlib
import c21_train_r1 as original
from cc_core import ROOT,atomic_json
a=ROOT/"audit/c29_fit_regime_20261101"
cfg=json.loads((ROOT/"runs/c21r1_small_a_s20261119/config.json").read_text())
for name,sha in cfg["source_hashes"].items():
 assert hashlib.sha256((ROOT/"src"/name).read_bytes()).hexdigest()==sha
original.A=a/"hard_small"
args=argparse.Namespace(mode="small",name="c29_hard_small_original_regime",geometry=0,steps=6500,queries=2048,chunk=512,eval_chunk=1024,eval_every=250,seed=20261119,lr=.001,resume="")
atomic_json(a/"hard_small_provenance.json",{"reference":cfg,"executed_wrapper_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"data_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in original.A.glob("*.npy")},"only_intervention":"FIT identities","reuse_control_sources_equal":True})
original.small(args)
