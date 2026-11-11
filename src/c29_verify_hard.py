"""Independent reload of the hard-four checkpoint, exporting every predicted pixel."""
import json,hashlib
import torch
from cc_core import ROOT,atomic_json,check_source_unchanged
from c21_train_r1 import setup,model
from c21_pair_r1 import full_dev
A=ROOT/"audit/c29_fit_regime_20261101"
gpu=setup(20261203);check_source_unchanged()
run=ROOT/"runs/c29_hard_small_original_regime";p=json.loads((A/"protocol.json").read_text())
ckpath=run/"best.pt";ck=torch.load(ckpath,map_location="cpu",weights_only=False)
net=model(False);net.load_state_dict(ck["model"])
res=full_dev(net,p["hard_ids"],run/"fit_complete",scope="C29 four selected hard FIT maps, original small-fit regime; not officialVAL")
ref=json.loads((run/"complete.json").read_text())
delta=max(abs(x-y) for x,y in zip(res["rmse"],ref["rmse"]))
assert delta<.001,delta
atomic_json(A/"hard_reload_check.json",{"gpu":gpu,"checkpoint_sha256":hashlib.sha256(ckpath.read_bytes()).hexdigest(),"step":ck["step"],"rmse":res["rmse"],"difference_vs_training_evaluation_db_max":delta,"no_parameter_update":True})
print(json.dumps(res))
