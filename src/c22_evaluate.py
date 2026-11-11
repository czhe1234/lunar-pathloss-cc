"""Full-pixel evaluation for C22/C23 checkpoints; no parameter updates."""
from cc_core import ROOT,atomic_json,check_source_unchanged
import argparse,json,hashlib,time
import numpy as np
import torch
from c21_train_r1 import setup,model
from c21_pair_r1 import full_dev
from c21_data import Data
from c23_model import Model as ConditionalModel
def main(args):
 gpu=setup(20261129);check_source_unchanged();run=ROOT/"runs"/args.run
 ckpath=run/args.checkpoint;ck=torch.load(ckpath,map_location="cpu",weights_only=False)
 if args.condition=="none":net=model(False)
 else:
  n=np.load(ROOT/"audit/c21_execution_20261101/normalization.npz")
  net=ConditionalModel(torch.from_numpy(n["mean"]),torch.from_numpy(n["std"]),conditional=args.condition=="true").cuda()
 net.load_state_dict(ck["model"]);step=ck["step"];del ck
 protocol=json.loads((ROOT/"audit/c22_scale_diagnosis_20261101/protocol.json").read_text())
 ids=protocol["fit_panel"] if args.part=="fit" else Data().ids["dev"]
 dest=run/args.dest;assert not dest.exists()
 metrics=full_dev(net,ids,dest,scope=f"complete valid pixels of fixed128 FIT" if args.part=="fit" else "complete terrain-disjoint DEV2562; official formula, not officialVAL")
 atomic_json(dest/"provenance.json",{"checkpoint_sha256":hashlib.sha256(ckpath.read_bytes()).hexdigest(),"step":step,"gpu":gpu,"condition":args.condition,"parameters_updated":False})
 print(json.dumps({"event":"complete_evaluation","directory":str(dest),"metrics":metrics}),flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);ap.add_argument("--checkpoint",default="best.pt")
 ap.add_argument("--part",choices=["fit","dev"],default="dev");ap.add_argument("--dest",required=True)
 ap.add_argument("--condition",choices=["none","false","true"],default="none")
 main(ap.parse_args())
