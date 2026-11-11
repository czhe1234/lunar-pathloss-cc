"""Post-C22: compare single and averaged weights, then full DEV for the chosen option."""
from cc_core import ROOT,atomic_json,check_source_unchanged
import argparse,json,time,hashlib,shutil
import numpy as np
import torch
from c21_train_r1 import setup,model
from c22_train import monitor
from c21_pair_r1 import full_dev
from c21_data import Data
A=ROOT/"audit/c22_scale_diagnosis_20261101"
def main(args):
 gpu=setup(20261129);check_source_unchanged()
 protocol=json.loads((A/"protocol.json").read_text());pixels=np.load(ROOT/"runs/c21r1_pair_s20261123/monitor_pixels.npy")
 net=model(False);rows=[]
 for suffix in ["","_swa"]:
  run=ROOT/"runs"/(args.run+suffix);ckpath=run/"best.pt";ck=torch.load(ckpath,map_location="cpu",weights_only=False)
  net.load_state_dict(ck["model"]);step=ck["step"];del ck
  row={"run":args.run+suffix,"step":step,"checkpoint_sha256":hashlib.sha256(ckpath.read_bytes()).hexdigest(),
       "FIT":monitor(net,protocol["fit_panel"],pixels),"DEV":monitor(net,protocol["dev_panel"],pixels)}
  atomic_json(run/"post_monitor.json",row);rows.append(row);print(json.dumps(row),flush=True)
 chosen=min(rows,key=lambda r:r["DEV"]["score"]);run=ROOT/"runs"/chosen["run"]
 ck=torch.load(run/"best.pt",map_location="cpu",weights_only=False);net.load_state_dict(ck["model"]);del ck
 metrics=full_dev(net,Data().ids["dev"],run/"dev_complete",scope="complete terrain-disjoint DEV2562; official formula, not officialVAL")
 out={"gpu":gpu,"options":rows,"chosen":chosen["run"],"complete_DEV":metrics,"selection":"smaller fixedDEV128 mean per-band RMSE; all valid pixels then evaluated onDEV2562"}
 atomic_json(A/(args.run+"_post.json"),out);print(json.dumps(out),flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);main(ap.parse_args())
