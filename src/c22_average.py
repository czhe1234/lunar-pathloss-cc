"""Prospective late-checkpoint averaging; no training or GPU work."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import time,json,hashlib
import torch
R=Path(__file__).resolve().parents[1];A=R/"audit/c22_scale_diagnosis_20261101"
names=["c22_full_low","c22_full_rewarm"];steps=[4000,5000,6000]
def main():
 torch.set_num_threads(4);saved={n:{} for n in names};start=time.time()
 while any(len(saved[n])<3 for n in names):
  if time.time()-start>7200:raise RuntimeError("averaging capture timeout")
  for name in names:
   run=R/"runs"/name;path=run/"last.pt";meta=run/"latest_eval.json"
   if not path.exists() or not meta.exists():continue
   step=json.loads(meta.read_text())["step"]
   if step not in steps or str(step) in saved[name] or time.time()-path.stat().st_mtime<2:continue
   try:ck=torch.load(path,map_location="cpu",weights_only=False)
   except (OSError,RuntimeError,EOFError):continue
   if ck["step"]!=step:continue
   outdir=A/"late_checkpoints"/name;outdir.mkdir(parents=True,exist_ok=True)
   out=outdir/f"{step}.pt"
   # Preserve only the model and provenance. No incompatible averaged optimizer.
   torch.save({"model":ck["model"],"step":step,"config":ck["config"],"metrics":ck["metrics"]},out)
   saved[name][str(step)]={"path":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest()}
   print(name,step,flush=True);del ck
  (A/"late_checkpoint_capture.json").write_text(json.dumps(saved,indent=2)+"\n")
  time.sleep(5)
 for name in names:
  checkpoints=[torch.load(saved[name][str(s)]["path"],map_location="cpu",weights_only=False) for s in steps]
  avg={}
  for key,first in checkpoints[0]["model"].items():
   vals=[c["model"][key] for c in checkpoints]
   if first.is_floating_point():avg[key]=torch.stack(vals).mean(0).to(first.dtype)
   else:
    assert all(torch.equal(v,first) for v in vals);avg[key]=first.clone()
  dest=R/"runs"/(name+"_swa");dest.mkdir(exist_ok=False)
  ck={"model":avg,"step":6000,"config":checkpoints[-1]["config"],
      "averaged_continuation_steps":steps,"parent_step":18000}
  torch.save(ck,dest/"best.pt")
  (dest/"config.json").write_text(json.dumps({"sources":saved[name],"averaged_steps":steps,"parameters_updated":False,"selection":"fixed arithmetic mean of4000,5000,6000 weights; selected prospectively before5000/6000 evaluation"},indent=2)+"\n")
 (A/"late_averaging_complete.json").write_text(json.dumps({"status":"complete","runs":[n+"_swa" for n in names]})+"\n")
if __name__=="__main__":main()
