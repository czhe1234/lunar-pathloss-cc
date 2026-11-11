"""C31 extends the capped C10 learner on exactly its original FIT/DEV point data."""
from pathlib import Path
import json,time,argparse,hashlib,shutil,resource
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[1]
def main(args):
 b=args.band;k=["415","58"].index(b);parent=R/"runs/c10_points_phase_geometry_s20261101"
 out=R/"runs/c31_extended";out.mkdir(exist_ok=True)
 cfg=json.loads((parent/"config.json").read_text());params=cfg["parameters"];params["num_threads"]=12
 arrays={}
 for part in ["fit","dev"]:
  root=R/"features/c6_points"
  x=np.load(R/"features/c10_points"/(part+"_x.npy"),mmap_mode="r")
  y=np.load(root/(part+"_y.npy"),mmap_mode="r")[:,k]
  fs=np.load(root/(part+"_fspl.npy"),mmap_mode="r")[:,k]
  mask=np.load(root/(part+"_mask.npy"),mmap_mode="r")[:,k].astype(bool)
  arrays[part]=(np.asarray(x[mask]),np.asarray(y[mask]-fs[mask],np.float32))
 old=lgb.Booster(model_file=str(parent/("band"+b+".txt")))
 assert old.num_trees()==1200 and old.num_feature()==294
 xp,yp=arrays["dev"];base=old.predict(xp,num_threads=12);oldrmse=float(np.sqrt(np.square(base-yp).mean()))
 expected=json.loads((parent/"result.json").read_text())["bands"][b]["rmse_db"]
 assert abs(oldrmse-expected)<1e-5,(oldrmse,expected)
 conf={"parameters":params,"max_total_trees":6000,"parent_sha256":hashlib.sha256((parent/("band"+b+".txt")).read_bytes()).hexdigest(),"baseline_rmse":oldrmse,"valid_counts":{n:len(v[1]) for n,v in arrays.items()},"no_GPU_used":True}
 (out/("config"+b+".json")).write_text(json.dumps(conf,indent=2)+"\n")
 tr=lgb.Dataset(*arrays["fit"],params=params,free_raw_data=False)
 dv=lgb.Dataset(*arrays["dev"],reference=tr,params=params,free_raw_data=False)
 hist={};t0=time.time()
 def progress(env):
  if (env.iteration+1)%100==0:
   d={"iteration":env.iteration+1,"eval":env.evaluation_result_list,"seconds":time.time()-t0}
   (out/("progress"+b+".json")).write_text(json.dumps(d,indent=2)+"\n");print(json.dumps(d),flush=True)
 model=lgb.train(params,tr,num_boost_round=4800,valid_sets=[tr,dv],valid_names=["fit","dev"],init_model=old,
    callbacks=[lgb.record_evaluation(hist),lgb.early_stopping(300,first_metric_only=True,verbose=True),progress])
 pred=model.predict(xp,num_iteration=model.best_iteration,num_threads=12)
 rmse=float(np.sqrt(np.square(pred-yp).mean()))
 model.save_model(str(out/("band"+b+".txt")),num_iteration=model.best_iteration)
 np.save(out/("dev_residual_pred"+b+".npy"),pred.astype(np.float32))
 result={"band":b,"selected_trees":model.best_iteration,"rmse":rmse,"baseline_rmse":oldrmse,"relative_gain":1-rmse/oldrmse,"seconds":time.time()-t0,"max_RSS_kB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 (out/("history"+b+".json")).write_text(json.dumps(hist)+"\n");(out/("result"+b+".json")).write_text(json.dumps(result,indent=2)+"\n")
 shutil.copy2(Path(__file__),out/"source_c31_extend.py");print(json.dumps(result),flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--band",choices=["415","58"],required=True);main(ap.parse_args())
