"""C33 GPU symmetric-tree regression, matched C10 physical inputs and split."""
from cc_core import ROOT,atomic_json,gpu_guard,check_source_unchanged
import argparse,json,time,hashlib,shutil
import numpy as np
import torch
from catboost import CatBoostRegressor,Pool
A=ROOT/"audit/c33_gpu_trees_20261101"
def main(args):
 gpu=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15);check_source_unchanged()
 p=json.loads((A/"protocol.json").read_text());k=["415","58"].index(args.band)
 run=ROOT/"runs"/("c33_catboost"+("_smoke" if args.smoke else ""));run.mkdir(exist_ok=True)
 pools={}
 for part in ["fit","dev"]:
  r=ROOT/"features/c6_points";mask=np.load(r/(part+"_mask.npy"),mmap_mode="r")[:,k].astype(bool)
  x=np.load(ROOT/"features/c10_points"/(part+"_x.npy"),mmap_mode="r")[mask]
  y=np.load(r/(part+"_y.npy"),mmap_mode="r")[:,k][mask]-np.load(r/(part+"_fspl.npy"),mmap_mode="r")[:,k][mask]
  if args.smoke:x=x[:65536];y=y[:65536]
  pools[part]=Pool(x,y);del x,y
 params=p["params"].copy();params["train_dir"]=str(run/("logs"+args.band));params["allow_writing_files"]=True
 if args.smoke:params["iterations"]=20;params["early_stopping_rounds"]=20
 model=CatBoostRegressor(**params);t0=time.time()
 atomic_json(run/("config"+args.band+".json"),{"gpu":gpu,"params":params,"rows":{k:v.num_row() for k,v in pools.items()},"catboost":"1.2.8","source_sha256":hashlib.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest()})
 model.fit(pools["fit"],eval_set=pools["dev"],verbose=100,use_best_model=True)
 model.save_model(str(run/("band"+args.band+".cbm")));model.save_model(str(run/("band"+args.band+".json")),format="json")
 pred=model.predict(pools["dev"],task_type="CPU",thread_count=8);rmse=float(np.sqrt(np.square(pred-pools["dev"].get_label()).mean()))
 np.save(run/("dev_residual_pred"+args.band+".npy"),pred.astype(np.float32))
 result={"band":args.band,"trees":model.tree_count_,"best_iteration":model.get_best_iteration(),"sampled_DEV_rmse":rmse,"seconds":time.time()-t0,"gpu":gpu,"smoke":args.smoke}
 atomic_json(run/("result"+args.band+".json"),result);shutil.copy2(__import__("pathlib").Path(__file__),run/"source_c33_train.py")
 print(json.dumps(result),flush=True);check_source_unchanged()
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--band",required=True,choices=["415","58"]);ap.add_argument("--smoke",action="store_true");main(ap.parse_args())
