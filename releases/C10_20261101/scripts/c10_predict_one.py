"""Predict two full maps from a raw heightmap and TX, using the fixed C10 model."""
import argparse,sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c10_inference import C10Predictor
p=argparse.ArgumentParser();p.add_argument('--model-directory',default='runs/c10_full_refit_s20261101');p.add_argument('--heightmap',required=True);p.add_argument('--hm-row',type=int);p.add_argument('--tx-row',type=int,required=True);p.add_argument('--tx-col',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
hp=Path(a.heightmap);hp=hp if hp.is_absolute() else R/hp;h=np.load(hp,mmap_mode='r',allow_pickle=False)
if a.hm_row is not None:h=h[a.hm_row]
out=(R/a.output).resolve()
if not out.is_relative_to(R) or out.exists() or out.suffix!='.npy':raise ValueError('Output must be a new .npy file inside CC')
out.parent.mkdir(parents=True,exist_ok=True);model=C10Predictor(R/a.model_directory);pr=model.predict(h,a.tx_row,a.tx_col);np.save(out,pr)
print(json.dumps({'output':str(out),'shape':list(pr.shape),'bands_hz':[415e6,5.775e9],'units':'positive pathloss dB','GPU':model.device,'model_sha256':model.model_sha256,'no_target_or_mask_read':True}),flush=True)

