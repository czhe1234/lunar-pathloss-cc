"""Fixed C10 full TRAIN refit. Does not load official validation or select iterations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,time,hashlib,shutil
import numpy as np,lightgbm as lgb
R=Path(__file__).resolve().parents[1];D=R/'features/c10_full_points';anchor=R/'runs/c10_points_phase_geometry_s20261101'
score=json.loads((R/'runs/c9_points_combined_s20261101/predictions/train_dev_gpu_c9_fresh/independent_score.json').read_text())
assert score['complete_requested_split'] and score['maps_per_band']==2562
# Full-map confirmation of the already selected branch is required.
assert all(score['bands'][b]['rmse_db'] < score['C4_reference'][b]['rmse_db'] for b in ['415','58'])
assert json.loads((D/'complete.json').read_text())['complete']
run=R/'runs/c10_full_refit_s20261101';run.mkdir(exist_ok=True);assert not (run/'config.json').exists();(run/'source').mkdir()
cfg=json.loads((anchor/'config.json').read_text());sel=json.loads((anchor/'result.json').read_text());params=cfg['parameters'];meta=json.loads((D/'metadata.json').read_text())
config={'anchor_run':str(anchor.relative_to(R)),'parameters':params,'trees_per_band':{b:sel['bands'][b]['best_iteration'] for b in ['415','58']},'point_data_metadata':meta,'no_official_validation_loaded_or_used':True,'selection':'Fixed tree counts selected on original DEV sampled points; C9 full-map framework confirmed. C10 full-map verification runs concurrently and must pass before official inference.','trained_maps':25720,'trained_terrains':6120,'sampled_points':1646080}
config['full_DEV_confirmation_required_before_official_export']=True
(run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for p in [Path(__file__),R/'scripts/c10_prepare_full_refit.py',R/'scripts/c10_predict_gpu.py',R/'src/gpu_forest.py',R/'src/c9_extra_gpu.py',R/'src/c7_surface_features.py',R/'src/c8_tx_context.py',R/'src/c3_geometry.py',R/'src/c4_geometry.py',R/'src/c10_phase_geometry.py']:
 shutil.copy2(p,run/'source'/p.name)
x=np.load(D/'x.npy',mmap_mode='r');y=np.load(D/'y.npy',mmap_mode='r');fs=np.load(D/'fspl.npy',mmap_mode='r');mask=np.load(D/'mask.npy',mmap_mode='r');start=time.time();result={}
for k,b in enumerate(['415','58']):
 keep=mask[:,k].astype(bool);ds=lgb.Dataset(np.ascontiguousarray(x[keep]),label=(y[:,k]-fs[:,k])[keep],feature_name=meta['features'])
 def progress(env):
  if env.iteration%100==0:
   rec={'band':b,'trees_completed':env.iteration+1,'trees_fixed':config['trees_per_band'][b],'seconds':time.time()-start}
   (run/'progress.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
 model=lgb.train(params,ds,num_boost_round=config['trees_per_band'][b],callbacks=[progress])
 model.save_model(str(run/f'band{b}.txt'));result[b]={'trees':config['trees_per_band'][b],'valid_train_sampled_pixels':int(keep.sum()),'sha256':hashlib.sha256((run/f'band{b}.txt').read_bytes()).hexdigest()}
 del ds,model
final={'state':'full_training_complete_not_yet_officially_scored','bands':result,'seconds':time.time()-start,'official_val_used':False,'GPU_used':False}
(run/'result.json').write_text(json.dumps(final,indent=2)+'\n');(run/'progress.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)

