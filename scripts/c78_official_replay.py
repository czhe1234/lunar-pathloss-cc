"""Replay frozen diagnostics on the deployed official-C76 predictions after the TRAIN-side decision."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c78_clear'
choice=json.load(open(A/'internal_decision.json'));assert not choice['train_new_network']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected=choice['coefficients_sha256'];assert digest(A/'internal_coefficients.json')==expected
needed=[A/f'val_canonical_{j}.npz' for j in range(3)]+[A/f'val_{kind}_{j}of3/complete.json' for kind in ('geom','power','sheet') for j in range(3)]+[A/'val_reference.json',A/'weak_marker_calibration.json']
for _ in range(600):
    if all(p.exists() for p in needed):break
    time.sleep(2)
else:raise RuntimeError('incomplete inputs')
ref=np.load(A/'val_reference.npy',mmap_mode='r+');seen=np.zeros(len(ref),bool)
for j in range(3):
    d=np.load(A/f'val_canonical_{j}.npz');pos=d['positions'];assert not seen[pos].any();seen[pos]=1;ref[pos,:,6]=d['values']
assert seen.all();ref.flush()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='5',OMP_NUM_THREADS='5')
for script in ['c78_geometry_attribution','c78_counterfactual','c78_matched']:
    with open(R/f'logs/{script}_val.log','w') as log:subprocess.run(['.venv/bin/python',f'scripts/{script}.py','val'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
assert digest(A/'internal_coefficients.json')==expected
(A/'official_replay_complete.json').write_text(json.dumps(dict(official_predictions='runs/c76_selected/predictions',official_maps=2330,terrains=680,uniform_locations_per_map=2048,
    marker_and_radiometry_fit_on='official TRAIN internal calibration terrains only',official_USED_for_new_model_selection=False,official_new_training=False,coefficient_hash_unchanged=expected),indent=2)+'\n')
print('deployed C76 frozen validation replay complete',flush=True)
