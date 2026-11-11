"""Independent saved-output verification of all16trainingmaps, no CUDA."""
import json
from pathlib import Path
import numpy as np
from src.c106_metrics import metrics
R=Path.cwd();A=R/'audit/c107_tailfit';out={}
for mode in ['control','tail8']:
 folder=R/'runs'/('c107_'+mode)
 if not (folder/'result.json').exists():continue
 z=np.load(folder/'fit_truth.npz');pred=np.load(folder/'fit_prediction.npy')
 x=lambda a:a.transpose(0,2,3,1)
 m=metrics(x(pred),x(z['base']),x(z['target']),x(z['mask']))
 y=z['target'].astype(np.float64);w=z['mask'];e=np.where(w,pred.astype(np.float64)-y,0.)
 per=np.sqrt(np.sum(e**2,axis=(2,3))/np.sum(w,axis=(2,3)))
 r=json.loads((folder/'result.json').read_text())
 assert np.allclose(m['rmse'],r['fit']['rmse'],rtol=1e-6,atol=1e-7)
 assert np.allclose(per,r['fit']['per_map_band_rmse'],rtol=1e-6,atol=1e-7)
 m.update(per_map_band_rmse=per.tolist(),max_map_band_rmse=per.max(0).tolist(),all32below1=bool((per<1).all()),selected_step=r['selected_step'],source='savedfloat32predictions+rawlabels;NumPyFP64')
 (folder/'fit_verified.json').write_text(json.dumps(m,indent=2)+'\n');out[mode]=m
print(json.dumps({k:{'rmse':v['rmse'],'max':v['max_map_band_rmse'],'deep_fade':v['deep_fade']} for k,v in out.items()}))
