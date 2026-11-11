from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[1];A=R/'audit/c124_spectrum'
inp=json.load(open(A/'pair_inputs.json'));allpairs=json.load(open(R/'audit/c121_pairs/monitor/pairs.json'));pairs=[allpairs[i] for i in inp['pair_indices']]
y=np.array([p['target'][:2] for p in pairs],dtype='float64');base=np.array([p['base'][:2] for p in pairs],dtype='float64');band=np.array([p['band'] for p in pairs])
out={'scope':'posthoc failure diagnostic on same 24 maps frozen before C122; no fitting or candidate selection','bands':[]}
for b in range(2):
 keep=band==b;target=y[keep,:,b];p=base[keep,:,b];res=target-p
 rec={'band':b,'pairs':int(keep.sum()),'C82_endpoint_rmse':float(np.sqrt(np.mean((p-target)**2))),'C82_correct_pathloss_order_fraction':float(np.mean(p[:,0]>p[:,1]))}
 for mode in ['total','spectrum']:
  delta=np.load(A/f'pair_delta_{mode}.npy')[keep,:,b].astype('float64');pred=p+delta
  rec[mode]=dict(endpoint_rmse=float(np.sqrt(np.mean((pred-target)**2))),both_correction_signs_correct=float(np.mean((delta[:,0]>0)&(delta[:,1]<0))),predicted_anchor_pathloss_greater=float(np.mean(pred[:,0]>pred[:,1])),median_true_pathloss_contrast_db=float(np.median(target[:,0]-target[:,1])),median_predicted_pathloss_contrast_db=float(np.median(pred[:,0]-pred[:,1])))
 out['bands'].append(rec)
(A/'pair_score.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
