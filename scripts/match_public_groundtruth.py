"""Match public illustration height fields to release data; no fitting of a predictor."""
from pathlib import Path
import json,csv
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';O=R/'audit/generation_alignment_20261101';V=O/'public_examples';rows={s:list(csv.DictReader((D/f'{s}_index.csv').open())) for s in ['train','val']};hm={s:np.load(D/f'{s}_hm.npy',mmap_mode='r') for s in rows};coarse={s:np.array(a[:,::8,::8],dtype=float).reshape(len(a),-1) for s,a in hm.items()};results=[]
def d4(a,k):
 b=np.rot90(a,k%4)
 return np.fliplr(b) if k>=4 else b
for sample in ['650','680']:
 a=np.load(V/f'sample_{sample}_gt_415.npz');public_h=a['height_yx'];public_rc=np.array([round(a['display_marker_points'][:,1].mean()),round(a['display_marker_points'][:,0].mean())]);candidates=[]
 for k in range(8):
  hh=d4(public_h,k);hc=hh[::8,::8].ravel()
  for split,h in coarse.items():
   mse=np.mean((h-hc[None])**2,axis=1)
   for idx in np.argsort(mse)[:3]:
    full=hm[split][idx].astype(float);e=full-hh;candidates.append({'split':split,'hm_row':int(idx),'public_to_release_d4':k,'height_rmse':float(np.sqrt(np.mean(e**2))),'height_max_abs':float(np.max(np.abs(e))),'height_correlation':float(np.corrcoef(full.ravel(),hh.ravel())[0,1])})
 candidates.sort(key=lambda x:x['height_rmse']);best=candidates[0];split=best['split'];idx=best['hm_row'];k=best['public_to_release_d4'];mark=np.zeros((256,256),dtype=int);mark[tuple(public_rc)]=1;txr,txc=np.argwhere(d4(mark,k)==1)[0];rs=[(i,r) for i,r in enumerate(rows[split]) if int(r['hm_row'])==idx];match=[(i,r) for i,r in rs if int(r['tx_row'])==txr and int(r['tx_col'])==txc];result={'public_sample':sample,'best_height_matches':candidates[:6],'public_marker_row_col':public_rc.tolist(),'aligned_marker_row_col':[int(txr),int(txc)],'best_match_terrain_id':rs[0][1]['terrain_id'],'samples_on_best_terrain':len(rs),'tx_coordinate_matches':[{'index_row':i,**r} for i,r in match]}
 if best['height_rmse']<.01 and match:
  i,r=match[0];result['label_comparison']={}
  for band in ['415','58']:
   pb=np.load(V/f'sample_{sample}_gt_{band}.npz');q=d4(pb['display_scalar_yx'],k);y=np.load(D/f'{split}_pl{band}.npy',mmap_mode='r')[i].astype(float);m=np.unpackbits(np.load(D/f'{split}_mask{band}.npy',mmap_mode='r')[i],bitorder='big').reshape(256,256).astype(bool);guess=30-q;e=guess-y
   result['label_comparison'][band]={'valid_pixels':int(m.sum()),'conversion':'positive_PL = 30 - public_display_scalar','valid_rmse':float(np.sqrt(np.mean(e[m]**2))),'valid_max_abs':float(np.max(np.abs(e[m]))),'valid_exact_after_float16_fraction':float(np.mean(guess.astype(np.float16)[m]==y[m])),'invalid_pixels':int((~m).sum()),'invalid_rmse':float(np.sqrt(np.mean(e[~m]**2))),'public_scalar_zero_nan_counts':[int((q==0).sum()),int(np.isnan(q).sum())]}
  np.savez_compressed(V/f'sample_{sample}_aligned_release.npz',height=d4(public_h,k),display_scalar415=d4(np.load(V/f'sample_{sample}_gt_415.npz')['display_scalar_yx'],k),display_scalar58=d4(np.load(V/f'sample_{sample}_gt_58.npz')['display_scalar_yx'],k),release_index_row=i)
 results.append(result);print(json.dumps(result),flush=True)
(O/'public_example_release_matches.json').write_text(json.dumps(results,indent=2)+'\n')
