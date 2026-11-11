"""Check an untested affine coordinate hypothesis from near-source flat FIT patches.
Fitted peak centers are signal-footprint proxies, not actual runtime TX records.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import least_squares
R=Path(__file__).resolve().parents[1];D=R.parent/'data';O=R/'audit/c16b_nearfield_affine_probe';O.mkdir(exist_ok=False)
rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r');Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
selection=json.loads((R/'audit/geometry_model_diagnosis_20261031/nearfield_geometry.json').read_text())['original_rows']
dr,dc=np.meshgrid(np.arange(-4,5),np.arange(-4,5),indexing='ij');dr=dr.ravel();dc=dc.ravel();rad=np.hypot(dr,dc)
def item(i):
 r=rows[i];tr=int(r['tx_row']);tc=int(r['tx_col']);h=H[int(r['hm_row'])];ht=float(h[tr,tc]);dh=h[tr+dr,tc+dc].astype(float)-ht
 out={'index_row':i,'terrain_id':r['terrain_id'],'tx_row':tr,'tx_col':tc,'tx_height':ht,'bands':{}}
 for k,b in enumerate(['415','58']):
  m=np.unpackbits(M[k][i],bitorder='big').reshape(256,256)[tr+dr,tc+dc].astype(bool)&(rad>=1.5);y=Y[k][i,tr+dr,tc+dc].astype(float);q0=20*np.log10(4*np.pi*[415e6,5.775e9][k]/299792458)
  def err(v):return 10*np.log10((dr[m]-v[0])**2+(dc[m]-v[1])**2+(dh[m]-v[2])**2)+v[3]-y[m]
  fit=least_squares(err,[.5,-.5,1.5,q0],bounds=([-2,-2,.2,q0-6],[2,2,4,q0+6]),loss='soft_l1',f_scale=.1,max_nfev=150)
  out['bands'][b]={'row_offset':float(fit.x[0]),'col_offset':float(fit.x[1]),'effective_vertical_separation':float(fit.x[2]),'gain_offset_db':float(fit.x[3]-q0),'residual_RMSE_db':float(np.sqrt(np.mean(err(fit.x)**2))),'n':int(m.sum()),'success':bool(fit.success)}
 return out
start=time.time()
with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(item,selection))
terrains=sorted({r['terrain_id'] for r in records});rng=np.random.default_rng(20261101);rng.shuffle(terrains);cal=set(terrains[:len(terrains)//2])
A=np.array([[1,(r['tx_row']-127.5)/128,(r['tx_col']-127.5)/128] for r in records]);split=np.array([r['terrain_id'] in cal for r in records]);results={}
for b in ['415','58']:
 target=np.array([[r['bands'][b]['row_offset'],r['bands'][b]['col_offset']] for r in records]);valid=np.array([r['bands'][b]['success'] and r['bands'][b]['residual_RMSE_db']<.5 and abs(r['bands'][b]['row_offset'])<1.95 and abs(r['bands'][b]['col_offset'])<1.95 for r in records]);fit=split&valid;check=(~split)&valid
 coef=np.linalg.lstsq(A[fit],target[fit],rcond=None)[0];constant=target[fit].mean(0)
 baseline=((target[check]-constant)**2).mean(0);new=((target[check]-A[check]@coef)**2).mean(0)
 height=np.array([r['tx_height'] for r in records]);z=np.array([r['bands'][b]['effective_vertical_separation'] for r in records]);phase=2-np.remainder(height,1)
 results[b]={'calibration_maps':int(fit.sum()),'confirmation_maps':int(check.sum()),'mean_center_offset_calibration_rc':constant.tolist(),'affine_coefficients_rows_intercept_rowcoord_colcoord':coef.tolist(),'confirmation_offset_RMSE_constant':np.sqrt(baseline).tolist(),'confirmation_offset_RMSE_affine':np.sqrt(new).tolist(),'affine_R2_vs_constant':(1-new/baseline).tolist(),'corr_effective_separation_vs_2_minus_TX_fraction':float(np.corrcoef(z[check],phase[check])[0,1]),'median_nearfield_fit_RMSE_db':float(np.median([r['bands'][b]['residual_RMSE_db'] for r in records]))}
out={'scope':'413 nearflat FIT maps from248 terrains; terrain-disjoint124/124 calibration/confirmation; no official VAL or DEV','selection':'Reused input-only local9x9height range<=0.25m selection. Signal-footprint fits use only radii>=1.5m.','vertical_convention':'Corrected physical distance uses (Hrx-Htx-z)^2 for TX above RX; C16 used the opposite sign in nearflat patches. Both runs retained.', 'warning':'Signal-footprint center/vertical-separation fits can reflect cell averaging and propagation. They are not recovered actual TX coordinates.','bands':results,'seconds':time.time()-start}
(O/'records.json').write_text(json.dumps(records)+'\n');(O/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
