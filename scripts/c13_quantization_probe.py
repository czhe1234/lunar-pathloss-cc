"""Test explicit terrain/RX quantization hypotheses on two disjoint FIT terrain halves."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[1];D=R/'features/c6_points';O=R/'audit/c13_rx_geometry_probe';O.mkdir(exist_ok=True)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
ids=np.load(D/'fit_map_indices.npy');positions={int(v):j for j,v in enumerate(ids)}
selected=json.loads((R/'audit/c11_coordinate_probe/selection.json').read_text())['index_rows'];sel=[positions[i] for i in selected]
pix=np.load(D/'fit_pixel.npy',mmap_mode='r').reshape(len(ids),64);y=np.load(D/'fit_y.npy',mmap_mode='r').reshape(len(ids),64,2);fs=np.load(D/'fit_fspl.npy',mmap_mode='r').reshape(len(ids),64,2);mask=np.load(D/'fit_mask.npy',mmap_mode='r').reshape(len(ids),64,2)
t=(np.arange(256,dtype=np.float64)+.5)/256
names=['nominal','TX_floor','TX_RX_floor','TX_floor_RX_half_floor','all_grid_floor','all_grid_round','floor_after_surface_interpolation']
def one(j):
 row=rows[int(ids[j])];h=H[int(row['hm_row'])];tr=int(row['tx_row']);tc=int(row['tx_col']);rr,cc=np.divmod(pix[j],256);ht=float(h[tr,tc]);hr=h[rr,cc].astype(float)
 xx=tc+(cc-tc)[:,None]*t;yy=tr+(rr-tr)[:,None]*t;coords=np.stack((yy,xx))
 surfaces={n:map_coordinates(z,coords,order=1,mode='nearest',prefilter=False) for n,z in [('raw',h),('floor',np.floor(h)),('round',np.round(h))]}
 cases=[(surfaces['raw'],ht,hr),(surfaces['raw'],np.floor(ht),hr),(surfaces['raw'],np.floor(ht),np.floor(hr)),(surfaces['raw'],np.floor(ht),hr-.5*np.remainder(hr,1)),(surfaces['floor'],np.floor(ht),np.floor(hr)),(surfaces['round'],np.round(ht),np.round(hr)),(np.floor(surfaces['raw']),np.floor(ht),np.floor(hr))]
 output={}
 for name,(hh,zt,zr) in zip(names,cases):
  e=hh-(zt+3+(zr+1-zt-3)[:,None]*t)
  output[name]=(e/(1-t)).max(1)
 z=y[j]-fs[j];joint=mask[j].all(1)
 return output,joint&(z[:,0]<3)&(z[:,1]<3),joint&(z[:,0]>6)&(z[:,1]>12)
start=time.time()
with ThreadPoolExecutor(max_workers=8) as pool:res=list(pool.map(one,sel))
parts={}
for part,ix in [('calibrate',range(512)),('confirm',range(512,1024))]:
 parts[part]={'clear':np.concatenate([res[j][1] for j in ix]),'blocked':np.concatenate([res[j][2] for j in ix]),'x':{n:np.concatenate([res[j][0][n] for j in ix]) for n in names}}
grid=np.arange(-1.5,.5001,.025);ans={}
for n in names:
 a=parts['calibrate'];z=parts['confirm'];ba=np.array([.5*((a['x'][n][a['clear']]<=d).mean()+(a['x'][n][a['blocked']]>d).mean()) for d in grid]);delta=float(grid[ba.argmax()]);pred=z['x'][n]<=delta
 ans[n]={'calibrated_additional_TX_height_m':delta,'calibration_BA':float(ba.max()),'confirmation_BA':float(.5*(pred[z['clear']].mean()+(~pred[z['blocked']]).mean())),'confirmation_clear_recall':float(pred[z['clear']].mean()),'confirmation_blocked_recall':float((~pred[z['blocked']]).mean())}
old=json.loads((R/'audit/c11_coordinate_probe/result.json').read_text());assert abs(ans['TX_floor']['confirmation_BA']-old['hypotheses']['original_phase0']['confirmation_BA'])<1e-8
for n,a in ans.items():a['confirmation_BA_delta_vs_TX_floor_pp']=100*(a['confirmation_BA']-ans['TX_floor']['confirmation_BA'])
result={'scope':'Same 1024 distinct FIT terrains as C11, one map and 64 uniform pixels per terrain; first512 calibration, last512 confirmation. No official VAL or internal DEV labels.','classes':'joint-valid; clear both excess<3dB; blocked low>6 and high>12dB. These are attenuation proxies, not actual ray states.','not_a_new_model_score':True,'confirmation_counts':{k:int(parts['confirm'][k].sum()) for k in ['clear','blocked']},'hypotheses':ans,'seconds':time.time()-start,'no_model_training_or_GPU':True}
(O/'quantization_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
