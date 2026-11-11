"""Transfer nearfield-calibrated affine location hypotheses to disjoint FIT terrains."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[1];D=R/'features/c6_points';O=R/'audit/c16b_nearfield_affine_probe'
aff=json.loads((O/'result.json').read_text());coef=np.mean([aff['bands'][b]['affine_coefficients_rows_intercept_rowcoord_colcoord'] for b in ['415','58']],axis=0)
mat=np.eye(2)+coef[1:].T/128;intercept=coef[0]-np.array([127.5/128,127.5/128])@coef[1:];inv=np.linalg.inv(mat)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');ids=np.load(D/'fit_map_indices.npy');pos={int(i):j for j,i in enumerate(ids)}
excluded={r['terrain_id'] for r in json.loads((O/'records.json').read_text())}
old=json.loads((R/'audit/c11_coordinate_probe/selection.json').read_text())['index_rows'];selected=[i for i in old if rows[i]['terrain_id'] not in excluded];sel=[pos[i] for i in selected];cut=len(sel)//2
pix=np.load(D/'fit_pixel.npy',mmap_mode='r').reshape(len(ids),64);y=np.load(D/'fit_y.npy',mmap_mode='r').reshape(len(ids),64,2);fs=np.load(D/'fit_fspl.npy',mmap_mode='r').reshape(len(ids),64,2);mask=np.load(D/'fit_mask.npy',mmap_mode='r').reshape(len(ids),64,2)
names=['TX_floor_original','TX_constant_shift','TX_affine_given_height','TX_affine_resampled_height','RX_constant_shift','RX_inverse_affine_resampled_height','RX_inverse_affine_given_height']
t=(np.arange(256,dtype=float)+.5)/256
def one(j):
 row=rows[int(ids[j])];h=H[int(row['hm_row'])];tr=int(row['tx_row']);tc=int(row['tx_col']);rr,cc=np.divmod(pix[j],256);ht=float(h[tr,tc]);tx=np.array([tr,tc],float);rx=np.stack((rr,cc),1).astype(float);ta=mat@tx+intercept;ra=(rx-intercept)@inv.T
 hr=h[rr,cc].astype(float);hactual=float(map_coordinates(h,np.array([[ta[0]],[ta[1]]]),order=1,mode='nearest',prefilter=False)[0])
 cases=[(tx,rx,ht,hr),(tx+np.array([.5,-.5]),rx,ht,hr),(ta,rx,ht,hr),(ta,rx,hactual,hr),(tx,rx+np.array([-.5,.5]),ht,None),(tx,ra,ht,None),(tx,ra,ht,hr)]
 out={}
 for name,(tp,rp,th,rh) in zip(names,cases):
  rp=np.clip(rp,0,255);tp=np.clip(tp,0,255)
  if rh is None:rh=map_coordinates(h,rp.T,order=1,mode='nearest',prefilter=False).astype(float)
  q=tp[None,:,None]+(rp-tp)[:,:,None]*t;surf=map_coordinates(h,np.stack((q[:,0],q[:,1])),order=1,mode='nearest',prefilter=False)
  zt=np.floor(th)+3;e=surf-(zt+(rh+1-zt)[:,None]*t)
  out[name]=(e/(1-t)).max(1)
 z=y[j]-fs[j];joint=mask[j].all(1)
 return out,joint&(z[:,0]<3)&(z[:,1]<3),joint&(z[:,0]>6)&(z[:,1]>12)
start=time.time()
with ThreadPoolExecutor(max_workers=8) as pool:res=list(pool.map(one,sel))
parts={}
for part,ix in [('calibrate',range(cut)),('confirm',range(cut,len(sel)))]:
 parts[part]={'clear':np.concatenate([res[j][1] for j in ix]),'blocked':np.concatenate([res[j][2] for j in ix]),'x':{n:np.concatenate([res[j][0][n] for j in ix]) for n in names}}
ans={};grid=np.arange(-1.5,.5001,.025)
for n in names:
 a=parts['calibrate'];z=parts['confirm'];ba=np.array([.5*((a['x'][n][a['clear']]<=d).mean()+(a['x'][n][a['blocked']]>d).mean()) for d in grid]);delta=float(grid[ba.argmax()]);pred=z['x'][n]<=delta
 ans[n]={'calibrated_TX_vertical_offset':delta,'calibration_BA':float(ba.max()),'confirmation_BA':float(.5*(pred[z['clear']].mean()+(~pred[z['blocked']]).mean())),'clear_recall':float(pred[z['clear']].mean()),'blocked_recall':float((~pred[z['blocked']]).mean())}
for n in names:ans[n]['confirmation_BA_delta_vs_original_pp']=100*(ans[n]['confirmation_BA']-ans[names[0]]['confirmation_BA'])
out={'scope':'Original1024 FIT terrains excluding all248 nearfield-fit terrains. Remaining terrains split disjointly for vertical-offset calibration/confirmation. No internal DEV/official VAL.','terrains_total':len(sel),'calibration_terrains':cut,'confirmation_terrains':len(sel)-cut,'confirmation_counts':{k:int(parts['confirm'][k].sum()) for k in ['clear','blocked']},'affine_center_matrix':mat.tolist(),'affine_center_intercept':intercept.tolist(),'cases':ans,'seconds':time.time()-start,'training':False}
(O/'geometry_transfer.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
