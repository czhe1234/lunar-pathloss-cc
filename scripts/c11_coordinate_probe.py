"""Input-coordinate hypothesis test on FIT terrains only; no model training.
TX shift and RX shift are distinct. A fixed TX-height offset is calibrated on
half the FIT terrains and checked on the other half.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,csv,time,numpy as np
from scipy.ndimage import map_coordinates
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parents[1];D=R/'features/c6_points';rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
ids=np.load(D/'fit_map_indices.npy');pix=np.load(D/'fit_pixel.npy',mmap_mode='r').reshape(len(ids),64);y=np.load(D/'fit_y.npy',mmap_mode='r').reshape(len(ids),64,2);fs=np.load(D/'fit_fspl.npy',mmap_mode='r').reshape(len(ids),64,2);mask=np.load(D/'fit_mask.npy',mmap_mode='r').reshape(len(ids),64,2)
rng=np.random.default_rng(20261101);order=rng.permutation(len(ids));sel=[];seen=set()
for j in order:
 tid=rows[ids[j]]['terrain_id']
 if tid not in seen:sel.append(int(j));seen.add(tid)
 if len(sel)==1024:break
t=(np.arange(256)+.5)/256
families={'original':(0.,0.,0.,0.),'TX_half':(.5,-.5,0.,0.),'RX_half':(0.,0.,-.5,.5)}
def one(j):
 row=rows[ids[j]];h=H[int(row['hm_row'])];tr=int(row['tx_row']);tc=int(row['tx_col']);rr,cc=np.divmod(pix[j],256);ht=float(h[tr,tc]);out={}
 for name,(sr,sc,dr,dc) in families.items():
  ty=tr+sr;tx=tc+sc;ry=np.clip(rr+dr,0,255);rx=np.clip(cc+dc,0,255)
  hr=map_coordinates(h,np.stack((ry,rx)),order=1,mode='nearest',prefilter=False)
  yy=ty+(ry-ty)[:,None]*t;xx=tx+(rx-tx)[:,None]*t
  hh=map_coordinates(h,np.stack((yy,xx)),order=1,mode='nearest',prefilter=False)
  e=hh-(ht+3+(hr+1-ht-3)[:,None]*t)
  critical=3+(e/(1-t)).max(1)
  out[name+'_constant']=critical-3
  out[name+'_phase0']=critical-(3-ht%1)
 return j,out
start=time.time();parts={};res=[]
with ThreadPoolExecutor(max_workers=8) as pool:
 for j,out in pool.map(one,sel):
  z=y[j]-fs[j];joint=mask[j].all(1)
  clear=joint&(z[:,0]<3)&(z[:,1]<3);blocked=joint&(z[:,0]>6)&(z[:,1]>12)
  res.append((out,clear,blocked))
for split,ix in [('calibrate',range(512)),('confirm',range(512,1024))]:
 parts[split]={'c':np.concatenate([res[j][1] for j in ix]),'b':np.concatenate([res[j][2] for j in ix]),'x':{name:np.concatenate([res[j][0][name] for j in ix]) for name in res[0][0]}}
ans={}
for name in res[0][0]:
 a=parts['calibrate'];grid=np.arange(-1.5,.5001,.025);ba=np.array([.5*((a['x'][name][a['c']]<=d).mean()+(a['x'][name][a['b']]>d).mean()) for d in grid]);delta=float(grid[ba.argmax()]);z=parts['confirm'];pred=z['x'][name]<=delta
 ans[name]={'calibrated_height_offset_m':delta,'calibration_BA':float(ba.max()),'confirmation_BA':float(.5*(pred[z['c']].mean()+(~pred[z['b']]).mean())),'confirmation_clear_recall':float(pred[z['c']].mean()),'confirmation_blocked_recall':float((~pred[z['b']]).mean())}
result={'scope':'1024 FIT terrains, one map each, 64 fixed uniform pixels/map; split first/last512 selected terrains; no official VAL or DEV labels','classes':'joint-valid, clear: both excess<3dB; blocked: low>6dB and high>12dB. Attenuation proxies, not path labels.','confirm_clear_count':int(parts['confirm']['c'].sum()),'confirm_blocked_count':int(parts['confirm']['b'].sum()),'hypotheses':ans,'seconds':time.time()-start}
o=R/'audit/c11_coordinate_probe';o.mkdir(exist_ok=True);(o/'result.json').write_text(json.dumps(result,indent=2)+'\n');(o/'selection.json').write_text(json.dumps({'index_rows':[int(ids[j]) for j in sel]})+'\n');print(json.dumps(result,indent=2),flush=True)

