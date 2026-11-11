"""CPU-only numerical sensitivity check for the descriptive clearance audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';OUT=ROOT/'audit/physical_representations_20261031'
rows=list(csv.DictReader((DATA/'train_index.csv').open()))
indices=json.loads((ROOT/'runs/b0_raw_s20261031/predictions/train_dev_bf16/indices.json').read_text())['index_rows']
seen=set();unique=[]
for i in indices:
 if rows[i]['terrain_id'] not in seen:unique.append(i);seen.add(rows[i]['terrain_id'])
selected=[unique[k] for k in np.linspace(0,len(unique)-1,64,dtype=int)]
hm=np.load(DATA/'train_hm.npy',mmap_mode='r',allow_pickle=False)
yr,xc=np.meshgrid(np.arange(4,256,8),np.arange(4,256,8),indexing='ij');yr=yr.ravel();xc=xc.ravel()
def profile(h,tr,tc,n):
 t=np.arange(1,n+1,dtype=np.float64)/(n+1)
 gy=tr+(yr-tr)[:,None]*t;gx=tc+(xc-tc)[:,None]*t
 txz=float(h[tr,tc])+3.;rxz=h[yr,xc].astype(np.float64)+1.
 return (map_coordinates(h,np.stack((gy,gx)),order=1,mode='nearest',prefilter=False)-(txz+(rxz[:,None]-txz)*t)).max(1)
start=time.time();a=[];b=[]
for i in selected:
 r=rows[i];h=hm[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col'])
 a.append(profile(h,tr,tc,192));b.append(profile(h,tr,tc,768))
a=np.concatenate(a);b=np.concatenate(b);difference=np.abs(a-b)
zone=lambda v:np.where(v<-.25,0,np.where(v>.25,2,1))
z1=zone(a);z2=zone(b);counts=np.bincount(z1*3+z2,minlength=9).reshape(3,3)
d={'description':'192 versus 768 interior samples on the same bilinear height surface; numerical sensitivity, not original mesh validation','maps':len(selected),'unique_terrains':len(selected),'pixels':len(a),'indices':selected,'absolute_clearance_difference_m':{str(q):float(np.quantile(difference,q)) for q in [.5,.9,.99,1.]},'zone_disagreement_fraction':float(np.mean(z1!=z2)),'zone_confusion_rows192_cols768':counts.tolist(),'seconds':time.time()-start,'uses_gpu':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert counts.sum()==len(a)
(OUT/'geometry_sampling_check.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({k:v for k,v in d.items() if k!='indices'}))
