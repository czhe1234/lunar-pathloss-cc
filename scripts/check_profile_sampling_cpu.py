"""Input-only sampling resolution study on three fixed training terrains."""
import os,csv,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8})
examples=json.loads((ROOT/'audit/raw_map_inspection.json').read_text());rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r');truth=np.load(DATA/'train_pl58.npy',mmap_mode='r');mask=np.load(DATA/'train_mask58.npy',mmap_mode='r');yy,xx=np.indices((256,256),dtype=np.float32);xx=xx.flatten();yy=yy.flatten()
fig,axs=plt.subplots(3,3,figsize=(183/25.4,175/25.4),layout='constrained');records=[];t0=time.time()
for k,item in enumerate(examples):
 i=item['index_row'];row=rows[i];h=np.array(hm[int(row['hm_row'])]);r=int(row['tx_row']);c=int(row['tx_col']);result={}
 for ns in (48,192,384):
  output=np.empty(65536,dtype=np.float32);t=np.linspace(0,1,ns+2,dtype=np.float32)[1:-1]
  for lo in range(0,65536,4096):
   x=xx[lo:lo+4096];y=yy[lo:lo+4096];px=c+(x[:,None]-c)*t;py=r+(y[:,None]-r)*t
   profile=map_coordinates(h,np.stack([py,px]),order=1,mode='nearest',prefilter=False)
   line=(h[r,c]+3)*(1-t)+(h.flatten()[lo:lo+4096,None]+1)*t
   output[lo:lo+len(x)]=(profile-line).max(-1)
  result[ns]=output.reshape(256,256)
 rec={'sample_id':row['sample_id'],'terrain_relief_m':item['relief_m'],'comparisons_to_384':{}}
 for ns in (48,192):
  err=np.abs(result[ns]-result[384]);rec['comparisons_to_384'][str(ns)]={'mean_abs_m':float(err.mean()),'p99_abs_m':float(np.percentile(err,99)),'max_abs_m':float(err.max()),'blocked_unblocked_disagreement_fraction':float(((result[ns]>0)!=(result[384]>0)).mean())}
 records.append(rec)
 valid=np.unpackbits(mask[i]).reshape(256,256).astype(bool);y=np.where(valid,truth[i].astype(np.float32),np.nan)
 panels=[h,result[384],y];titles=[f"Terrain: {row['sample_id']}",'Max terrain excess above segment (m)','5.8 GHz (dB); white = zero excess']
 for j in range(3):
  ax=axs[k,j];im=ax.imshow(panels[j],cmap='terrain' if j==0 else 'RdBu_r' if j==1 else 'viridis');fig.colorbar(im,ax=ax,shrink=.8,pad=.02);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(titles[j]);ax.plot(c,r,'x',color='red' if j==0 else 'black',ms=4)
  if j==2 and result[384].min()<0<result[384].max():ax.contour(result[384],levels=[0],colors=['white'],linewidths=.5)
fig.savefig(ROOT/'audit/profile_sampling_inspection.png',dpi=180);plt.close(fig);im=Image.open(ROOT/'audit/profile_sampling_inspection.png').convert('RGB');im.thumbnail((900,900));im.save(ROOT/'audit/profile_sampling_inspection_preview.jpg',quality=70)
summary={'samples':records,'scope':'Three predefined fit-set input terrains. 384 samples is a numerical reference, not an exact visibility oracle. No propagation labels generated. Target arrays used only for overlay visualization.','seconds':time.time()-t0};(ROOT/'audit/profile_sampling_cpu.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
