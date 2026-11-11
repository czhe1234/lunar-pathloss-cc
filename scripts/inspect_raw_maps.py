"""Original-data visual check on three preselected training inputs; no model ranking."""
import os,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'pdf.fonttype':42,'ps.fonttype':42})
rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r');relief=np.ptp(hm,axis=(1,2));dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids'])
candidates=[i for i,r in enumerate(rows) if r['terrain_id'] not in dev and r['group']=='single'];selected=[min(candidates,key=lambda i:abs(float(relief[int(rows[i]['hm_row'])])-t)) for t in (10,35,150)]
ys=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];ms=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
fig,axs=plt.subplots(3,4,figsize=(183/25.4,147/25.4),layout='constrained')
records=[]
for k,i in enumerate(selected):
 r=rows[i];h=hm[int(r['hm_row'])];y=[a[i].astype(np.float32) for a in ys];m=[np.unpackbits(a[i]).reshape(256,256).astype(bool) for a in ms]
 maps=[h,np.where(m[0],y[0],np.nan),np.where(m[1],y[1],np.nan),np.where(m[0]&m[1],y[1]-y[0],np.nan)]
 titles=[f'Terrain ({float(h.max()):.1f} m relief)','415 MHz (dB)','5.8 GHz (dB)','5.8 GHz - 415 MHz (dB)']
 limits=[(0,float(h.max())),(20,193),(43,228),(0,70)]
 for j in range(4):
  ax=axs[k,j];im=ax.imshow(maps[j],cmap='terrain' if j==0 else 'viridis',vmin=limits[j][0],vmax=limits[j][1],origin='upper');ax.plot(int(r['tx_col']),int(r['tx_row']),'x',color='red' if j==0 else 'white',ms=4,mew=.8);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(titles[j]);fig.colorbar(im,ax=ax,shrink=.75,pad=.02)
 records.append({'index_row':i,'sample_id':r['sample_id'],'terrain_id':r['terrain_id'],'relief_m':float(h.max()),'valid_fraction415':float(m[0].mean()),'valid_fraction58':float(m[1].mean())})
out=ROOT/'audit';fig.savefig(out/'raw_map_inspection.png',dpi=200);fig.savefig(out/'raw_map_inspection_preview.png',dpi=120);plt.close(fig);(out/'raw_map_inspection.json').write_text(json.dumps(records,indent=2)+'\n');print(records)
