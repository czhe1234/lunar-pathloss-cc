"""Inspect representative development errors using saved independent predictions."""
import os,json,csv,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'pdf.fonttype':42,'ps.fonttype':42})
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);args=p.parse_args();out=ROOT/args.predictions
mapping=json.loads((out/'indices.json').read_text());split=mapping['source_split'];indices=mapping['index_rows'];loc={i:j for j,i in enumerate(indices)}
rows=list(csv.DictReader((DATA/f'{split}_index.csv').open()));errors=list(csv.DictReader((out/'error_by_map.csv').open()));errors.sort(key=lambda r:float(r['rmse58']));chosen={f'p{q}':errors[round((len(errors)-1)*q/100)] for q in (50,95,100)}
hm=np.load(DATA/f'{split}_hm.npy',mmap_mode='r');truths=[np.load(DATA/f'{split}_pl{b}.npy',mmap_mode='r') for b in ('415','58')];preds=[np.load(out/f'pl{b}.npy',mmap_mode='r') for b in ('415','58')];packed=[np.load(DATA/f'{split}_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
records=[]
for label,item in chosen.items():
 i=int(item['index_row']);row=rows[i];h=hm[int(row['hm_row'])];truth=[a[i].astype(np.float32) for a in truths];pred=[a[loc[i]] for a in preds];mask=[np.unpackbits(a[i]).reshape(256,256).astype(bool) for a in packed]
 fig,axs=plt.subplots(3,3,figsize=(183/25.4,173/25.4),layout='constrained')
 top=[h,mask[0],mask[1]]
 for j,title in enumerate(['Terrain height (m)','415 MHz valid mask','5.8 GHz valid mask']):
  ax=axs[0,j];im=ax.imshow(top[j],origin='upper',cmap='terrain' if j==0 else 'gray',vmin=0,vmax=float(h.max()) if j==0 else 1);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.8,pad=.02)
 for b,band in enumerate(('415 MHz','5.8 GHz')):
  values=truth[b][mask[b]];low=float(np.percentile(values,1));high=float(np.percentile(values,99));limit=max(5.,float(np.percentile(np.abs(pred[b][mask[b]]-values),99)))
  for j,title in enumerate(['True pathloss','Predicted pathloss','Prediction - truth']):
   ax=axs[b+1,j];v=truth[b] if j==0 else pred[b] if j==1 else pred[b]-truth[b];v=np.where(mask[b],v,np.nan)
   im=ax.imshow(v,origin='upper',cmap='RdBu_r' if j==2 else 'viridis',vmin=-limit if j==2 else low,vmax=limit if j==2 else high);ax.set_title(f'{band}: {title} (dB)');fig.colorbar(im,ax=ax,shrink=.8,pad=.02)
 for ax in axs.ravel():ax.set_xticks([]);ax.set_yticks([]);ax.plot(int(row['tx_col']),int(row['tx_row']),'x',ms=4,mew=.8,color='red' if ax is axs[0,0] else 'black')
 fig.suptitle(f"{label} by 5.8 GHz error | {row['sample_id']} | RMSE: {float(item['rmse415']):.3f} / {float(item['rmse58']):.3f} dB",fontsize=9)
 path=out/f'error_map_{label}.png';fig.savefig(path,dpi=180);plt.close(fig);im=Image.open(path).convert('RGB');im.thumbnail((950,900));im.save(out/f'error_map_{label}_preview.jpg',quality=70);records.append({'selection':label,**item})
(out/'error_map_examples.json').write_text(json.dumps(records,indent=2)+'\n');print(records)
