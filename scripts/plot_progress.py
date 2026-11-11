"""Render the saved, completed epoch results only; no synthetic or predicted points."""
import os,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'Arial','font.size':6,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
series=[]
for run in sorted((ROOT/'runs').iterdir()):
 p=run/'history.json'
 if not p.exists():continue
 cfg=json.loads((run/'config.json').read_text())
 if cfg['part']!='fit':continue
 h=json.loads(p.read_text())
 if h:series.append((run.name,h,cfg))
if not series:raise SystemExit('No completed development epochs')
styles={'raw':('B0','#0072B2'),'geometry':('B1','#D55E00'),'profile':('B2','#009E73')}
fig,axes=plt.subplots(1,2,figsize=(183/25.4,82/25.4))
rows=[]
for j,(name,h,cfg) in enumerate(series):
 label,color=styles[cfg['mode']]
 if cfg['seed']!=20261031:label+=f" s{cfg['seed']}"
 for col,key in enumerate(['rmse415','rmse58']):
  epochs=[r['epoch'] for r in h];values=[r[key] for r in h]
  axes[col].plot(epochs,values,color=color,lw=.85,marker='o',ms=1.8,ls='-' if cfg['seed']==20261031 else '--')
  axes[col].annotate(label,(epochs[-1],values[-1]),xytext=(3,3+j*3),textcoords='offset points',color=color,fontsize=6)
 for r in h:rows.append({'run':name,'label':label,'epoch':r['epoch'],'rmse415_db':r['rmse415'],'rmse58_db':r['rmse58']})
for i,ax in enumerate(axes):
 ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',length=2.5,width=.5)
 ax.axhline(1,color='#777777',lw=.6,ls='--')
 ax.set_xlabel('Epoch');ax.set_ylabel(('415 MHz' if i==0 else '5.8 GHz')+' RMSE (dB)');ax.set_ylim(bottom=0)
 ax.text(-.12,1.02,'ab'[i],transform=ax.transAxes,fontweight='bold',fontsize=8)
 ax.margins(x=.07)
fig.subplots_adjust(left=.08,right=.97,bottom=.18,top=.94,wspace=.28)
out=ROOT/'results'
for ext in ['pdf','svg','eps']:fig.savefig(out/f'development_rmse.{ext}')
fig.savefig(out/'development_rmse.png',dpi=400)
fig.savefig(out/'development_rmse_preview.png',dpi=150)
plt.close(fig)
with (out/'development_rmse.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
caption='开发验证误差。每个频段包含 2,562 张图，来自 612 块训练集内留出的地形。按全部有效像素累计平方误差与像素数后开方。左图为 415 MHz，右图为 5.8 GHz。仅绘制已经完成并评分的 epoch。虚线为 1 dB 参考线。开发结果不是官方验证结果。\n\n'
caption+='\n'.join(f"{styles[cfg['mode']][0]}：{name}，已完成 {len(h)} 个 epoch。" for name,h,cfg in series)
(out/'图注.md').write_text(caption+'\n')
print(json.dumps({'runs':[name for name,_,_ in series],'last_epochs':[h[-1]['epoch'] for _,h,_ in series],'figure':str(out/'development_rmse.png')},ensure_ascii=False))
