"""A measured TRAIN-held-out counterexample. Same H/TX, nearby RX locations."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.c121_pairs import Reader
rd=Reader();data=json.load(open(R/'audit/c121_pairs/pair_report.json'));e=data['examples'][0];i=e['mapid'];b=e['band']
h,tr,tc=rd.scene(i);ar,ac=e['anchor_row'],e['anchor_col'];br,bc=e['opposite_row'],e['opposite_col']
rr=slice(ar-5,ar+6);cc=slice(ac-5,ac+6);truth=np.array(rd.Y[b][i],dtype=float);pred=np.array(rd.pred(i)[b],dtype=float)
valid=np.unpackbits(np.asarray(rd.M[b][i])).reshape(256,256)>0;truth[~valid]=np.nan;pred[~valid]=np.nan
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','savefig.facecolor':'white'})
seq=LinearSegmentedColormap.from_list('muted',['#f3f1eb','#b6c3b8','#53756d','#294d48']);seq.set_bad('#dddcd8')
fig,axes=plt.subplots(1,3,figsize=(12,4.8));fig.subplots_adjust(left=.06,right=.95,bottom=.27,top=.82,wspace=.42)
extent=[ac-5.5,ac+5.5,ar+5.5,ar-5.5]
shared_min=float(np.nanmin(truth[rr,cc]));shared_max=float(np.nanmax(truth[rr,cc]))
for k,(ax,z,title) in enumerate(zip(axes,[h[rr,cc]-h[ar,ac],truth[rr,cc],pred[rr,cc]],['Released terrain: relative height','True pathloss','C82 predicted pathloss'])):
 kw={'vmin':shared_min,'vmax':shared_max} if k else {}
 im=ax.imshow(z,origin='upper',extent=extent,cmap=seq,interpolation='nearest',**kw);ax.set_title(title,pad=9);ax.set_xlabel('Receiver column');ax.set_ylabel('Receiver row' if k==0 else '')
 ax.set_xticks([ac-4,ac,ac+4]);ax.set_yticks([ar-4,ar,ar+4])
 for label,r,c,color in [('A',ar,ac,'#976d3c'),('B',br,bc,'#715c82')]:
  ax.plot(c,r,'o',ms=7,mfc=color,mec='white',mew=1.4);tx=ax.text(c+.55,r-.35,label,color='white',weight='bold',fontsize=11);tx.set_path_effects([pe.withStroke(linewidth=2,foreground='#333333')])
 cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.ax.set_title('m' if k==0 else 'dB',fontsize=9,pad=7)
fig.suptitle('Same terrain and TX; opposite corrections one diagonal pixel apart',fontsize=15,y=.98)
fig.text(.5,.90,f'TRAIN-held-out map {i} | 5.775 GHz | TX row {tr}, column {tc} | 7 x 7 terrain-patch difference: {e["patch7_delta_rms"]*100:.2f} cm RMS',ha='center',fontsize=10,color='#555555')
y=np.array(e['target']);p=np.array(e['base']);res=y-p
fig.text(.07,.14,f'A  ({ar}, {ac})    True {y[0,b]:.3f} dB    C82 {p[0,b]:.3f} dB    Needed correction {res[0,b]:+.3f} dB',fontsize=11,color='#73502a')
fig.text(.07,.085,f'B  ({br}, {bc})    True {y[1,b]:.3f} dB    C82 {p[1,b]:.3f} dB    Needed correction {res[1,b]:+.3f} dB',fontsize=11,color='#604a72')
fig.text(.07,.025,'True and predicted pathloss share the same color scale. Higher pathloss means weaker received power.',fontsize=9,color='#666666')
out=R/'results/C121_pair_9435.png';fig.savefig(out,dpi=180,bbox_inches='tight',pad_inches=.12);plt.close(fig)
json.dump({'source':'audit/c121_pairs/pair_report.json; published TRAIN H/PL; audit/c88_mesh/base.npy','script':'scripts/c121_figure.py','export':str(out),'surface':'internal_review','claim':'Same H/TX and similar local terrain do not imply the same correction direction. Does not prove missing raw inputs.','review':'pending'},open(R/'audit/c121_pairs/figure_record.json','w'),indent=2)
