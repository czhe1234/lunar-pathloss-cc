from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];A=R/'audit/c129_fitgap'
summary=json.loads((A/'summary.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
colors={'n16':'#117733','n64':'#0072B2','w64':'#D55E00'}
names={'n16':'16 maps, 1.53M parameters','n64':'64 maps, 1.53M parameters','w64':'64 maps, 5.04M parameters'}
for b,title in enumerate(['415 MHz','5.775 GHz']):
 ax=axes[0,b]
 for tag in colors:
  history=json.loads((R/f'runs/c129_capacity_{tag}/history.json').read_text())
  ax.plot([x['exposures'] for x in history],[x['FIT']['rmse'][b] for x in history],label=names[tag],color=colors[tag],lw=2)
 ax.axhline(1,color='0.35',ls='--',lw=1);ax.set(title=f'{title}: FIT error',xlabel='Training passes per map',ylabel='Valid-pixel RMSE (dB)',ylim=(.65,2.5))
 ax.grid(alpha=.18)
axes[0,0].legend(fontsize=8)
ax=axes[1,0]
cases=['n16_ep250','n64_ep250','w64_ep250'];x=np.arange(3)
for b,(color,label) in enumerate([('#009E73','415 MHz'),('#CC79A7','5.775 GHz')]):
 vals=[summary['shared_anchor_probe'][k]['shared16anchor']['rmse'][b] for k in cases]
 ax.bar(x+(b-.5)*.32,vals,.32,color=color,label=label)
 for xx,yy in zip(x+(b-.5)*.32,vals):ax.text(xx,yy+.025,f'{yy:.3f}',ha='center',fontsize=8)
ax.set(xticks=x,xticklabels=['16 maps\nnarrow','64 maps\nnarrow','64 maps\nwide'],ylim=(0,1.8),ylabel='Valid-pixel RMSE (dB)',title='Same 16 FIT anchors, 250 passes per map')
ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18)
ax=axes[1,1];names2=['C82','16 maps\nnarrow','64 maps\nnarrow','64 maps\nwide'];x=np.arange(4)
for b,(color,label) in enumerate([('#009E73','415 MHz'),('#CC79A7','5.775 GHz')]):
 vals=[summary['monitor_C82_RMSE'][b]]+[summary['capacity'][k]['monitor']['RMSE'][b] for k in ['n16','n64','w64']]
 ax.bar(x+(b-.5)*.32,vals,.32,color=color,label=label)
 for xx,yy in zip(x+(b-.5)*.32,vals):ax.text(xx,yy+.035,f'{yy:.3f}',ha='center',fontsize=8)
ax.axhline(1,color='0.35',ls='--',lw=1)
ax.set(xticks=x,xticklabels=names2,ylim=(0,4.15),ylabel='Valid-pixel RMSE (dB)',title='Same 128-terrain monitor at FIT-selected endpoints')
ax.grid(axis='y',alpha=.18)
fig.suptitle('C129: Lower training error did not transfer to held-out terrains',fontsize=14)
for ax in axes.flat:
 ax.spines[['top','right']].set_visible(False)
out=R/'results/C129_fit_vs_transfer.png';fig.savefig(out,dpi=170);fig.savefig(out.with_suffix('.pdf'))
print(out)
