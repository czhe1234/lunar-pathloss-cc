"""Plot saved C21r1 monitoring trajectories; no training or GPU work."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
R=Path(__file__).resolve().parents[1];run=R/'runs/c21r1_pair_s20261123';out=R/'results'
records=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines() if json.loads(l).get('event')=='monitor'];anchor=json.loads((R/'audit/c21_execution_20261101/C10_monitor_anchor.json').read_text());steps=[r['step'] for r in records]
fig,axes=plt.subplots(1,2,figsize=(11.6,4.6),sharex=True)
for b,ax in enumerate(axes):
 for k,(label,color) in enumerate([('A','#0072B2'),('B','#D55E00')]):
  for scope,style in [('FIT','--'),('DEV','-')]:
   ax.plot(steps,[r[scope][k]['rmse'][b] for r in records],style,marker='o' if scope=='DEV' else 's',markersize=4,color=color,label=f'{label} {scope}',linewidth=1.9)
 ax.axhline(anchor[['415','58'][b]]['rmse'],color='#555555',linestyle=':',linewidth=1.8,label='C10 DEV')
 ax.axhline(1,color='#009E73',linestyle='-.',linewidth=1.4,label='1 dB objective')
 ax.set_title(['415 MHz','5.775 GHz'][b]);ax.set_ylabel('Pooled masked RMSE (dB)');ax.set_xlabel('Optimizer steps per model');ax.set_ylim(.8,4.45);ax.grid(alpha=.22)
 ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{int(x/1000)}k'));ax.set_xticks([3000,6000,9000,12000,15000,18000]);ax.legend(fontsize=8,loc='lower left',bbox_to_anchor=(.01,.10),ncol=2)
fig.suptitle('C21r1 paired control: A content attention / B geometry bias',fontsize=13)
fig.text(.5,.012,'128 FIT + 128 DEV maps; 8,192 fixed queries per map; one matched seed. Complete DEV is scored separately.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.94));fig.savefig(out/'C21r1_monitor_trajectories.png',dpi=150);fig.savefig(out/'C21r1_monitor_trajectories.pdf');plt.close(fig)
print(out/'C21r1_monitor_trajectories.png')
