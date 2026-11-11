from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1];O=R/'audit/c4_remaining_error_20261031';F=O/'figures';F.mkdir(exist_ok=True)
s=json.loads((O/'statistics_summary.json').read_text());q=json.loads((O/'structure_audit.json').read_text());v=s['official_val'];tr=s['matched_train'];plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'axes.labelsize':7,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,'lines.linewidth':1,'savefig.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(183/25.4,125/25.4));fig.subplots_adjust(left=.085,right=.985,bottom=.145,top=.92,wspace=.32,hspace=.54);x=np.arange(2);w=.32;blue='#0072B2';orange='#D55E00';gray='#B8B8B8';yellow='#E69F00'
a=axes[0,0]
for shift,values,color,label in [(-w/2,tr['rmse'],gray,'Matched training'),(w/2,v['rmse'],blue,'Official validation')]:
 b=a.bar(x+shift,values,w,color=color,label=label)
 for z in b:a.text(z.get_x()+z.get_width()/2,z.get_height()+.09,f'{z.get_height():.2f}',ha='center',fontsize=7)
a.axhline(1,color=orange,ls='--',lw=.9);a.text(.98,.94,'Dashed line: 1 dB target',transform=a.transAxes,ha='right',color=orange,fontsize=6.5);a.set(xticks=x,xticklabels=['415 MHz','5.8 GHz'],ylabel='Masked RMSE (dB)',ylim=(0,6.7),title='A  Error is high on training data too');a.legend(loc='upper left',frameon=False)
a=axes[0,1];left=np.zeros(2)
for name,label,color in [('height_sensitive','Height-sensitive',yellow),('shallow_blocked','Shallow blocked',orange),('other','All other pixels',gray)]:
 if name=='other':values=100-left
 else:values=100*np.array(q['official_val']['subzones'][name]['sse_fraction'])
 a.barh(x,values,left=left,height=.45,color=color,label=label)
 for j,z in enumerate(values):a.text(left[j]+z/2,j,f'{z:.1f}%',ha='center',va='center',fontsize=7,color='white' if name=='shallow_blocked' else 'black')
 left+=values
a.set(yticks=x,yticklabels=['415 MHz','5.8 GHz'],xlim=(0,100),xlabel='Share of validation squared error (%)',title='B  Where the squared error accumulates');a.invert_yaxis();a.legend(loc='upper center',bbox_to_anchor=(.5,1.0),ncol=1,frameon=False);a.set_ylim(1.5,-1.25)
a=axes[1,0]
for shift,values,color,label in [(-w/2,v['rmse'],blue,'Actual'),(w/2,v['zones']['height_sensitive']['perfect_fix_here_remaining_rmse'],gray,'Perfect fix in sensitive region')]:
 b=a.bar(x+shift,values,w,color=color,label=label)
 for z in b:a.text(z.get_x()+z.get_width()/2,z.get_height()+.09,f'{z.get_height():.2f}',ha='center',fontsize=7)
a.axhline(1,color=orange,ls='--',lw=.9);a.set(xticks=x,xticklabels=['415 MHz','5.8 GHz'],ylabel='Masked RMSE (dB)',ylim=(0,6.7),title='C  Correcting this region alone is insufficient');a.legend(loc='upper left',frameon=False);a.text(.98,-.25,'Label-informed diagnostic; not a predictor',transform=a.transAxes,ha='right',fontsize=6)
a=axes[1,1];cs=q['official_val']['same_terrain_nearest_tx']['conditions']['geometric_deep_blocked_both'];truth=[cs[b]['excess_over_fspl_db']['target_delta_std'] for b in ['415','58']];pred=[cs[b]['excess_over_fspl_db']['predicted_delta_std'] for b in ['415','58']]
for shift,values,color,label in [(-w/2,truth,gray,'Label change'),(w/2,pred,blue,'Predicted change')]:
 b=a.bar(x+shift,values,w,color=color,label=label)
 for z in b:a.text(z.get_x()+z.get_width()/2,z.get_height()+.09,f'{z.get_height():.2f}',ha='center',fontsize=7)
a.set(xticks=x,xticklabels=['415 MHz','5.8 GHz'],ylabel='STD of excess-loss change (dB)',ylim=(0,8.5),title='D  Moving TX: changes are too small in prediction');a.legend(loc='upper left',frameon=False);a.text(.98,-.25,'Same terrain and RX; deeply blocked for both TX',transform=a.transAxes,ha='right',fontsize=6)
for a in axes.ravel():a.spines['top'].set_visible(False);a.spines['right'].set_visible(False);a.grid(axis='x' if a==axes[0,1] else 'y',color='#eeeeee',lw=.4);a.set_axisbelow(True)
fig.suptitle('Frozen C4 diagnosis | No parameter updates | Official validation: 2,330 maps per band',fontsize=8.5,y=.985)
for ext in ['pdf','svg','eps','png']:fig.savefig(F/f'c4_remaining_error.{ext}',dpi=400)
fig.savefig(F/'c4_remaining_error_preview.jpg',dpi=165)
(F/'figure_provenance.json').write_text(json.dumps({'source_json':['statistics_summary.json','structure_audit.json'],'font_file':font_manager.findfont('Arial'),'width_mm':183,'height_mm':125,'png_dpi':400,'definitions':'See report. C is a label-informed hypothetical regional correction; D uses fixed 1,024 RX points/map and correlated same-terrain TX pairs.'},indent=2)+'\n');print(str(F/'c4_remaining_error_preview.jpg'))
