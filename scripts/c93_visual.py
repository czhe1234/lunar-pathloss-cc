"""One deterministic, non-tail example for each band, shown at true grid resolution."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']='Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus']=False
from c93_score import A,R
def main():
    tag='dense_probe';q=np.load(A/'patch_target.npz');pr=np.load(A/tag/'patch.npz')['pred'];f=np.load(A/'patch_features.npz')
    ids=q['ids'];p=q['c82'];phys=q['c92'];truth=q['y'];w=q['valid'];evt=q['weak']
    rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
    thresh=json.load(open(A/'dense_probe/choice.json'))['threshold'];ar=q['arm'];px=q['pixels'].reshape(-1,16,7,7)
    fig,axs=plt.subplots(2,3,figsize=(10.6,6.2),layout='constrained');report={}
    for b in [0,1]:
        ev=evt[:,:,b].reshape(-1,16,49);ok=w[:,:,b].reshape(-1,16,49);deficit=(truth[:,:,b]-np.maximum(p[:,:,b],phys[:,:,b])).reshape(-1,16,49)
        elig=(ar=='uniform')&(ev.sum(2)==1)&(ok.sum(2)>=40)
        vals=deficit[elig][ev[elig]];med=np.median(vals)
        idxs=np.column_stack(np.where(elig));m,t=idxs[np.argmin(abs(vals-med))]
        i=int(ids[m]);pselect=slice(t*49,(t+1)*49);e=ev[m,t].reshape(7,7);mask=ok[m,t].reshape(7,7)
        u,v=np.where(e);row=rows[i];band=('415 MHz','5.775 GHz')[b]
        excess=truth[m,pselect,b]-f['fs'][m,pselect,b];risk=pr[m,pselect,b,0];power=pr[m,pselect,b,2]
        fields=[excess.reshape(7,7),risk.reshape(7,7),power.reshape(7,7)]
        changes=px[m,t];yd,xd=divmod(int(changes[u[0],v[0]]),256)
        for c,(d,title,unit) in enumerate(zip(fields,['实测路损超过自由空间的部分','预测为共同漏点的概率','若为漏点，估计的路损'],['dB','概率','dB'])):
            ax=axs[b,c];d=np.ma.masked_where(~mask,d);img=ax.imshow(d,interpolation='nearest',cmap=('cividis','magma','cividis')[c],origin='upper')
            ax.scatter(v,u,s=116,facecolors='none',edgecolors='#29dce6',linewidths=1.8)
            if c==1:
                call=np.argwhere((fields[1]>=thresh[b])&mask);ax.scatter(call[:,1],call[:,0],marker='x',s=25,color='#ff9d30',linewidths=1.1)
            ax.set_title(title,fontsize=9);ax.set_xticks([0,3,6],[(changes[0,0]%256),(changes[0,3]%256),(changes[0,6]%256)]);ax.set_yticks([0,3,6],[changes[0,0]//256,changes[3,0]//256,changes[6,0]//256])
            ax.tick_params(labelsize=8);ax.set_xlabel('接收位置列号',fontsize=8);fig.colorbar(img,ax=ax,shrink=.78,label=unit,pad=.02)
            if c==0:ax.set_ylabel(f"{band}\n{row['sample_id']}\n接收位置行号",fontsize=9)
        k=int(u[0]*7+v[0]);report[band]=dict(sample=row['sample_id'],row=yd,col=xd,weakness_below_both=float(vals[np.argmin(abs(vals-med))]),truth=float(truth[m,t*49+k,b]),
                    c82=float(p[m,t*49+k,b]),c92=float(phys[m,t*49+k,b]),risk=float(risk[k]),threshold=thresh[b],conditional_PL=float(power[k]))
    fig.suptitle('同一块留出地形的两个频段，窗口为 7 × 7 像素\n圆圈为真实共同漏点，叉号为几何方法的报警',fontsize=13)
    out=R/'results/C93共同漏点相邻像素对照_20261103.png';fig.savefig(out,dpi=170);plt.close(fig)
    (A/'shown_examples.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
