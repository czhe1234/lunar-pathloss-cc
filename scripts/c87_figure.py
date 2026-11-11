"""Draw the two opposite errors using only original validation labels and frozen predictions."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,patheffects
from matplotlib.colors import TwoSlopeNorm
from c87_read import ROOT,OUT,maps
def main():
    data=json.load(open(OUT/'examples.json'))['clear_pair']
    assert data
    i,row,y,p,w=next(maps(lambda i:i==data['index']))
    chinese='Noto Sans CJK SC'
    from fontTools.ttLib import TTCollection
    for weight in ('Regular','Bold'):
        dest=ROOT/'.cache/matplotlib'/('NotoSansCJKSC-'+weight+'.otf')
        if not dest.exists():
            ttc=TTCollection('/usr/share/fonts/opentype/noto/NotoSansCJK-'+weight+'.ttc',lazy=True)
            face=next(f for f in ttc.fonts if f['name'].getDebugName(1)==chinese);face.save(str(dest))
        font_manager.fontManager.addfont(dest)
    font_manager.findfont(font_manager.FontProperties(family=chinese),fallback_to_default=False)
    plt.rcParams.update({'font.family':chinese,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
                       'figure.facecolor':'#ffffff','savefig.facecolor':'#ffffff','axes.edgecolor':'#7b8188','font.size':10.5})
    pts=[(k,data[k]['pixel']) for k in ('A','B')]
    cy,cx=np.rint(np.mean([rc for k,rc in pts],axis=0)).astype(int)
    x0=int(np.clip(cx-9,0,256-19));y0=int(np.clip(cy-9,0,256-19));x1=x0+19;y1=y0+19
    fig=plt.figure(figsize=(11.8,7.8))
    fig.text(.075,.95,'C82 把两个相邻接收点的信号强弱顺序预测反了',fontsize=21,weight='bold',color='#17212b')
    fig.text(.075,.904,f"验证样本 {data['sample_id']}     A、B 相距 {data['positions_distance_pixels']:.2f} 米     每格 1 米",fontsize=12,color='#49535f')
    gs=fig.add_gridspec(2,5,left=.08,right=.94,bottom=.12,top=.82,width_ratios=[1,1,.065,1,.065],wspace=.28,hspace=.23)
    mag=plt.get_cmap('viridis').copy();mag.set_bad('#dde0e3');div=plt.get_cmap('RdBu_r').copy();div.set_bad('#dde0e3')
    for b,band in enumerate(('415 MHz','5.775 GHz')):
        axes=[fig.add_subplot(gs[b,0]),fig.add_subplot(gs[b,1]),fig.add_subplot(gs[b,3])]
        fig.text(.016,.66-.376*b,band,ha='center',va='center',rotation=90,fontsize=13,weight='bold',color='#273440')
        vals=[y[b],p[b],p[b]-y[b]];norm=matplotlib.colors.Normalize(*([(68,88),(91,111)][b]))
        images=[]
        for j,(ax,v) in enumerate(zip(axes,vals)):
            view=v[y0:y1,x0:x1].copy();view[~w[b,y0:y1,x0:x1]]=np.nan
            im=ax.imshow(view,origin='upper',extent=(x0-.5,x1-.5,y1-.5,y0-.5),interpolation='nearest',cmap=mag if j<2 else div,
                         norm=norm if j<2 else TwoSlopeNorm(vmin=-12,vcenter=0,vmax=12))
            images.append(im)
            if b==0:ax.set_title(['验证标签','C82 预测','预测路损 − 标签'][j],loc='left',fontsize=13,pad=9,weight='bold')
            ax.set_xticks(np.arange((x0+4)//5*5,x1,5));ax.set_yticks(np.arange((y0+4)//5*5,y1,5))
            ax.tick_params(length=2.5,width=.6,labelsize=8.5,color='#6b747b')
            if b==1:ax.set_xlabel('列',fontsize=9,labelpad=2)
            if j==0:ax.set_ylabel('行',fontsize=9,labelpad=1)
            for k,(ry,rx) in pts:
                ax.scatter(rx,ry,s=45,marker='o' if k=='A' else 's',facecolor='none',edgecolor='white',linewidth=1.15,
                           path_effects=[patheffects.withStroke(linewidth=2.7,foreground='#222222')])
                ox,oy=(-4.0,4.2) if k=='A' else (3.25,-3.1)
                ax.annotate(k,xy=(rx,ry),xytext=(rx+ox,ry+oy),ha='center',va='center',color='white',fontsize=10,weight='bold',
                        bbox=dict(boxstyle='round,pad=.13',fc='#253442',ec='none',alpha=.93),
                        arrowprops=dict(arrowstyle='-',lw=.95,color='white',shrinkA=0,shrinkB=4,
                        path_effects=[patheffects.withStroke(linewidth=1.8,foreground='#253442')]))
        for j,im,label in [(2,images[1],'路损 dB'),(4,images[-1],'误差 dB')]:
            cb=fig.colorbar(im,cax=fig.add_subplot(gs[b,j]));cb.ax.tick_params(labelsize=8,length=2);cb.ax.set_xlabel(label,fontsize=8)
    fig.text(.075,.037,'圆圈 A 的信号实际更弱。方框 B 的信号实际更强。右栏蓝色表示 C82 把功率估高，红色表示估低。灰色像素不计分。',fontsize=10.5,color='#414b56')
    dest=ROOT/'results/C82官方验证误差_通视区实例.png';fig.savefig(dest,dpi=155);plt.close(fig);print(dest)
if __name__=='__main__':main()
