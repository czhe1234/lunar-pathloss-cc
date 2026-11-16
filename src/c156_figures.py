"""C156 server-side scientific plots from saved case arrays, CPU Agg only."""
from pathlib import Path
import json
import numpy as np
from src.c156_data import ROOT,OUT,load,sha
def render():
 import matplotlib;matplotlib.use("Agg")
 from matplotlib import pyplot as plt,font_manager,colors
 plt.style.use(OUT/"academic.mplstyle")
 font="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";font_manager.fontManager.addfont(font)
 plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":[font_manager.FontProperties(fname=font).get_name(),"DejaVu Sans"],
  "axes.unicode_minus":False,"axes.titlesize":12,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10})
 cmap=colors.LinearSegmentedColormap.from_list("cc_pathloss",["#f3eee5","#c9cdbb","#94a8a2","#627f88","#3c535f"])
 palette={"GT":"#333b42","C82":"#9a8a78","C155":"#477f75","Deygout":"#a7776c"}
 data=load(OUT/"result.json");manifest=[]
 for case in data["top3_regions"]:
  arrays=np.load(ROOT/case["data"],allow_pickle=False)
  mask=arrays["mask"];rows,cols=np.where(mask)
  lo,hi=max(0,int(rows.min())-8),min(256,int(rows.max())+9)
  left,right=max(0,int(cols.min())-8),min(256,int(cols.max())+9)
  sl=np.s_[lo:hi,left:right];anchor=case["profile_anchor"];valid=arrays["valid"]
  fig=plt.figure(figsize=(13.6,10.3),layout="constrained")
  gs=fig.add_gridspec(3,4,height_ratios=[1,1,.92])
  for b,band in enumerate(("415 MHz","5.775 GHz")):
   allvalues=np.concatenate([arrays[k][b][sl][valid[b][sl]].ravel() for k in ("GT","C82","C155","Deygout")])
   vmin=float(allvalues.min());vmax=float(allvalues.max());axes=[]
   for j,(key,title) in enumerate((("GT","GT"),("C82","C82"),("C155","C155 · Epoch 52"),("Deygout","FSPL + L_Deygout"))):
    ax=fig.add_subplot(gs[b,j]);axes.append(ax)
    img=np.ma.array(arrays[key][b][sl],mask=~valid[b][sl])
    im=ax.imshow(img,cmap=cmap,vmin=vmin,vmax=vmax,extent=[left-.5,right-.5,hi-.5,lo-.5],interpolation="nearest")
    outline=arrays["common_tail"][b][sl]
    if outline.any() and not outline.all():
     xx,yy=np.meshgrid(np.arange(left,right),np.arange(lo,hi))
     ax.contour(xx,yy,outline,levels=[.5],colors="white",linewidths=1.2)
     ax.contour(xx,yy,outline,levels=[.5],colors="#3f4850",linewidths=.35)
    ax.scatter([anchor["col"]],[anchor["row"]],c="white",s=33,marker="x",linewidths=1.4)
    ax.set_title(title)
    ax.set_xlabel("RX 列（pixel）")
    if j==0:ax.set_ylabel(band+"\nRX 行（pixel）")
    else:ax.tick_params(labelleft=False)
    ax.grid(False)
   cb=fig.colorbar(im,ax=axes,fraction=.024,pad=.015);cb.set_label("路损（dB）")
  ax=fig.add_subplot(gs[2,:2])
  dist=arrays["profile_distance_m"];h=arrays["profile_height_m"];ray=arrays["profile_direct_ray_m"]
  ax.plot(dist,h,color="#7c887b",label="地形")
  ax.plot(dist,ray,color="#668596",linestyle="--",label="TX–RX 直线")
  ax.fill_between(dist,h,ray,where=h>ray,color="#c1a493",alpha=.5)
  ax.scatter([dist[0],dist[-1]],[ray[0],ray[-1]],color="#4b5964",s=20)
  ax.set_xlabel("沿剖面的水平距离（m）");ax.set_ylabel("高度（m）")
  ax.set_title(f"典型 RX 的地形剖面：n={case['profile_crossings']}，遮挡段={case['profile_blocked_segments']}")
  ax.legend(loc="best",fontsize=10)
  ax=fig.add_subplot(gs[2,2:])
  b=0 if case["driving_band"]=="415MHz" else 1;rr=anchor["row"]
  x=np.arange(left,right);fs=arrays["FSPL"][b,rr,left:right]
  for key,title in (("GT","GT"),("C82","C82"),("C155","C155"),("Deygout","FSPL + L_Deygout")):
   vals=arrays[key][b,rr,left:right]-fs
   ax.plot(x,np.where(valid[b,rr,left:right],vals,np.nan),color=palette[key],label=title,linewidth=1.6)
  ax.axvline(anchor["col"],color="#a9a296",linestyle=":",linewidth=1)
  ax.set_xlabel(f"经过典型 RX 的栅格行：row={rr}");ax.set_ylabel("路损 − FSPL（dB）")
  ax.set_title(("415 MHz" if b==0 else "5.775 GHz")+"：同一行的额外衰减");ax.legend(loc="best",fontsize=9)
  fig.suptitle(f"共同大错区域 {case['rank']} · {case['sample_id']} · 联合 SSE 排名\n白线：共同大错边界；×：剖面接收点（同一位置、原始预测）",fontsize=15)
  file=OUT/f"region_{case['rank']}.png";fig.savefig(file,dpi=160);plt.close(fig)
  manifest.append({"rank":case["rank"],"source_data":case["data"],"script":"src/c156_figures.py","export":str(file.relative_to(ROOT)),
   "surface_class":"internal_review","claim":"Frozen Top3 common-error fields and a representative current-geometry profile",
   "render_inspected":False,"script_sha256":sha(Path(__file__))})
 (OUT/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
 return manifest
if __name__=="__main__":print(render())
