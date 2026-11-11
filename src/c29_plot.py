"""Evidence figure for scale, map sharing and optimization diagnostics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axs=plt.subplots(2,2,figsize=(12,8.4),layout="constrained")
for name,label,color in [("c21r1_small_a_s20261119","Original 4 FIT maps","#555555"),("c29_hard_small_original_regime","4 difficult FIT maps","#0067a5")]:
 rows=[json.loads(s) for s in (R/"runs"/name/"history.jsonl").read_text().splitlines()]
 rows=[r for r in rows if r["event"]=="eval"]
 for b in [0,1]:
  ax=axs[0,b]
  ax.plot([r["step"] for r in rows],[r["rmse"][b] for r in rows],color=color,label=label,lw=2)
  ax.set_yscale("log");ax.set_ylim(.25,40)
for b in [0,1]:
 ax=axs[0,b];ax.set_xlim(0,5750);ax.set_title(["415 MHz","5.775 GHz"][b]+": same cold-start training recipe")
 ax.set_xlabel("Updates / visits per map");ax.set_ylabel("Full valid-pixel FIT RMSE (dB)")
 ax.axhline(1,color="#999999",ls="--",lw=1);ax.grid(alpha=.2);ax.legend(fontsize=9)
d=json.loads((R/"audit/c28_matched_exposure_20261101/full_comparison.json").read_text())
vals=np.array([d["models"][k]["anchor"]["rmse"] for k in ["anchor_only","joint","project_other"]])
x=np.arange(3);ax=axs[1,0]
for b,c in [(0,"#0067a5"),(1,"#d76714")]:
 bars=ax.bar(x+(b-.5)*.32,vals[:,b],width=.32,color=c,label=["415 MHz","5.775 GHz"][b])
 ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=9)
ax.set_xticks(x,["Focal only","Joint maps","Joint + projection"]);ax.set_ylim(0,2.5)
ax.set_title("Same 4 difficult maps, same 4500 visits\nC28: warm start, peak LR 0.0002")
ax.set_ylabel("Full valid-pixel focal FIT RMSE (dB)");ax.legend(fontsize=9);ax.grid(axis="y",alpha=.2)
d=json.loads((R/"audit/c29_fit_regime_20261101/full_comparison.json").read_text())
vals=np.array([d["models"][k]["rmse"] for k in ["low","mid","high"]]);ax=axs[1,1]
for b,c in [(0,"#0067a5"),(1,"#d76714")]:
 bars=ax.bar(x+(b-.5)*.32,vals[:,b],width=.32,color=c,label=["415 MHz","5.775 GHz"][b])
 ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=9)
ax.set_xticks(x,["0.0002","0.0005","0.001"]);ax.set_ylim(0,2.9)
ax.set_title("128 FIT maps: matched 3000-update LR control")
ax.set_xlabel("Peak learning rate");ax.set_ylabel("Full valid-pixel FIT RMSE (dB)")
ax.legend(fontsize=9);ax.grid(axis="y",alpha=.2)
fig.suptitle("Small-set fitting does not establish full-data performance\nTraining diagnostics only; official validation remains 2.967 / 3.388 dB",fontsize=14)
out=R/"results/C22-C29拟合诊断_20261101"
fig.savefig(str(out)+".png",dpi=160)
fig.savefig(str(out)+".jpg",dpi=105,pil_kwargs={"quality":85})
print(str(out)+".png")
