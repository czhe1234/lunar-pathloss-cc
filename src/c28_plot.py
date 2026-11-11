"""C28 diagnostic trajectories; all curves include every recorded monitor."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
fig,axes=plt.subplots(1,3,figsize=(14,4.1),constrained_layout=True)
styles=[("anchor_only","Focal maps only","#222222"),("joint","Focal + other maps","#1565c0"),("project_other","Joint + gradient projection","#d85b16")]
for run,label,color in styles:
 rows=[json.loads(s) for s in (R/"runs"/("c28_"+run)/"history.jsonl").read_text().splitlines()]
 rows=[r for r in rows if r["event"]=="monitor"]
 x=[r["step"] for r in rows]
 for b in [0,1]:
  axes[b].plot(x,[r["anchor"]["rmse"][b] for r in rows],label=label,color=color,lw=2,marker="o",markersize=3)
 axes[2].plot(x,[r["other"]["score"] for r in rows],label=label,color=color,lw=2,marker="o",markersize=3)
for b in [0,1]:
 axes[b].set_title("Same 4 difficult FIT maps: "+["415 MHz","5.775 GHz"][b])
 axes[b].axhline(1,color="#777777",ls="--",lw=1)
 axes[b].set_ylabel("Masked RMSE (dB)")
axes[2].set_title("Remaining 124 FIT maps")
axes[2].set_ylabel("Mean of two band RMSEs (dB)")
for ax in axes:
 ax.set_xlabel("Updates / visits per focal map");ax.grid(alpha=.22);ax.set_xlim(left=0)
 axes[0].legend(fontsize=8,loc="upper right")
fig.suptitle("C28: equal focal-map exposure, same initial model and sampled pixels\nFixed 8192 evaluation pixels/map; FIT diagnostics, not official validation",fontsize=12)
dest=R/"results/C28相同曝光次数对照_20261101.png"
fig.savefig(dest,dpi=160)
print(dest)
