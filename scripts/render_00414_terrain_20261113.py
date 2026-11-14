"""Render the published 00414_00 height grid, without training or GPU use."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import base64
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/terrain_00414_3d_20261113"
rows = list(csv.DictReader((ROOT.parent / "data/train_index.csv").open()))
ix = next(i for i, r in enumerate(rows) if r["sample_id"] == "00414_00")
rec = rows[ix]
h = np.array(np.load(ROOT.parent / "data/train_hm.npy", mmap_mode="r")[int(rec["hm_row"])], dtype="<f4")
assert h.shape == (256, 256) and np.isfinite(h).all()
tr, tc = int(rec["tx_row"]), int(rec["tx_col"])
markers = [
    dict(label="TX 发射点", row=tr, col=tc, agl=3.0),
    dict(label="RX 核查接收点", row=23, col=38, agl=1.0),
]
for m in markers:
    m["ground"] = float(h[m["row"], m["col"]])
    m["height"] = m["ground"] + m["agl"]
raw = h.tobytes()
data = dict(sample="00414_00", index=ix, height_row=int(rec["hm_row"]),
            zmin=float(h.min()), zmax=float(h.max()),
            heights_base64=base64.b64encode(raw).decode("ascii"), markers=markers)
template = (OUT / "terrain-template.html").read_text()
assert template.count("__TERRAIN_DATA__") == 1
html = template.replace("__TERRAIN_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
assert len(html.encode()) < 1_000_000
(OUT / "terrain-00414.html").write_text(html)

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                     "font.size": 11})
xx, yy = np.meshgrid(np.arange(256), np.arange(256))
fig = plt.figure(figsize=(11, 7.2), facecolor="white")
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
shade = LightSource(315, 35).shade(h, cmap=plt.get_cmap("gist_earth"),
                                  vert_exag=1, blend_mode="soft")
ax.plot_surface(xx, yy, h, facecolors=shade, rstride=1, cstride=1,
                linewidth=0, antialiased=False, shade=False, rasterized=True, zorder=1)
for m, color, symbol in zip(markers, ("#b83b30", "#246caa"), ("D", "o")):
    ax.plot([m["col"]]*2, [m["row"]]*2, [m["ground"], m["height"]], color=color, lw=2, zorder=5)
    ax.scatter(m["col"], m["row"], m["height"], s=65, marker=symbol,
               color=color, edgecolor="white", linewidth=.8, depthshade=False, zorder=6)
    ax.text(m["col"], m["row"], m["height"]+2, m["label"], fontsize=11, color=color, zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=2))
ax.set(xlim=(0,255), ylim=(0,255), zlim=(0,32),
       xlabel="列索引", ylabel="行索引", zlabel="高度（米）")
ax.set_xticks([0,64,128,192,255]); ax.set_yticks([0,64,128,192,255])
ax.set_zticks([0,10,20,30])
ax.set_box_aspect((255,255,32))
ax.view_init(elev=33, azim=-60)
ax.xaxis.labelpad=17; ax.yaxis.labelpad=17; ax.zaxis.labelpad=8
ax.tick_params(labelsize=9)
ax.set_title("00414_00 · 由发布高度图恢复的三维地形", pad=10, fontsize=16)
fig.text(.5,.045,"全部 256×256 高度点 · 高度 0–27.97 米 · TX 名义离地 3 米，RX 名义离地 1 米",
         ha="center", fontsize=11, color="#444444")
fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.91)
fig.savefig(OUT / "terrain-00414.png", dpi=180, bbox_inches="tight")
plt.close(fig)

decoded = np.frombuffer(base64.b64decode(data["heights_base64"]), dtype="<f4").reshape(256,256)
assert np.array_equal(decoded, h)
record = dict(sample=rec, data_index=ix, grid_shape=list(h.shape),
              height_min=float(h.min()), height_max=float(h.max()),
              height_sha256=hashlib.sha256(raw).hexdigest(),
              all_65536_heights_preserved=bool(np.array_equal(decoded,h)),
              horizontal_axes="published grid column/row indices",
              markers=markers, gpu_used=False,
              outputs=["terrain-00414.html","terrain-00414.png"])
(OUT / "source.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("sample",)},ensure_ascii=False))
print("html_bytes",len(html.encode()))
