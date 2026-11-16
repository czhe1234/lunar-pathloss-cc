"""C153 phase1: new Grazing roughness and metadata-group scores, CPU only."""
from pathlib import Path
from datetime import datetime, timezone
import csv, json, os, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.c152_los_diagnostic import Rough, Stats, box, nine, read, sha, atomic
from src.c145_verify import validate_export
OUT = ROOT / "audit/c153_grazing_groups"
RUN = ROOT / "runs/c145_capacity_native"
BANDS = ("415MHz", "5p775GHz")

def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This diagnostic requires CUDA_VISIBLE_DEVICES=''")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"result.json").exists():
        raise FileExistsError("Completed diagnostic exists; reuse rather than repeat")
    started = time.monotonic()
    old = read(ROOT/"audit/c149_error_audit/summary.json")
    plan = read(ROOT/"audit/c149_error_audit/protocol.json")
    ids = plan["ordered_panel_ids"]
    assert ids == read(ROOT/"audit/c145_capacity/protocol.json")["panels"]["monitor"]
    assert len(ids) == 628
    predpath = RUN/"selected_monitor_prediction.npy"
    identity = validate_export(RUN, "selected", {"monitor": ids}, {"monitor": predpath})
    assert identity["model_sha256"] == old["model_sha256"]
    profpath = ROOT/"audit/c149_error_audit/profiles.npy"
    assert sha(profpath) == old["profile_cache_sha256"]
    raw = ROOT.parent/"data"
    with (raw/"train_index.csv").open() as f:
        rows = list(csv.DictReader(f))
    fdir = ROOT/"features/c150_diffraction/part0of1"
    complete = read(fdir/"complete.json")
    assert complete["status"] == "complete"
    lookup = {int(i): j for j, i in enumerate(read(fdir/"indices.json"))}
    fs = np.load(fdir/"features.npy", mmap_mode="r")
    pred = np.load(predpath, mmap_mode="r")
    profiles = np.load(profpath, mmap_mode="r")
    yall = [np.load(raw/f"train_pl{b}.npy", mmap_mode="r") for b in ("415", "58")]
    wall = [np.load(raw/f"train_mask{b}.npy", mmap_mode="r") for b in ("415", "58")]
    audit = read(ROOT/"audit/raw_audit.json")
    for name in ("train_pl415.npy", "train_pl58.npy", "train_mask415.npy", "train_mask58.npy", "train_index.csv"):
        st=(raw/name).stat(); known=audit["files"][name]
        assert (st.st_size, st.st_mtime_ns)==(known["size_bytes"],known["mtime_ns"]), name
    protected = {str(p.relative_to(ROOT)): sha(p) for p in (
        RUN/identity["checkpoint"], RUN/"selected_export.json", RUN/"selected_monitor_prediction.json",
        ROOT/"src/c152_los_diagnostic.py", ROOT/"audit/c152_los_diagnostic/result.json")}
    rough = {k: Rough() for k in ("all9_Grazing_valid", "center_Grazing_all9_valid")}
    stats = {k: Stats() for k in ("all", "single", "multi", "Grazing")}
    images = {k: 0 for k in ("single", "multi")}
    terrains = {k: set() for k in images}
    wavelength = np.asarray(old["wavelength_m"], np.float64)[:,None,None]
    for j,i in enumerate(ids):
        row=rows[i]; group=row["group"]
        assert group in images
        images[group]+=1; terrains[group].add(row["terrain_id"])
        y=np.stack([a[i] for a in yall]).astype(np.float64)
        w=np.stack([np.unpackbits(a[i],bitorder="big").reshape(256,256) for a in wall]).astype(bool)
        p=np.asarray(pred[j],np.float64)
        f=np.asarray(fs[lookup[i],10:12],np.float64)
        h=np.asarray(profiles[j,0])
        nu=np.asarray(profiles[j,1],np.float64)[None]/np.sqrt(wavelength)
        grazing=w&(h[None]<0)&(nu>=-.7)&(nu<0)
        assert np.isfinite(y[w]).all() and np.isfinite(p[w]).all() and np.isfinite(f).all()
        e=p-y
        stats["all"].add(e,w);stats[group].add(e,w);stats["Grazing"].add(e,grazing)
        valid9=nine(w)&np.isfinite(box(y))&np.isfinite(box(p))
        rough["all9_Grazing_valid"].add(y-f,p-f,nine(grazing)&valid9)
        rough["center_Grazing_all9_valid"].add(y-f,p-f,grazing[:,1:-1,1:-1]&valid9)
        if (j+1)%100==0:
            print(json.dumps({"stage":"CPU_statistics","maps":j+1,"total_maps":len(ids)}),flush=True)
    assert sum(images.values())==628
    assert len(terrains["single"]|terrains["multi"])==128
    for key, oldkey in (("all",None),("Grazing","Fresnel_Grazing")):
        reference=old["all"] if oldkey is None else old["groups"][oldkey]
        assert np.array_equal(stats[key].n,reference["valid_pixels"])
        assert np.allclose(stats[key].se,reference["SSE_dB2"],rtol=1e-12,atol=1e-5)
    assert np.array_equal(stats["single"].n+stats["multi"].n,stats["all"].n)
    assert np.allclose(stats["single"].se+stats["multi"].se,stats["all"].se,rtol=1e-12)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    result={
        "status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),
        "scope":{"maps":628,"terrains":128},"band_order":list(BANDS),
        "training_run":False,"network_inference_run":False,"geometry_run":False,
        "model_weights_modified":False,"torch_imported":"torch" in sys.modules,
        "definitions":{"error":"pred-GT","Grazing":"valid & highest<0 & -0.7<=nu_max(f)<0",
            "nu":"C149 nu_lambda1m / sqrt(lambda_m)",
            "FSPL":"existing C150 Float32 registered3D FSPL planes10:12",
            "highpass":"v(center)-equal mean of nine3x3 pixels; no padding",
            "primary_neighborhood":"all nine pixels valid and all nine belong to Grazing",
            "secondary_neighborhood":"center belongs to Grazing and all nine pixels valid",
            "group":"train_index.csv row by frozen global map index, column group",
            "pixel_denominator":"all-valid monitor pixels, separately for each frequency",
            "group_score":"pixel-pooled SSE/count, no training image weights",
            "decomposition":"HP/LP not orthogonal; retain cross term for exact MSE closure"},
        "roughness":{k:v.result() for k,v in rough.items()},
        "groups":{k:{"images":images[k],"terrains":len(terrains[k]),
            **stats[k].result(stats["all"],stats["all"])} for k in images},
        "all":stats["all"].result(stats["all"],stats["all"]),
        "validation":{"original_C149_all_and_Grazing_counts_SSE_match":True,
            "group_partition_exact":True,"protected_sources_unchanged":True},
        "source_identity":{"source_sha256":sha(Path(__file__)),"prediction":identity["panels"]["monitor"],
            "profile_sha256":sha(profpath),"train_index_sha256":sha(raw/"train_index.csv"),
            "FSPL_cache_sha256_recorded":complete["features_sha256"],"protected":protected},
        "elapsed_seconds":time.monotonic()-started}
    for v in result["roughness"].values():
        assert max(abs(x) for x in v["decomposition_error_dB2"])<1e-8
    assert not result["torch_imported"]
    atomic(OUT/"result.json",result)
    lines=["# C153第一阶段：CPU统计","","邻域主定义：3×3九像素全部有效且均为Grazing。单位dB。",
        "","|邻域|频段|有效中心数|GT-FSPL高频RMS|Pred-FSPL高频RMS|高频误差RMS|局部均值误差RMS|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name,v in result["roughness"].items():
        for j,b in enumerate(BANDS):
            keys=("GT_minus_FSPL_highpass_RMS_dB","Pred_minus_FSPL_highpass_RMS_dB","highpass_error_RMS_dB","localmean_error_RMS_dB")
            vals="|".join(f"{v[k][j]:.6f}" for k in keys)
            lines.append(f"|{name}|{b}|{v['count'][j]}|{vals}|")
    lines+=["","|group|频段|图像数|有效像素数|占该频段有效像素%|Bias dB|RMSE dB|","|---|---|---:|---:|---:|---:|---:|"]
    for name,v in result["groups"].items():
        for j,b in enumerate(BANDS):
            lines.append(f"|{name}|{b}|{v['images']}|{v['count'][j]}|{v['pixel_percent_all'][j]:6f}|{v['MeanBias_dB'][j]:+.6f}|{v['RMSE_dB'][j]:.6f}|")
    (OUT/"result.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"complete","path":str(OUT/"result.json"),"elapsed_seconds":result["elapsed_seconds"]}),flush=True)
if __name__=="__main__":main()
