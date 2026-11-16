"""C155 immutable Grazing support and CPU statistics from existing predictions."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,os,time
import numpy as np
from src.c152_los_diagnostic import Rough,nine
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/"data";AUDIT=ROOT/"audit/c155_unet"
KEYS=("GT_hp2","Pred_hp2","error_hp2","error_localmean2")
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".partial")
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def identity(p):
    st=Path(p).stat();return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]
def support_path(name):return AUDIT/f"{name}_grazing9_packed.npy"
def box(v):return sum(v[...,dy:dy+254,dx:dx+254] for dy in range(3) for dx in range(3))/9.
def finish_stats(count,sums):
    names={"GT_hp2":"GT_minus_FSPL_highpass_RMS_dB","Pred_hp2":"Pred_minus_FSPL_highpass_RMS_dB",
        "error_hp2":"highpass_error_RMS_dB","error_localmean2":"localmean_error_RMS_dB"}
    n=np.asarray(count,np.int64)
    assert np.all(n>0)
    return {"count":n.tolist(),**{names[k]:np.sqrt(np.asarray(sums[k],float)/n).tolist() for k in KEYS}}
def raw_arrays():
    return ([np.load(DATA/f"train_pl{b}.npy",mmap_mode="r") for b in ("415","58")],
        [np.load(DATA/f"train_mask{b}.npy",mmap_mode="r") for b in ("415","58")])
def diffraction():
    root=ROOT/"features/c150_diffraction/part0of1"
    a=np.load(root/"features.npy",mmap_mode="r");ix=load(root/"indices.json")
    return a,{int(i):j for j,i in enumerate(ix)}

def prepare():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU preparation only")
    marker=AUDIT/"grazing_support.json"
    if marker.exists():
        d=load(marker);assert d["status"]=="complete"
        for name in ("fit","monitor"):assert sha(support_path(name))==d["supports"][name]["sha256"]
        print(json.dumps({"event":"reuse_completed_Grazing_support"}));return
    AUDIT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    plan=load(ROOT/"audit/c153_standalone_c151a/protocol.json")
    rows=list(csv.DictReader((DATA/"train_index.csv").open()))
    for name,h in plan["source_sha256"].items():assert sha(ROOT/name)==h,name
    control_root=ROOT/"runs/c153_standalone_c151a"
    ctrl_export=load(control_root/"selected_export.json");ctrl_result=load(control_root/"result.json")
    assert ctrl_result["status"]=="complete" and ctrl_result["selected_epoch"]==12
    assert ctrl_export["model_sha256"]==ctrl_result["selected_predictor_sha256"]
    protected=load(ROOT/"audit/c154_d4/protocol.json")["protected_readonly_sha256"]
    assert load(ROOT/"audit/c154_d4/result.json")["protected_files_unchanged"]
    identities={n:identity(ROOT/n) for n in protected}
    labels,masks=raw_arrays();diff,lookup=diffraction()
    profiles=np.load(ROOT/"audit/c149_error_audit/profiles.npy",mmap_mode="r")
    assert load(ROOT/"audit/c149_error_audit/protocol.json")["ordered_panel_ids"]==plan["panels"]["monitor"]
    waves=np.asarray(load(ROOT/"audit/c149_error_audit/summary.json")["wavelength_m"])[:,None,None]
    support_meta={};control={}
    raw_audit=load(ROOT/"audit/raw_audit.json")
    raw_files=["train_pl415.npy","train_pl58.npy","train_mask415.npy","train_mask58.npy","train_index.csv"]
    for n in raw_files:
        st=(DATA/n).stat();old=raw_audit["files"][n]
        assert (st.st_size,st.st_mtime_ns)==(old["size_bytes"],old["mtime_ns"]),n
    for name in ("fit","monitor"):
        ids=plan["panels"][name]
        pred=np.load(control_root/f"selected_{name}_prediction.npy",mmap_mode="r")
        packed=np.lib.format.open_memmap(support_path(name),mode="w+",dtype=np.uint8,shape=(len(ids),2,254,32))
        rough=Rough();all_sse=np.zeros(2);all_count=np.zeros(2,np.int64)
        count9=np.zeros(2,np.int64);rounding_diff=np.zeros(2,np.int64)
        for j,i in enumerate(ids):
            y=np.stack([a[i] for a in labels]).astype(np.float64)
            w=np.stack([np.unpackbits(a[i],bitorder="big").reshape(256,256) for a in masks]).astype(bool)
            nu=np.asarray(diff[lookup[i],[0,5]],np.float64)
            grazing=w&(nu>=-.7)&(nu<0)
            if name=="monitor":
                reference=np.asarray(profiles[j,1],np.float64)[None]/np.sqrt(waves)
                exact=w&(np.asarray(profiles[j,0])[None]<0)&(reference>=-.7)&(reference<0)
                rounding_diff+=(grazing!=exact).sum((1,2));grazing=exact
            valid9=nine(grazing);count9+=valid9.sum((1,2))
            packed[j]=np.packbits(valid9,axis=-1,bitorder="big")
            fs=np.asarray(diff[lookup[i],10:12],np.float64)
            p=np.asarray(pred[j],np.float64)
            rough.add(y-fs,p-fs,valid9)
            e=p-y;all_sse+=np.where(w,e*e,0).sum((1,2),dtype=np.float64);all_count+=w.sum((1,2))
            if (j+1)%200==0 or j+1==len(ids):
                print(json.dumps({"event":"Grazing_CPU_prepare","panel":name,"maps":j+1,"total":len(ids)}),flush=True)
        packed.flush();del packed
        rmse=np.sqrt(all_sse/all_count)
        expected=next(h for h in load(control_root/"history.json") if h["epoch"]==12)["evaluation"][name]["rmse"]
        assert np.allclose(rmse,expected,rtol=0,atol=1e-9)
        rr=rough.result()
        control[name]={"rmse":rmse.tolist(),"grazing":rr,"checkpoint_epoch":12}
        support_meta[name]={"path":str(support_path(name).relative_to(ROOT)),"sha256":sha(support_path(name)),
            "shape":[len(ids),2,254,32],"bitorder":"big","unpack_count":254,"count":count9.tolist(),
            "C150_vs_C149_rounding_membership_difference_pixels":rounding_diff.tolist()}
    assert all(identity(ROOT/n)==v for n,v in identities.items())
    d={"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-start,
        "definition":"all9 valid AND all9 highest<0 AND all9 -.7<=nu_max(f)<0; no padding; val=GT-FSPL or Pred-FSPL",
        "FIT_nu_source":"Frozen C150 Float32 nu; nu_max<0 iff highest<0 on valid registered profiles, since the Fresnel scale is positive. No geometry rerun.",
        "monitor_nu_source":"Exact existing C149 Float32 nu_lambda1m / sqrt(lambda_m) and highest; matches prior C153 phase1.",
        "supports":support_meta,"C153_selected12":control,"control_predictor_sha256":ctrl_export["model_sha256"],
        "control_protocol_sha256":sha(ROOT/"audit/c153_standalone_c151a/protocol.json"),
        "protected_readonly_sha256_from_C154":protected,"protected_file_identities_before_after_unchanged":identities,
        "cache_file_identities":{str(p.relative_to(ROOT)):identity(p) for p in [ROOT/"features/c150_diffraction/part0of1/features.npy",
            ROOT/"features/c150_diffraction/part0of1/indices.json",ROOT/"features/c150_diffraction/part0of1/complete.json"]},
        "source_sha256":sha(Path(__file__)),"training_run":False,"network_inference_run":False,
        "new_statistic_only":"C153 selected12 Grazing FIT/monitor statistics computed from saved predictions; old training not rerun"}
    atomic(marker,d)
    print(json.dumps({"event":"Grazing_CPU_preparation_complete","elapsed":d["elapsed_seconds"],"supports":support_meta}),flush=True)

def score_saved(run,tag):
    data=load(AUDIT/"grazing_support.json");plan=load(AUDIT/"protocol.json")
    labels,masks=raw_arrays();diff,lookup=diffraction();output={}
    for name in ("fit","monitor"):
        ids=plan["panels"][name];pred=np.load(run/f"{tag}_{name}_prediction.npy",mmap_mode="r")
        supports=np.load(support_path(name),mmap_mode="r");rough=Rough()
        for j,i in enumerate(ids):
            y=np.stack([a[i] for a in labels]).astype(np.float64);p=np.asarray(pred[j],np.float64)
            fs=np.asarray(diff[lookup[i],10:12],np.float64)
            valid=np.unpackbits(supports[j],axis=-1,count=254,bitorder="big").astype(bool)
            rough.add(y-fs,p-fs,valid)
            if (j+1)%400==0:print(json.dumps({"event":"Grazing_CPU_verify","tag":tag,"panel":name,"maps":j+1}),flush=True)
        output[name]=rough.result()
        assert output[name]["count"]==data["supports"][name]["count"]
    return output

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    if args.prepare:prepare()
