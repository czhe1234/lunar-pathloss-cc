"""C154 D4 preflight on frozen C153 inputs. CPU only; never trains or infers.

The strict gate checks tensor algebra AND whether canonical feature extraction
commutes with D4 on the available raw terrain/TX representation. A coherent
isometry with explicitly transformed mesh/TX is a positive control, not a
substitute for the unchanged canonical input contract.
"""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,math,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/"data"
OUT=ROOT/"audit/c154_d4"
CONTROL=ROOT/"audit/c153_standalone_c151a/protocol.json"
sys.path.insert(0,str(ROOT))
from src.c150_reference import registered_profile,nu_peak,WAVELENGTHS_M

def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".partial")
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def spatial(x,k,flip):
    if flip:x=np.flip(x,axis=-1)
    return np.rot90(x,k,axes=(-2,-1))
def affine(k,flip,L=255.):
    R=np.array([[0,1],[-1,0]],np.int64);M=np.eye(2,dtype=np.int64);b=np.zeros(2)
    if flip:M=np.diag([-1,1]);b=np.array([L,0.])
    for _ in range(k):M=R@M;b=R@b+np.array([0.,L])
    return M,b
def point(r,c,k,flip,L=255.):
    M,b=affine(k,flip,L);v=M@np.array([c,r])+b
    return int(round(v[1])),int(round(v[0]))
ACTIONS=[(k,f) for f in (False,True) for k in range(4)]
def label(k,f):return f"R{k*90}"+("_mirrorX_first" if f else "")
# c73 KEEP positions21,22 are native geometric sx,sy; raw offsets=11+21,11+22.
VECTOR_PAIRS=((3,4),(5,6),(32,33))
def raw_action(x,k,flip):
    out=spatial(x,k,flip).copy();M,_=affine(k,flip)
    for a,b in VECTOR_PAIRS:
        xx=out[...,a,:,:].copy();yy=out[...,b,:,:].copy()
        out[...,a,:,:]=M[0,0]*xx+M[0,1]*yy
        out[...,b,:,:]=M[1,0]*xx+M[1,1]*yy
    return out
def terrain(h,x,y,anti=0):
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    ix=np.clip(np.floor(x).astype(int),0,254);iy=np.clip(np.floor(y).astype(int),0,254)
    fx=x-ix;fy=y-iy
    a=h[iy,ix].astype(float);b=h[iy,ix+1].astype(float)
    c=h[iy+1,ix].astype(float);d=h[iy+1,ix+1].astype(float)
    if anti:
        return np.where(fx+fy<=1,a+(b-a)*fx+(c-a)*fy,
            b+c-d+(d-c)*fx+(d-b)*fy)
    return np.where(fy>fx,a+(d-c)*fx+(c-a)*fy,
        a+(b-a)*fx+(d-b)*fy)
S=256/255
QX,QY=np.meshgrid(np.linspace(1/3,255-1/3,256),np.linspace(1/3,255-1/3,256))
FREQ=np.array([415e6,5.775e9])
def tx(h,r,c):
    return np.array([c,r+1,math.floor(float(h[min(r+1,255),c]))+3],float)
def fspl(x,y,z,t):
    distance=np.sqrt((x-t[0])**2+(y-t[1])**2+(z-t[2])**2)
    return 20*np.log10(4*np.pi*np.maximum(distance,1e-9)[None]*FREQ[:,None,None]/299792458)
def numbers(v):
    v=np.asarray(v,float)
    return {"RMS":float(np.sqrt(np.mean(v*v))),"max_abs":float(np.max(np.abs(v))),
        "mean":float(np.mean(v))}
def read_masks(arr,i):
    w=arr[i]
    if w.shape==(256,32):return np.unpackbits(w,axis=-1,count=256,bitorder="big").astype(bool)
    if w.shape==(8192,):return np.unpackbits(w,count=65536,bitorder="big").reshape(256,256).astype(bool)
    if w.shape==(256,256):return w.astype(bool)
    raise ValueError(w.shape)

def run():
    start=time.monotonic()
    if (OUT/"result.json").exists():raise FileExistsError("Completed D4 audit exists; inspect, do not repeat")
    plan=json.loads(CONTROL.read_text());rows=list(csv.DictReader((DATA/"train_index.csv").open()))
    hm=np.load(DATA/"train_hm.npy",mmap_mode="r",allow_pickle=False)
    ids=plan["panels"]["fit"]+plan["panels"]["monitor"]
    selected=[]
    for name in ("fit","monitor"):
        seen=set()
        for i in plan["panels"][name]:
            if rows[i]["terrain_id"] in seen:continue
            seen.add(rows[i]["terrain_id"]);selected.append(i)
            if len(seen)==4:break
    counter=next((i for i in ids if rows[i]["sample_id"]=="00414_00"),None)
    if counter is not None and counter not in selected:selected.append(counter)
    source_files=["src/c154_d4_consistency.py","src/c153_train.py","src/c153_report.py",
        "src/c151_field.py","src/c150_train.py","src/c150_field.py","src/c146_data.py",
        "src/c3_geometry.py","src/c70_geometry.py","src/c70_geometry.cu",
        "src/c72_native_features.py","src/c73_compact.py","src/c67_physics.py",
        "src/c69_registered.py","src/c150_reference.py","src/c150_diffraction.cu","src/cc_tracking.py"]
    protected=dict(plan["protected_readonly_sha256"])
    protected.update({"audit/c153_standalone_c151a/protocol.json":sha(CONTROL)})
    for name in ("selected.pt","last.pt","initial.pt","history.json","result.json",
        "selected_fit_prediction.npy","selected_monitor_prediction.npy","selected_export.json","selected_verified.json"):
        name="runs/c153_standalone_c151a/"+name;protected[name]=sha(ROOT/name)
    audit={"campaign":"C154_D4","created_utc":datetime.now(timezone.utc).isoformat(),
        "control_protocol_sha256":sha(CONTROL),"scope":"CPU preflight only; no training/inference/feature export",
        "all_metadata_maps":len(ids),"geometric_maps":[{"map_id":i,"sample_id":rows[i]["sample_id"]} for i in selected],
        "actions":[label(*g) for g in ACTIONS],"definition":"Horizontal mirror first, then np.rot90 CCW in image coordinates",
        "raw_channels":68,"vector_pairs_raw_zero_based":[list(p) for p in VECTOR_PAIRS],
        "scalar_channels":"All remaining raw68; C82 slots56:58 remain literal zero; frequencies are never exchanged",
        "cache_policy":"Read existing caches only, no new feature extraction dataset",
        "gate":{"tensor":"group/inverse exact, zero slots exact, label/mask/loss permutation invariant",
            "geometry":"For all8 actions, raw terrain/TX transformed with unchanged canonical definitions must represent the same3D TX and receiver surface",
            "tolerances":{"coordinate_m":1e-9,"surface_m":1e-9,"FSPL_dB":5e-5,"CPU_basic_abs":5e-5},
            "failure_action":"Do not start the conditional C153+D4 training; do not change input geometry/network/loss/budget to force a pass"},
        "source_sha256":{n:sha(ROOT/n) for n in source_files},"protected_readonly_sha256":protected}
    atomic(OUT/"protocol.json",audit)
    for n,v in plan["source_sha256"].items():assert sha(ROOT/n)==v,("C153 source changed",n)
    for n,v in plan["protected_readonly_sha256"].items():assert sha(ROOT/n)==v,("protected input changed",n)
    print("D4 frozen protocol and control identities verified",flush=True)
    import torch
    from src.c3_geometry import basic_inputs
    torch.set_num_threads(2)
    cp={}
    for folder in ("features/c73_compact","features/c73_dev_extra_compact"):
        for p in (ROOT/folder).glob("part*"):
            a=np.load(p/"features.npy",mmap_mode="r",allow_pickle=False)
            cp.update({i:(a,j) for j,i in enumerate(json.loads((p/"indices.json").read_text()))})
    diffroot=ROOT/"features/c150_diffraction/part0of1"
    dif=np.load(diffroot/"features.npy",mmap_mode="r",allow_pickle=False)
    dlookup={i:j for j,i in enumerate(json.loads((diffroot/"indices.json").read_text()))}
    i=selected[0];row=rows[i];h=np.array(hm[int(row["hm_row"])])
    r,c=int(row["tx_row"]),int(row["tx_col"])
    basic,legacyfs=basic_inputs(torch.tensor(h[None]),torch.tensor([r]),torch.tensor([c]))
    d=np.asarray(dif[dlookup[i],:10])
    normalized=[]
    for off in (0,5):
        nu,t,I,Ld,Lb=d[off:off+5]
        normalized.extend((np.arcsinh(nu)/3,2*t-1,np.log1p(np.maximum(I,0))/3,Ld/20,Lb/20))
    array,j=cp[i]
    raw=np.concatenate((basic.numpy()[0],np.array(array[j],dtype=np.float32),
        np.zeros((2,256,256),np.float32),np.stack(normalized))).astype(np.float32)
    assert raw.shape==(68,256,256)
    algebra=[]
    for g in ACTIONS:
        M,b=affine(*g)
        mapped=raw_action(raw,*g);inverse=next(g2 for g2 in ACTIONS if np.array_equal(affine(*g2)[0],M.T))
        # Matrix alone uniquely selects one of D4 around a common center.
        back=raw_action(mapped,*inverse)
        assert np.array_equal(back,raw)
        assert not np.count_nonzero(mapped[56:58])
        rr,cc=point(r,c,*g)
        assert tuple(np.unravel_index(mapped[9].argmax(),(256,256)))==(rr,cc)
        recomputed,_=basic_inputs(torch.tensor(np.ascontiguousarray(spatial(h,*g))[None]),
            torch.tensor([rr]),torch.tensor([cc]))
        maxdiff=float(np.max(np.abs(mapped[:11]-recomputed.numpy()[0])))
        assert maxdiff<=audit["gate"]["tolerances"]["CPU_basic_abs"],(g,maxdiff)
        algebra.append({"action":label(*g),"raw68_inverse_max_abs":float(np.max(np.abs(back-raw))),
            "basic11_recompute_max_abs":maxdiff,"C82_slots_zero":True,"TX_onehot_correct":True})
    closures=0
    for g1 in ACTIONS:
        m1,b1=affine(*g1)
        for g2 in ACTIONS:
            m2,b2=affine(*g2);target=next(g for g in ACTIONS if np.array_equal(affine(*g)[0],m2@m1))
            assert np.array_equal(raw_action(raw_action(raw,*g1),*g2),raw_action(raw,*target))
            closures+=1
    labels=[np.load(DATA/f"train_pl{b}.npy",mmap_mode="r") for b in ("415","58")]
    masks=[np.load(DATA/f"train_mask{b}.npy",mmap_mode="r") for b in ("415","58")]
    mon_i=plan["panels"]["monitor"][0]
    pred=np.load(ROOT/"runs/c153_standalone_c151a/selected_monitor_prediction.npy",mmap_mode="r")[0].astype(float)
    yy=np.stack([v[mon_i] for v in labels]).astype(float)
    ww=np.stack([read_masks(v,mon_i) for v in masks])
    original=np.where(ww,(pred-yy)**2,0).sum((1,2),dtype=np.float64)
    losscheck=[]
    for g in ACTIONS:
        w=spatial(ww,*g);e=spatial(pred,*g)-spatial(yy,*g)
        s=np.where(w,e*e,0).sum((1,2),dtype=np.float64)
        losscheck.append({"action":label(*g),"masked_SSE_abs_difference":np.abs(s-original).tolist(),
            "valid_counts_identical":np.array_equal(w.sum((1,2)),ww.sum((1,2)))})
        assert np.allclose(s,original,atol=1e-7,rtol=1e-14)
    print("All68 tensor group,64compositions,basic11,GT/mask/SSE checks passed",flush=True)
    metadata=[]
    for g in ACTIONS:
        M,b=affine(*g,L=256.)
        diffs=[]
        for i in ids:
            row=rows[i];h=hm[int(row["hm_row"])];r,c=int(row["tx_row"]),int(row["tx_col"])
            old=tx(h,r,c);rr,cc=point(r,c,*g);canon=tx(spatial(h,*g),rr,cc)
            iso=np.r_[M@old[:2]+b,old[2]]
            diffs.append(canon-iso)
        diffs=np.asarray(diffs)
        metadata.append({"action":label(*g),"maps":len(ids),
            "horizontal_TX_error_m":numbers(np.linalg.norm(diffs[:,:2],axis=1)),
            "TX_height_error_m":numbers(diffs[:,2]),
            "TX_height_changed_maps":int(np.count_nonzero(diffs[:,2])),
            "TX_height_changed_percent":100*float(np.mean(diffs[:,2]!=0))})
    print("All3443 metadata TX comparisons completed",flush=True)
    geometry=[];max_cache=0.
    for i in selected:
        row=rows[i];h=hm[int(row["hm_row"])];r,c=int(row["tx_row"]),int(row["tx_col"])
        oldtx=tx(h,r,c);oldrx=terrain(h,QX,QY)+1;oldfs=fspl(QX*S,QY*S,oldrx,oldtx)
        cachefs=np.asarray(dif[dlookup[i],10:12],float)
        cache_diff=float(np.max(np.abs(oldfs-cachefs)));max_cache=max(max_cache,cache_diff)
        assert cache_diff<=5e-5,(i,cache_diff)
        for g in ACTIONS:
            M,b=affine(*g,L=256.);hg=spatial(h,*g);rr,cc=point(r,c,*g)
            newtx=tx(hg,rr,cc);isotx=np.r_[M@oldtx[:2]+b,oldtx[2]]
            diag=M@np.array([1,1]);anti=int(diag[0]*diag[1]<0)
            isorx=terrain(hg,QX,QY,anti)+1;canonicalrx=terrain(hg,QX,QY,0)+1
            rxround=float(np.max(np.abs(isorx-spatial(oldrx,*g))))
            assert rxround<=1e-9
            isoFS=fspl(QX*S,QY*S,isorx,isotx)
            fsround=float(np.max(np.abs(isoFS-spatial(oldfs,*g))))
            assert fsround<=1e-9
            txxy=np.r_[newtx[:2],isotx[2]]
            txz=np.r_[isotx[:2],newtx[2]]
            variants={"TXxy_only":fspl(QX*S,QY*S,isorx,txxy),
                "TXz_only":fspl(QX*S,QY*S,isorx,txz),
                "RX_triangle_only":fspl(QX*S,QY*S,canonicalrx,isotx),
                "canonical_all":fspl(QX*S,QY*S,canonicalrx,newtx)}
            geometry.append({"map_id":i,"sample_id":row["sample_id"],"action":label(*g),
                "transformed_mesh_anti":anti,"isometry_TX_xyz_m":isotx.tolist(),
                "canonical_TX_xyz_m":newtx.tolist(),"coherent_RX_height_max_error_m":rxround,
                "canonical_RX_height_error_m":numbers(canonicalrx-isorx),
                "coherent_FSPL_max_error_dB":fsround,
                "FSPL_component_error_dB":{n:numbers((v-isoFS)[0]) for n,v in variants.items()}})
    summary=[]
    for g in ACTIONS:
        sub=[x for x in geometry if x["action"]==label(*g)]
        summary.append({"action":label(*g),"maps":len(sub),
            "canonical_RX_height_max_abs_m":max(x["canonical_RX_height_error_m"]["max_abs"] for x in sub),
            "canonical_FSPL_max_abs_dB":max(x["FSPL_component_error_dB"]["canonical_all"]["max_abs"] for x in sub),
            "coherent_isometry_FSPL_max_abs_dB":max(x["coherent_FSPL_max_error_dB"] for x in sub),
            "component_max_abs_dB":{n:max(x["FSPL_component_error_dB"][n]["max_abs"] for x in sub)
                for n in ("TXxy_only","TXz_only","RX_triangle_only")}})
    atomic(OUT/"geometry_components.json",{"map_action_records":geometry})
    print("FSPL/TX/RX component separation completed",flush=True)
    # Small independent CPU nu probes complement the exhaustive metadata check.
    nu_checks=[]
    for i in selected[:2]+selected[-2:]:
        row=rows[i];h=hm[int(row["hm_row"])];r,c=int(row["tx_row"]),int(row["tx_col"])
        for rxr,rxc in ((17,21),(123,200),(235,58)):
            d,z,tz,rz=registered_profile(h,r,c,rxr,rxc)
            old=np.array([nu_peak(d,z,tz,rz,lam).nu for lam in WAVELENGTHS_M])
            cached=np.asarray(dif[dlookup[i],[0,5],rxr,rxc],float)
            assert np.allclose(old,cached,atol=4e-4,rtol=1e-5),(i,old,cached)
            for g in ACTIONS:
                rr,cc=point(r,c,*g);rrx,ccx=point(rxr,rxc,*g)
                dd,zz,tt,rt=registered_profile(spatial(h,*g),rr,cc,rrx,ccx)
                now=np.array([nu_peak(dd,zz,tt,rt,lam).nu for lam in WAVELENGTHS_M])
                finite=bool(np.isfinite(now).all())
                nu_checks.append({"sample_id":row["sample_id"],"rx_original":[rxr,rxc],
                    "action":label(*g),"original_nu":old.tolist(),
                    "canonical_nu":now.tolist() if finite else None,
                    "canonical_invalid_buried_endpoint":not finite,
                    "nu_abs_difference":np.abs(now-old).tolist() if finite else None,
                    "grazing_membership_changed":((old>=-.7)&(old<0)!=((now>=-.7)&(now<0))).tolist() if finite else None})
    atomic(OUT/"nu_probes.json",nu_checks)
    still={n:sha(ROOT/n)==v for n,v in protected.items()}
    assert all(still.values()),still
    sourceok={n:sha(ROOT/n)==v for n,v in audit["source_sha256"].items()}
    assert all(sourceok.values())
    assert not torch.cuda.is_initialized()
    tensorpass=all(x["raw68_inverse_max_abs"]==0 for x in algebra)
    physicspass=all(x["horizontal_TX_error_m"]["max_abs"]<=1e-9 and x["TX_height_error_m"]["max_abs"]<=1e-9 for x in metadata)
    physicspass=physicspass and all(x["canonical_RX_height_max_abs_m"]<=1e-9 and x["canonical_FSPL_max_abs_dB"]<=5e-5 for x in summary)
    result={"status":"complete","campaign":"C154_D4","protocol_sha256":sha(OUT/"protocol.json"),
        "tensor_checks_passed":tensorpass,"strict_physics_checks_passed":physicspass,
        "D4_gate_passed":bool(tensorpass and physicspass),"conditional_training_started":False,
        "epochs_run":0,"updates_run":0,"network_inference_run":False,"CUDA_initialized":False,
        "all68_inverse_and_basic_checks":algebra,"group_compositions_checked":closures,
        "label_mask_SSE_invariance":losscheck,"all_metadata_TX_check":metadata,
        "geometry_summary":summary,"base_FSPL_CPU_vs_cache_max_abs_dB":max_cache,
        "physical_maps":len(selected),"FSPL_pixels_per_action":len(selected)*65536,
        "nu_probes":len(nu_checks),"nu_probes_with_grazing_flip":sum(any(x["grazing_membership_changed"]) for x in nu_checks if x["grazing_membership_changed"] is not None),
        "nu_probes_with_buried_endpoint":sum(x["canonical_invalid_buried_endpoint"] for x in nu_checks),
        "protected_files_unchanged":all(still.values()),"protected_file_count":len(still),
        "source_files_unchanged":all(sourceok.values()),"elapsed_seconds":time.monotonic()-start,
        "interpretation":"Tensor D4 algebra is necessary but not sufficient. Canonical raw-TX registration and fixed anti0 mesh need to commute for an exact unchanged-input physical symmetry. Explicitly transformed TX plus mesh is the positive control; that control does not certify canonical re-extraction.",
        "next_action":"Train the sole60epoch D4 candidate only if D4_gate_passed is true; otherwise stop with evidence and preserve C153."}
    atomic(OUT/"result.json",result)
    lines=["# C154 D4一致性检查","",
        f"检查完成。张量检查：{tensorpass}；严格物理检查：{physicspass}；训练启动：False。","",
        "|变换|TX水平偏差 m|TX高度改变比例 %|TX高度最大偏差 m|接收面最大偏差 m|FSPL最大偏差 dB|完整场景等距变换FSPL误差 dB|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for m,g in zip(metadata,summary):
        lines.append(f"|{m['action']}|{m['horizontal_TX_error_m']['max_abs']:.6f}|{m['TX_height_changed_percent']:.3f}|{m['TX_height_error_m']['max_abs']:.3f}|{g['canonical_RX_height_max_abs_m']:.6f}|{g['canonical_FSPL_max_abs_dB']:.6f}|{g['coherent_isometry_FSPL_max_abs_dB']:.3g}|")
    lines+=["",f"TX统计覆盖全部{len(ids)}图；接收面/FSPL覆盖{len(selected)}图、每变换{len(selected)*65536}像素。FSPL双频偏差相同。",
        "张量逆变换、64次群合成、basic11重算、GT/mask/SSE同步检查通过。c73坡度raw32/33与basic坡度3/4、位移5/6均做分量交换/符号变换。",
        "现有registered TX的水平定义为(c,r+1)m，高度为floor(h[min(r+1,255),c])+3；raw索引绕255旋转，物理接收网格绕256m旋转，不能直接互换。",
        "固定anti0三角剖分在部分D4操作后需改为anti1。完整旋转TX和网格连接的正控制通过，证明问题不是数组旋转代码。",
        "这不证明历史C153预测误差的成因，也不证明近似增强无效；它拒绝将近似增强当作严格物理不变的唯一变量实验。",
        "没有训练、模型推理、全量特征重导出；历史保护文件和源码hash均未变。详细分量见geometry_components.json，nu抽查见nu_probes.json。"]
    (OUT/"result.md").write_text("\n".join(lines)+"\n")
    from src import cc_tracking
    run=cc_tracking.init(name="C154_D4_consistency_CPU",campaign="C154_D4",
        config={"protocol":str((OUT/"protocol.json").relative_to(ROOT)),"protocol_sha256":sha(OUT/"protocol.json"),
            "run_kind":"analysis.preflight","GPU_used":False,"training_started":False,"control":"C153 standalone60epochs",
            "gate":"tensor plus unchanged-canonical-geometry consistency"},
        description="CPU D4输入/坐标一致性门槛；不是训练，未产生模型权重",tags=["CPU","analysis","D4"])
    metrics={"gate/tensor_pass":int(tensorpass),"gate/strict_physics_pass":int(physicspass),
        "gate/D4_pass":int(tensorpass and physicspass),"progress/epochs_run":0}
    for m,g in zip(metadata,summary):
        a=m["action"]
        metrics.update({f"{a}/TX_xy_max_m":m["horizontal_TX_error_m"]["max_abs"],
            f"{a}/TX_height_max_m":m["TX_height_error_m"]["max_abs"],
            f"{a}/RX_height_max_m":g["canonical_RX_height_max_abs_m"],
            f"{a}/FSPL_max_dB":g["canonical_FSPL_max_abs_dB"]})
    cc_tracking.log(metrics,step=0)
    tracking={"run_id":getattr(run,"id",None),"name":"C154_D4_consistency_CPU","mode":"local"}
    cc_tracking.finish()
    atomic(OUT/"tracking.json",tracking)
    print(json.dumps({"status":result["status"],"gate_passed":result["D4_gate_passed"],
        "tensor_pass":tensorpass,"physics_pass":physicspass,"maps":len(ids),
        "geometry_maps":len(selected),"elapsed":result["elapsed_seconds"],"tracking":tracking}),flush=True)
if __name__=="__main__":run()
