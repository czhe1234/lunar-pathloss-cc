"""C149: attribute saved selected12 monitor error; never run/train a network.

Geometry is an audit-only exact ray/triangle calculation under the existing
registered convention. Existing c73 descriptors remain untouched.
"""
from pathlib import Path
import argparse, csv, ctypes, hashlib, json, math, os, subprocess, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"audit/c149_error_audit"
RUN=ROOT/"runs/c145_capacity_native"
FREQ=np.array([415e6,5.775e9],np.float64)
WAVELENGTH=299792458.0/FREQ
BANDS=("415MHz","5p775GHz")
APPROVED={
"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
"GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
"GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"}
BUCKETS=("Clear","Grazing","Shallow_Shadow","Deep_Shadow","undefined_horizontal_distance")

def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n");tmp.replace(path)

def read(path):return json.loads(Path(path).read_text())

def source_record(path):return {"path":str(Path(path).relative_to(ROOT)),"sha256":digest(path)}

def load_inputs():
    from src.c145_verify import validate_export
    plan=read(ROOT/"audit/c145_capacity/protocol.json")
    ids=list(plan["panels"]["monitor"])
    assert len(ids)==628 and len(set(ids))==628
    identity=validate_export(RUN,"selected",{"monitor":ids},{"monitor":RUN/"selected_monitor_prediction.npy"})
    verified=read(RUN/"selected_verified.json")
    assert identity["model_sha256"]==verified["export_identity"]["model_sha256"]
    assert identity["model_sha256"]=="2f13de5e344462264b9184d20ab920d89d645899c2820f0799e334947cb55cd7"
    prediction=np.load(RUN/"selected_monitor_prediction.npy",mmap_mode="r")
    with (ROOT.parent/"data/train_index.csv").open() as f:rows=list(csv.DictReader(f))
    data=ROOT.parent/"data"
    labels=[np.load(data/f"train_pl{band}.npy",mmap_mode="r") for band in ("415","58")]
    masks=[np.load(data/f"train_mask{band}.npy",mmap_mode="r") for band in ("415","58")]
    h=np.load(data/"train_hm.npy",mmap_mode="r")
    old=read(ROOT/"audit/c88_mesh/base_index.json")["ids"]
    base=np.load(ROOT/"audit/c88_mesh/base.npy",mmap_mode="r")
    base_lookup={int(i):j for j,i in enumerate(old)}
    physical={}
    for folder in ("c73_compact","c73_dev_extra_compact"):
        for directory in sorted((ROOT/"features"/folder).glob("part*")):
            if not (directory/"complete.json").exists():continue
            values=np.load(directory/"features.npy",mmap_mode="r")
            for j,i in enumerate(read(directory/"indices.json")):
                if int(i) in ids:
                    if int(i) in physical:raise ValueError("Duplicate complete c73 cache")
                    physical[int(i)]=(values,j)
    assert all(i in physical and i in base_lookup for i in ids)
    expected=verified["panels"]["monitor"]["groups"]["all"]
    assert expected["valid_pixels"]==[40864973,37837148]
    terrain=sorted({rows[i]["terrain_id"] for i in ids})
    assert len(terrain)==128
    return dict(plan=plan,ids=ids,identity=identity,verified=verified,prediction=prediction,
                rows=rows,labels=labels,masks=masks,h=h,base=base,base_lookup=base_lookup,
                physical=physical,expected=expected,terrains=terrain)

def surface(h,x,y):
    ix=min(254,max(0,int(math.floor(x))));iy=min(254,max(0,int(math.floor(y))))
    fx=x-ix;fy=y-iy
    a,b,c,d=map(float,(h[iy,ix],h[iy,ix+1],h[iy+1,ix],h[iy+1,ix+1]))
    gx,gy=(d-c,c-a) if fy>fx else (b-a,d-b)
    return a+gx*fx+gy*fy

def profile_reference(h,tr,tc,r,c):
    """Independent CPU oracle: all lattice/diagonal intersection parameters."""
    x=tc*255/256;y=(tr+1)*255/256;z=math.floor(float(h[min(tr+1,255),tc]))+3
    rx=1/3+c*(255-2/3)/255;ry=1/3+r*(255-2/3)/255
    rz=surface(h,rx,ry)+1;dx=rx-x;dy=ry-y;dz=rz-z
    distance=(256/255)*math.hypot(dx,dy);scale=math.sqrt(2/distance) if distance>1e-10 else 0
    cuts=[0.,1.]
    for origin,direction,values in [(x,dx,range(1,255)),(y,dy,range(1,255)),(x-y,dx-dy,range(-254,255))]:
        if abs(direction)>1e-16:
            cuts += [(k-origin)/direction for k in values if 0<(k-origin)/direction<1]
    cuts=np.unique(cuts);highest=-math.inf;nu=-math.inf;arg=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        mid=(lo+hi)/2;px=x+dx*mid;py=y+dy*mid
        ix=min(254,max(0,int(math.floor(px))));iy=min(254,max(0,int(math.floor(py))))
        a,b,c0,d=map(float,(h[iy,ix],h[iy,ix+1],h[iy+1,ix],h[iy+1,ix+1]))
        gx,gy=(d-c0,c0-a) if py-iy>px-ix else (b-a,d-b)
        intercept=a+gx*(x-ix)+gy*(y-iy)-z;slope=gx*dx+gy*dy-dz
        candidates=[lo,hi]
        if abs(slope+2*intercept)>1e-20:
            critical=intercept/(slope+2*intercept)
            if lo<critical<hi:candidates.append(critical)
        for t in candidates:
            excess=intercept+slope*t;highest=max(highest,excess)
            if 0<t<1:
                value=excess*scale/math.sqrt(t*(1-t))
                if value>nu:nu=value;arg=t
    e0=surface(h,x,y)-z
    flags=(int(distance<1e-10)|2*int(e0>0)|4*int(e0==0))
    if e0>0:nu=math.inf;arg=0.
    elif e0==0 and nu<0:nu=0.;arg=0.
    if distance<1e-10:nu=0.;arg=0.
    return np.array([highest,nu,arg,e0,flags],float)

class Profiles:
    def __init__(self):
        uuid=os.environ.get("CUDA_VISIBLE_DEVICES")
        assert uuid in APPROVED and os.environ.get("CC_GPU_UUID")==uuid,"One complete approved UUID required"
        import torch
        from src.cc_core import gpu_guard, check_source_unchanged
        self.torch=torch;self.gpu=gpu_guard();check_source_unchanged()
        library=OUT/"libc149_profiles.so"
        source=ROOT/"src/c149_profiles.cu"
        binary_meta=OUT/"kernel_build.json"
        if not library.exists() or not binary_meta.exists() or read(binary_meta).get("source_sha256")!=digest(source):
            cmd=["/usr/local/cuda-11.8/bin/nvcc","-O3","--shared","--cudart=shared","-Xcompiler","-fPIC","-Xlinker","-rpath","-Xlinker","/usr/local/cuda-11.8/lib64","-gencode","arch=compute_80,code=compute_80",str(source),"-o",str(library)]
            subprocess.run(cmd,check=True,cwd=ROOT)
            write(binary_meta,{"source_sha256":digest(source),"command":cmd,"library_sha256":digest(library)})
        self.lib=ctypes.CDLL(str(library))
        self.fn=self.lib.c149_profiles
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        self.fn.restype=ctypes.c_int
    def __call__(self,h,tr,tc):
        t=self.torch
        values=t.tensor(np.ascontiguousarray(h),dtype=t.float32,device="cuda").contiguous()
        output=t.empty((5,256,256),dtype=t.float64,device="cuda")
        error=self.fn(values.data_ptr(),int(tr),int(tc),output.data_ptr(),t.cuda.current_stream().cuda_stream)
        if error:raise RuntimeError(f"Geometry kernel launch failed: {error}")
        result=output.cpu().numpy()
        if np.any(result[4].astype(np.int64)&8):raise RuntimeError("C149 ray failed to traverse the scene")
        return result

def profile_checks(engine,inputs):
    checks=[]
    flat=np.zeros((256,256),np.float32)
    ridge=np.broadcast_to(np.maximum(0,12-np.abs(np.arange(256)-128)),(256,256)).astype(np.float32)
    sloped=(.03*np.arange(256)[None]+.02*np.arange(256)[:,None]).astype(np.float32)
    scenarios=[("flat",flat,80,40),("ridge",ridge,100,80),("slope",sloped,70,60)]
    for name,h,tr,tc in scenarios:
        result=engine(h,tr,tc)
        for r,c in ((17,21),(123,200),(235,58),(100,170)):
            reference=profile_reference(h,tr,tc,r,c)
            value=result[:,r,c]
            if not np.allclose(value,reference,atol=2e-8,rtol=2e-9,equal_nan=True):
                raise AssertionError((name,r,c,value,reference))
            checks.append({"scene":name,"row":r,"col":c,"max_absolute_difference":float(np.max(np.abs(value-reference)))})
    flat_value=engine(flat,80,40)
    assert np.allclose(flat_value[0],-1.,atol=1e-10)
    assert np.allclose(flat_value[2],.75,atol=1e-10)
    for i in inputs["ids"][:2]:
        row=inputs["rows"][i];h=inputs["h"][int(row["hm_row"])]
        tr,tc=int(row["tx_row"]),int(row["tx_col"]);result=engine(h,tr,tc)
        for r,c in ((27,31),(193,201),(100,170)):
            reference=profile_reference(h,tr,tc,r,c);value=result[:,r,c]
            assert np.allclose(value,reference,atol=5e-8,rtol=5e-9,equal_nan=True),(i,r,c,value,reference)
            checks.append({"scene":f"raw_map_{i}","row":r,"col":c,"max_absolute_difference":float(np.max(np.abs(value-reference)))})
    return checks

class Stats:
    def __init__(self):
        self.count=np.zeros(2,np.int64);self.sse=np.zeros(2);self.base_sse=np.zeros(2)
    def add(self,error,base_error,valid):
        if valid.ndim==2:valid=np.broadcast_to(valid,error.shape)
        self.count += valid.sum((1,2),dtype=np.int64)
        self.sse += np.where(valid,error*error,0).sum((1,2),dtype=np.float64)
        self.base_sse += np.where(valid,base_error*base_error,0).sum((1,2),dtype=np.float64)
    def value(self,total):
        safe=lambda x,d:np.divide(x,d,out=np.full(2,np.nan),where=d>0)
        return {"valid_pixels":self.count,"pixel_fraction":safe(self.count,total.count),
                "RMSE_dB":np.sqrt(safe(self.sse,self.count)),"SSE_dB2":self.sse,
                "share_of_total_SSE_percent":100*safe(self.sse,total.sse),
                "C82_RMSE_dB":np.sqrt(safe(self.base_sse,self.count)),
                "C82_SSE_dB2":self.base_sse,
                "C82_share_of_total_SSE_percent":100*safe(self.base_sse,total.base_sse)}

class Correlation:
    def __init__(self):self.values=np.zeros(6,np.float64)
    def add(self,error,joint):
        x,y=error[:,joint]
        self.values += [len(x),x.sum(),y.sum(),np.dot(x,x),np.dot(y,y),np.dot(x,y)]
    def value(self):
        n,sx,sy,sxx,syy,sxy=self.values
        vx=sxx-sx*sx/max(n,1);vy=syy-sy*sy/max(n,1)
        return {"valid_intersection_pixels":int(n),"Pearson_signed_error":(sxy-sx*sy/n)/math.sqrt(vx*vy) if n>0 and vx>0 and vy>0 else None,
                "error_definition":"saved_prediction - original TRAIN label (dB)"}

def audit(inputs,profiles,tracking):
    total=Stats()
    groups={}
    def add(name,condition,error,base_error,valid):
        if name not in groups:groups[name]=Stats()
        groups[name].add(error,base_error,valid & condition)
    terrain={name:Stats() for name in inputs["terrains"]}
    selected_corr=Correlation();base_corr=Correlation()
    t0=time.monotonic()
    for j,i in enumerate(inputs["ids"]):
        label=np.stack([a[i] for a in inputs["labels"]]).astype(np.float64)
        valid=np.stack([np.unpackbits(a[i],bitorder="big").reshape(256,256) for a in inputs["masks"]]).astype(bool)
        pred=np.asarray(inputs["prediction"][j],np.float64)
        base=np.asarray(inputs["base"][inputs["base_lookup"][i]],np.float64)
        error=pred-label;base_error=base-label
        total.add(error,base_error,valid)
        terrain[inputs["rows"][i]["terrain_id"]].add(error,base_error,valid)
        p=np.asarray(profiles[j]);highest=p[0];nu=p[1][None]/np.sqrt(WAVELENGTH[:,None,None]);flags=p[4].astype(np.int64)
        defined=(flags&1)==0
        add("profile_LoS",highest<0,error,base_error,valid);add("profile_NLoS",highest>=0,error,base_error,valid)
        for name,condition in zip(BUCKETS,[defined&(nu<-.7),defined&(nu>=-.7)&(nu<0),defined&(nu>=0)&(nu<1.5),defined&(nu>=1.5),~defined]):
            add("Fresnel_"+name,condition,error,base_error,valid)
        ar,pos=inputs["physical"][i]
        cache=np.array(ar[pos,[8,16,25,26,31,32,39,40,41,42]],dtype=np.float32)
        proxy_los=cache[0]<0
        add("cached_proxy_LoS",proxy_los,error,base_error,valid);add("cached_proxy_NLoS",~proxy_los,error,base_error,valid)
        add("profile_vs_cache_LoS_disagreement",proxy_los!=(highest<0),error,base_error,valid)
        add("cached_mixed_LoS_mass",(cache[1]>0)&(cache[1]<1),error,base_error,valid)
        direct=cache[6:8]<=-3.;power_total=cache[8:10]<=-3.
        for name,hit in [("direct_linear_mean",direct),("total_linear_mean",power_total),
                         ("direct_or_total_linear_mean",direct|power_total),
                         ("direct_mean_log",cache[2:4]<=-3.),("total_mean_log",cache[4:6]<=-3.)]:
            add("floor_"+name,hit,error,base_error,valid);add("above_floor_"+name,~hit,error,base_error,valid)
        add("selected12_error_abs_ge_5dB",np.abs(error)>=5,error,base_error,valid)
        add("selected12_error_abs_lt_5dB",np.abs(error)<5,error,base_error,valid)
        add("C82_error_abs_ge_5dB",np.abs(base_error)>=5,error,base_error,valid)
        add("TX_below_reconstructed_terrain",(flags&2)>0,error,base_error,valid)
        add("TX_on_reconstructed_terrain",(flags&4)>0,error,base_error,valid)
        selected_corr.add(error,valid.all(0));base_corr.add(base_error,valid.all(0))
        if (j+1)%16==0 or j+1==len(inputs["ids"]):
            elapsed=time.monotonic()-t0
            progress={"stage":"scoring_saved_predictions","maps":j+1,"total":len(inputs["ids"]),"elapsed_seconds":elapsed,"eta_seconds":elapsed/(j+1)*(len(inputs["ids"])-j-1)}
            write(OUT/"progress.json",progress)
            tracking.log({"progress/maps":j+1,"progress/total_maps":len(inputs["ids"]),"time/elapsed_seconds":elapsed,"time/eta_seconds":progress["eta_seconds"]},step=len(inputs["ids"])+j+1)
            print(json.dumps(progress),flush=True)
    overall=total.value(total)
    assert np.array_equal(total.count,inputs["expected"]["valid_pixels"])
    assert np.allclose(overall["RMSE_dB"],inputs["expected"]["RMSE_dB"],rtol=0,atol=1e-10)
    assert np.allclose(total.sse,inputs["expected"]["SSE_dB2"],rtol=1e-12,atol=1e-6)
    assert np.allclose(overall["C82_RMSE_dB"],[2.1463905073492664,2.4180144849255294],rtol=0,atol=1e-10)
    for names in [("profile_LoS","profile_NLoS"),("cached_proxy_LoS","cached_proxy_NLoS"),tuple("Fresnel_"+s for s in BUCKETS),("selected12_error_abs_ge_5dB","selected12_error_abs_lt_5dB")]:
        assert np.array_equal(sum((groups[n].count for n in names)),total.count)
        assert np.allclose(sum((groups[n].sse for n in names)),total.sse,atol=1e-6,rtol=1e-12)
    terrains=[]
    for name,value in terrain.items():
        v=value.value(total);v["terrain"]=name
        v["joint_mean_MSE_dB2"]=float(np.mean(value.sse/value.count))
        terrains.append(v)
    rank=sorted(terrains,key=lambda x:x["joint_mean_MSE_dB2"],reverse=True)
    result={"status":"complete","scope":"C145 selected12 fixed TRAIN-internal monitor: 628 maps / 128 terrains; repeatedly used development set, not blind/official VAL",
            "prediction_checkpoint":"runs/c145_capacity_native/best_monitor.pt","model_sha256":inputs["identity"]["model_sha256"],
            "training_or_network_inference_run":False,"frequency_hz":FREQ,"wavelength_m":WAVELENGTH,
            "all":overall,"groups":{k:v.value(total) for k,v in groups.items()},
            "cross_band_residual_correlation":{"selected12":selected_corr.value(),"C82":base_corr.value()},
            "worst_10_terrains_joint_MSE":rank[:10],
            "worst_10_terrains_per_band":{BANDS[k]:sorted(terrains,key=lambda x:x["RMSE_dB"][k],reverse=True)[:10] for k in range(2)},
            "geometry_convention":{"mesh":"anti=0 piecewise planar, heightmap lattice0..255","spacing_m":256/255,
                "precision":"Float64 analytic kernel then Float32 cache; near-threshold nu buckets use stored Float32 values",
                "TX":"x=tc*255/256; y=(tr+1)*255/256; z=floor(h[min(tr+1,255),tc])+3",
                "RX":"queryaxis linspace(1/3,255-1/3,256); z=piecewise-planar terrain(query)+1",
                "highest":"exact maximum terrain minus direct-ray height over current reconstructed piecewise-planar profile",
                "nu":"supremum e(t)*sqrt(2/(lambda*d2D*t*(1-t))) on open t in(0,1); triangle-boundary plus analytic interior extrema",
                "frequency_scaling":"same profile argmax for both wavelengths; neither materials nor solver powers enter calculation",
                "buried_TX":"e(0)>0 gives +infinite open-interval nu supremum, recorded in Deep_Shadow and separate source-below-terrain group",
                "cache_proxy":"c73[8] is weighted mean of asinh(highest/.25)/3 over9-ray quadrature then face interpolation, not a point-ray maximum; c73[16] is the interpolated LoS fraction"},
            "floor_definition":{"primary_channels":{"direct":[39,40],"total":[41,42]},"secondary_mean_log_channels":{"direct":[25,26],"total":[31,32]},
                "native_quantity":"Q=P_component*d_face^2, normalized log10(Q)/4; Q clamped at1e-12 before compacting",
                "floor_code":-3.0,"observability":"Reports floor-coded Float16 compact values, not exact raw preclamp power saturation or absolute P floor. Subface saturation may occur even when interpolated aggregate is above floor.",
                "floor_code_underlying_Q_interval_most_compact_extra":[1e-12,1e-12*10**(4*2**-10)],
                "Float16_half_ULP_log_code":2**-10},
            "tail_definitions":{"requested":"abs(selected12 prediction - label)>=5dB","old_C82_tail":"abs(frozen C82 - label)>=5dB, separately reported"},
            "MSE_share_definition":"groupSSE / totalSSE per band; equals pixel_fraction*groupMSE/overallMSE; groups within each declared partition sum100%",
            "official_VAL_used":False,"limitations":["LoS and Fresnel refer to the declared current reconstructed scene; c73 mixes receiver face samples and is not the same point geometry.",
                "Residual correlations use only the intersection of both frequency validity masks.",
                "Clamped Float16 compact features cannot identify exact preclamp individual path powers.",
                "Error-dependent masks are used only for scoring, never fed into geometric calculations or model inputs."]}
    return clean(result)

def markdown(result):
    lines=["# C149 误差归因审计","", "使用已有 selected12 预测；未训练、未重跑网络推理。以下为重复使用的开发 monitor，不是官方 VAL 或盲测。","",
           "MSE贡献 = 分组SSE / 全部SSE；两个频段各自累计有效像素。LoS按当前注册地形的点剖面分类，并与原C73面元聚合代理分开报告。",""]
    def table(title,names):
        lines.extend(["## "+title,"","| 分组 | 像素占比 415 / 5.775 GHz | RMSE (dB) 415 / 5.775 GHz | 总MSE贡献 415 / 5.775 GHz |","|---|---:|---:|---:|"])
        for name in names:
            v=result["all"] if name=="all" else result["groups"][name]
            fmt=lambda values,scale=1: " / ".join("无" if a is None else f"{a*scale:.4f}" for a in values)
            lines.append(f"| {name} | {fmt(v['pixel_fraction'],100)}% | {fmt(v['RMSE_dB'])} | {fmt(v['share_of_total_SSE_percent'])}% |")
        lines.append("")
    table("总量与 LoS / NLoS",["all","profile_LoS","profile_NLoS","cached_proxy_LoS","cached_proxy_NLoS","profile_vs_cache_LoS_disagreement","cached_mixed_LoS_mass"])
    table("Fresnel 分桶",["Fresnel_"+s for s in BUCKETS])
    table("求解器 floor 编码",["floor_direct_linear_mean","above_floor_direct_linear_mean","floor_total_linear_mean","above_floor_total_linear_mean","floor_direct_or_total_linear_mean","floor_direct_mean_log","floor_total_mean_log"])
    lines.extend(["这里的floor是 Q=P·d_face² 的归一化编码-3（Q≥1e-12），不是绝对路径功率P。Float16使编码-3对应的Q仍可能略高于floor；原始路径是否被clamp无法由压缩缓存精确恢复。",""])
    table("尾区与端点异常",["selected12_error_abs_ge_5dB","selected12_error_abs_lt_5dB","C82_error_abs_ge_5dB","TX_below_reconstructed_terrain","TX_on_reconstructed_terrain"])
    lines.extend(["## 跨频相关","",f"selected12 有符号残差 Pearson：{result['cross_band_residual_correlation']['selected12']['Pearson_signed_error']:.6f}；仅共同有效像素。","",
                  "## 误差最大的10个地形","","按双频各自MSE的均值排序，未按频段挑选不同模型。","",
                  "| 地形 | RMSE 415 / 5.775 GHz (dB) | 总MSE贡献 415 / 5.775 GHz |","|---|---:|---:|"])
    for v in result["worst_10_terrains_joint_MSE"]:
        pair=lambda a:" / ".join(f"{z:.4f}" for z in a)
        lines.append(f"| {v['terrain']} | {pair(v['RMSE_dB'])} | {pair(v['share_of_total_SSE_percent'])}% |")
    lines.extend(["","完整来源、分桶定义、逐频Top10和适用范围见 summary.json。",""])
    return "\n".join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true",help="CPU-only source/identity and analytic oracle checks")
    parser.add_argument("--resume",action="store_true",help="Reuse a partially completed geometry cache")
    args=parser.parse_args()
    inputs=load_inputs()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"summary.json").exists():
        result=read(OUT/"summary.json")
        if result.get("status")=="complete":
            print(json.dumps({"status":"already_complete_no_rerun","summary":"audit/c149_error_audit/summary.json"}));return
    source_files=["src/c149_error_audit.py","src/c149_profiles.cu","src/c70_geometry.cu","src/c70_geometry.py","src/c73_compact.py","src/c72_native_features.py","src/c69_registered.py"]
    protocol={"campaign":"C149","stage":"error_attribution_no_training","panels_source":"audit/c145_capacity/protocol.json",
              "panel_ids_sha256":hashlib.sha256(np.asarray(inputs["ids"],dtype="<i8").tobytes()).hexdigest(),
              "monitor_maps":628,"monitor_terrains":128,"model_sha256":inputs["identity"]["model_sha256"],
              "prediction_source":source_record(RUN/"selected_monitor_prediction.npy"),
              "ordered_panel_ids":inputs["ids"],"source_files":[source_record(ROOT/p) for p in source_files],
              "frequencies_hz":FREQ.tolist(),"wavelengths_m":WAVELENGTH.tolist(),"spacing_m":256/255,
              "classification_thresholds_nu":[-.7,0.,1.5],"tail_dB":5,
              "runtime":"GPU geometry-only analytic triangle intersections; CPU Float64 scoring of saved predictions",
              "forbidden_actions":["network inference","training","modify C70/C73/frozen caches","official VAL"],
              "LoS_conventions":"Both exact registered point highest<0 and existing compact proxy highest-code<0, separately",
              "floor_policy":"Float16 cached floor encoding; Q=P*d_face^2, not absolute/rawP saturation",
              "output_geometry_layout":["highest_m","nu_lambda_1m","t_nu","source_excess_m","flags"],
              "profile_precision":"Float64 analytic triangle kernel; Float32 stored audit profiles and classifications",
              "tracking_step":"geometry maps1..628; scoring map-operations629..1256; summary1257, not training updates",
              "official_VAL_used":False}
    if (OUT/"protocol.json").exists():
        old=read(OUT/"protocol.json")
        if old!=protocol:raise ValueError("Existing C149 protocol differs; do not silently overwrite")
    else:write(OUT/"protocol.json",protocol)
    # Independent CPU analytic result for flat ground has interior extremum t=.75.
    flat=np.zeros((256,256),np.float32);test=profile_reference(flat,80,40,123,200)
    assert abs(test[0]+1)<1e-12 and abs(test[2]-.75)<1e-12 and test[4]==0
    preflight={"status":"CPU_preflight_passed","maps":628,"terrains":128,
               "selected12_RMSE_dB":inputs["expected"]["RMSE_dB"],"flat_nu_argmax":test[2],
               "protocol_sha256":digest(OUT/"protocol.json"),"input_export_identity":inputs["identity"]}
    write(OUT/"preflight.json",preflight)
    if args.preflight:
        print(json.dumps(clean(preflight),ensure_ascii=False));return
    engine=Profiles()
    check=profile_checks(engine,inputs)
    prior_checks=read(OUT/"geometry_reference_checks.json") if (OUT/"geometry_reference_checks.json").exists() else {}
    write(OUT/"geometry_reference_checks.json",{"status":"passed","checks":check,"GPU":engine.gpu,"validation_history":prior_checks.get("validation_history",[])})
    shape=(len(inputs["ids"]),5,256,256)
    cache_path=OUT/"profiles.npy";checkpoint=OUT/"geometry_progress.json"
    start=0
    if cache_path.exists():
        if not args.resume:raise ValueError("Partial geometry exists; inspect and explicitly --resume")
        state=read(checkpoint);assert state["protocol_sha256"]==digest(OUT/"protocol.json")
        cache=np.load(cache_path,mmap_mode="r+");assert cache.shape==shape and cache.dtype==np.float32
        start=int(state["maps"])
    else:cache=np.lib.format.open_memmap(cache_path,mode="w+",dtype=np.float32,shape=shape)
    from src import cc_tracking
    tracking=cc_tracking.init(name="C149 error attribution audit",campaign="C149",
        config={"protocol":protocol,"run_kind":"analysis.error","_cc_dashboard":{"kind":"analysis","scope":"C145 selected12 monitor628maps128terrains; no training"},"GPU":engine.gpu},
        description="Saved prediction error attribution; exact registered geometric profiles; no model execution.",
        tags=["analysis","no-training","development-monitor"])
    write(OUT/"tracking.json",{"run_id":tracking.id,"dir":str(tracking.dir)})
    clock=time.monotonic()
    try:
        for j in range(start,len(inputs["ids"])):
            i=inputs["ids"][j];row=inputs["rows"][i]
            result=engine(inputs["h"][int(row["hm_row"])],int(row["tx_row"]),int(row["tx_col"]))
            cache[j]=result.astype(np.float32)
            if (j+1)%16==0 or j+1==len(inputs["ids"]):
                cache.flush();elapsed=time.monotonic()-clock
                value={"stage":"audit_geometry","maps":j+1,"total":len(inputs["ids"]),"elapsed_seconds":elapsed,
                       "eta_seconds":elapsed/max(1,j+1-start)*(len(inputs["ids"])-j-1),
                       "protocol_sha256":digest(OUT/"protocol.json")}
                write(checkpoint,value);write(OUT/"progress.json",value);print(json.dumps(value),flush=True)
                cc_tracking.log({"geometry/maps":j+1,"geometry/total_maps":len(inputs["ids"]),"geometry/elapsed_seconds":elapsed},step=j+1)
        result=audit(inputs,cache,cc_tracking)
        result.update(protocol_sha256=digest(OUT/"protocol.json"),profile_cache_sha256=digest(cache_path),
                      GPU=engine.gpu,geometry_reference_checks="audit/c149_error_audit/geometry_reference_checks.json",
                      elapsed_seconds=time.monotonic()-clock,source_identity=protocol["source_files"],
                      tracking=read(OUT/"tracking.json"))
        write(OUT/"summary.json",result);(OUT/"summary.md").write_text(markdown(result))
        metrics={}
        for name,values in {"all":result["all"],**result["groups"]}.items():
            for k,band in enumerate(BANDS):
                for key in ("RMSE_dB","pixel_fraction","share_of_total_SSE_percent"):
                    if values[key][k] is not None:metrics[f"audit/{name}/{key}_{band}"]=values[key][k]
        cc_tracking.log(metrics,step=2*len(inputs["ids"])+1)
        write(OUT/"progress.json",{"stage":"complete","maps":628,"total":628,"elapsed_seconds":time.monotonic()-clock})
        print(json.dumps({"status":"complete","all":result["all"],"cross_band":result["cross_band_residual_correlation"],
                          "summary":"audit/c149_error_audit/summary.json"},ensure_ascii=False),flush=True)
    finally:cc_tracking.finish()

if __name__=="__main__":main()
