"""C150 deterministic, input-only registered diffraction feature generation.

No existing solver feature is re-exported. Raw labels, validity masks, C82,
models, and training are never loaded by this module.
"""
from pathlib import Path
import argparse,csv,ctypes,hashlib,json,math,os,subprocess,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
AUDIT=ROOT/"audit/c150_diffraction"
PART=ROOT/"features/c150_diffraction/part0of1"
FREQUENCIES_HZ=np.array([415e6,5.775e9],np.float64)
WAVELENGTHS_M=299792458.0/FREQUENCIES_HZ
CHANNEL_NAMES=["nu415","t415","I415","Ld415","Lb415",
               "nu5775","t5775","I5775","Ld5775","Lb5775","FSPL415","FSPL5775"]
APPROVED={
"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
"GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
"GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"}
REFERENCE="https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.526-16-202511-I%21%21PDF-E.pdf"

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(path):return json.loads(Path(path).read_text())
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def write(path,data):
    p=Path(path);t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def terrain_height(h,x,y):
    ix=min(254,max(0,int(math.floor(x))));iy=min(254,max(0,int(math.floor(y))))
    fx=x-ix;fy=y-iy
    a,b,c,d=map(float,(h[iy,ix],h[iy,ix+1],h[iy+1,ix],h[iy+1,ix+1]))
    return a+(d-c)*fx+(c-a)*fy if fy>fx else a+(b-a)*fx+(d-b)*fy

def inputs():
    plan=read(ROOT/"audit/c145_capacity/protocol.json")
    fit=list(plan["panels"]["fit"]);monitor=list(plan["panels"]["monitor"]);ids=fit+monitor
    assert len(fit)==2815 and len(monitor)==628 and len(set(ids))==3443
    data=ROOT.parent/"data"
    with (data/"train_index.csv").open() as f:rows=list(csv.DictReader(f))
    heights=np.load(data/"train_hm.npy",mmap_mode="r",allow_pickle=False)
    return {"fit":fit,"monitor":monitor,"ids":ids,"rows":rows,"heights":heights}

def coordinate_preflight(data):
    gaps=[];nearest=[];axis=np.linspace(1/3,255-1/3,256)
    for i in data["ids"]:
        r=data["rows"][i];tr,tc=int(r["tx_row"]),int(r["tx_col"]);h=data["heights"][int(r["hm_row"])]
        x,y=tc*255/256,(tr+1)*255/256;z=math.floor(float(h[min(tr+1,255),tc]))+3
        gaps.append(z-terrain_height(h,x,y))
        nearest.append((256/255)*math.hypot(np.min(np.abs(axis-x)),np.min(np.abs(axis-y))))
    assert min(gaps)>0 and min(nearest)>0,"Buried/touching/zero-horizontal paths are not silently capped"
    return {"maps":len(data["ids"]),"min_TX_clearance_m":min(gaps),"max_TX_clearance_m":max(gaps),
            "min_horizontal_TX_to_query_m":min(nearest),"buried_or_touching_TX_count":sum(v<=0 for v in gaps),
            "labels_read":False,"masks_read":False,"C82_read":False}

def protocol(data):
    names=["src/c150_diffraction.py","src/c150_diffraction.cu","src/c150_reference.py"]
    sources={n:sha(ROOT/n) for n in names}
    result={
      "campaign":"C150","purpose":"Registered deterministic diffraction features for isolated residual-prior A/B",
      "source_sha256":sources,"reference":REFERENCE,
      "original_panels_protocol_sha256":sha(ROOT/"audit/c145_capacity/protocol.json"),
      "ordered_IDs":"C145 fit2815 then monitor628; no overlap","ordered_ids_sha256":hashlib.sha256(np.asarray(data["ids"],dtype="<i8").tobytes()).hexdigest(),
      "shape":[3443,12,256,256],"dtype":"float32","channel_names":CHANNEL_NAMES,
      "feature_directory":"features/c150_diffraction/part0of1","raw_diffraction_channels":list(range(10)),
      "Deygout_channels":[3,8],"FSPL_channels":[10,11],
      "frequency_hz":FREQUENCIES_HZ,"wavelength_m":WAVELENGTHS_M,
      "geometry_convention":{"terrain":"anti0 piecewise planar h[0..255,0..255]",
         "spacing_m":256/255,"TX":"x=tc*255/256,y=(tr+1)*255/256,z=floor(h[min(tr+1,255),tc])+3",
         "RX":"x,y use linspace(1/3,255-1/3,256); z=triangle terrain(x,y)+1",
         "FSPL":"20log10(4pi*d3D/lambda), true registered 3D distance; no model/floor fitting"},
      "nu_max":"Analytic continuous-triangle extrema (endpoints plus t=a/(b+2a) for e=a+bt); not midpoint marching",
      "positive_integral":"Exact integral of max(nu(t),0)dt over normalized t; split every triangle and zero of e. Primitive (2a+b)asin(sqrt(t))-b*sqrt(t(1-t)), scaled sqrt(2/(lambda*d2D)).",
      "Deygout":"Classical at-most-three-edge approximation: root J(nu) when nu>-.78, at most one left and one right subpath obstruction; secondary requires nu>1e-9. Stop if root J=0. No empirical ITU C/T correction and no deeper recursion.",
      "knife_edge_J":"max(0,6.9+20log10(sqrt((nu-.1)^2+1)+nu-.1)) for nu>-.78, else0; P526eq31",
      "Bullington":"P526-16section4.5.1eq49-57, Ce=0 local plane; LoS and trans-horizon construction plus tapered correction with d_km=d_m/1000; not complete delta-Bullington/spherical-Earth and not strict bounds",
      "numerical_policy":{"internal_arithmetic":"Float64 analytic geometry; primary node stays Float64 for secondary geometry",
         "outputs":"Float32; reported nu(lambda1)/t rounded like C149; internal J uses raw Float64 root",
         "endpoint_obstacles":"Known secondary obstacle endpoints lie exactly on terrain; snap their numerical zero, do not create automatic6dB edges",
         "secondary_positive_nu_tolerance":1e-9,"Bullington_tangent_height_tolerance_m":1e-10,
         "nonfinite_policy":"No arbitrary nu/loss caps; reject invalid terminal clearance or degenerate paths",
         "sampling":"Continuous affine-segment calculation is exact between mesh intersections, including extrema; no omitted sub-metre features"},
      "C149_reuse":"Monitor nu(lambda1)/t reused only as saved feature outputs after agreement check; Float32 C149 argmax never defines Deygout geometry. Fresh analytic primary is required by the new integral/Bullington/secondary quantities.",
      "deduplication":{"not_repeated":["C109 legacy integer-coordinate 32-sample ordered/permuted Fresnel profiles","C130 legacy sparse three Fresnel ridge nodes","C147 reflection-depth comparison","C148 parameter-anchor sweeps"],
         "new":"Registered analytic Ipos + explicit Deygout/Bullington loss and teacher-free prior A/B"},
      "solver_channels":"Existing45c73 unchanged; no solver power re-export/floor change",
      "labels_read":False,"masks_read":False,"C82_read":False,"network_or_training_executed":False,
      "official_VAL_used":False}
    return clean(result)

class DiffractionFeatures:
    def __init__(self):
        uuid=os.environ.get("CUDA_VISIBLE_DEVICES")
        assert uuid in APPROVED and os.environ.get("CC_GPU_UUID")==uuid,"One full approved UUID required"
        import torch
        from src.cc_core import gpu_guard,check_source_unchanged
        self.torch=torch;self.gpu=gpu_guard();check_source_unchanged()
        AUDIT.mkdir(parents=True,exist_ok=True)
        source=ROOT/"src/c150_diffraction.cu";library=AUDIT/"libc150_diffraction.so";metadata=AUDIT/"kernel_build.json"
        cmd=["/usr/local/cuda-11.8/bin/nvcc","-O3","--shared","--cudart=shared","-Xcompiler","-fPIC",
             "-Xlinker","-rpath","-Xlinker","/usr/local/cuda-11.8/lib64",
             "-gencode","arch=compute_80,code=compute_80",str(source),"-o",str(library)]
        if not library.exists() or not metadata.exists() or read(metadata).get("source_sha256")!=sha(source):
            subprocess.run(cmd,check=True,cwd=ROOT)
            write(metadata,{"source_sha256":sha(source),"library_sha256":sha(library),"command":cmd})
        else:assert read(metadata)["library_sha256"]==sha(library)
        self.lib=ctypes.CDLL(str(library));self.fn=self.lib.c150_diffraction
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self.fn.restype=ctypes.c_int
        self.max_reused_nu_difference=0.;self.max_reused_t_difference=0.;self.reused_maps=0
    def __call__(self,h,tr,tc,reuse=None):
        t=self.torch
        values=t.tensor(np.ascontiguousarray(h),dtype=t.float32,device="cuda").contiguous()
        reused=None if reuse is None else t.tensor(np.ascontiguousarray(reuse),dtype=t.float32,device="cuda").contiguous()
        output=t.empty((14,256,256),dtype=t.float32,device="cuda")
        error=self.fn(values.data_ptr(),int(tr),int(tc),0 if reused is None else reused.data_ptr(),
                      output.data_ptr(),t.cuda.current_stream().cuda_stream)
        if error:raise RuntimeError(f"C150 CUDA launch failed:{error}")
        result=output.cpu().numpy()
        if not np.isfinite(result).all():raise ArithmeticError("Nonfinite C150 feature; no clipping permitted")
        if reuse is not None:
            old=np.asarray(reuse)
            if not np.allclose(result[12],old[0],atol=1e-6,rtol=1e-6):
                raise AssertionError("C149 and C150 primary nu disagree")
            if not np.allclose(result[13],old[1],atol=5e-6,rtol=0):
                raise AssertionError("C149 and C150 primary argmax disagree")
            self.max_reused_nu_difference=max(self.max_reused_nu_difference,float(np.max(np.abs(result[12]-old[0]))))
            self.max_reused_t_difference=max(self.max_reused_t_difference,float(np.max(np.abs(result[13]-old[1]))))
            self.reused_maps+=1
        v=result[:12]
        assert (v[[1,6]]>=0).all() and (v[[1,6]]<=1).all()
        assert (v[[2,3,4,7,8,9]]>=0).all()
        assert np.allclose(v[11]-v[10],20*math.log10(FREQUENCIES_HZ[1]/FREQUENCIES_HZ[0]),atol=3e-5,rtol=0)
        return v

def validate(engine,data,proto):
    from src.c150_reference import reference_array,self_check
    reference_checks=self_check()
    x=np.arange(256,dtype=np.float32)
    flat=np.zeros((256,256),np.float32)
    ridge=np.broadcast_to(np.maximum(0,12-np.abs(x-128)),(256,256)).astype(np.float32)
    two=np.broadcast_to(np.maximum(np.maximum(0,8-np.abs(x-115)),np.maximum(0,11-np.abs(x-160))),(256,256)).astype(np.float32)
    three=np.broadcast_to(np.maximum(np.maximum(np.maximum(0,7-np.abs(x-105)),np.maximum(0,12-np.abs(x-140))),np.maximum(0,8-np.abs(x-175))),(256,256)).astype(np.float32)
    affine=(np.arange(256)[None,:]/32+np.arange(256)[:,None]/64).astype(np.float32)
    near_tie=np.broadcast_to(np.maximum(np.maximum(0,8-np.abs(x-115)),np.maximum(0,7.200001-np.abs(x-160))),(256,256)).astype(np.float32)
    cases=[("flat",flat,80,40),("isolated_nonbinary_ridge",ridge,100,80),
           ("two_ridges",two,100,80),("three_ridges",three,100,80),
           ("affine",affine,70,60),("near_tie",near_tie,100,80)]
    checks=[];largest=0.
    for name,h,tr,tc in cases:
        values=engine(h,tr,tc)
        for r,c in [(17,21),(123,200),(235,58),(100,220)]:
            expected=reference_array(h,tr,tc,r,c)
            actual=values[:,r,c].astype(float)
            difference=float(np.max(np.abs(actual-expected)))
            if not np.allclose(actual,expected,atol=3e-4,rtol=3e-6):
                raise AssertionError((name,r,c,actual,expected,actual-expected))
            checks.append({"fixture":name,"row":r,"col":c,"max_abs_difference":difference})
            largest=max(largest,difference)
    for i in data["fit"][:2]+data["monitor"][:2]:
        r=data["rows"][i];h=data["heights"][int(r["hm_row"])]
        tr,tc=int(r["tx_row"]),int(r["tx_col"]);values=engine(h,tr,tc)
        for rr,cc in [(27,31),(193,201),(100,170)]:
            expected=reference_array(h,tr,tc,rr,cc)
            actual=values[:,rr,cc].astype(float)
            difference=float(np.max(np.abs(actual-expected)))
            assert np.allclose(actual,expected,atol=3e-4,rtol=3e-6),(i,rr,cc,actual,expected)
            checks.append({"fixture":f"raw_map_{i}","row":rr,"col":cc,"max_abs_difference":difference})
            largest=max(largest,difference)
    # Validate the output-only C149 reuse path without rounding recursive nodes.
    c149=read(ROOT/"audit/c149_error_audit/protocol.json")
    cached=np.load(ROOT/"audit/c149_error_audit/profiles.npy",mmap_mode="r",allow_pickle=False)
    assert c149["ordered_panel_ids"]==data["monitor"]
    for j in range(2):
        i=data["monitor"][j];r=data["rows"][i]
        engine(data["heights"][int(r["hm_row"])],int(r["tx_row"]),int(r["tx_col"]),np.array(cached[j,[1,2]]))
    return {"status":"passed","protocol_sha256":sha(AUDIT/"protocol.json"),"source_sha256":proto["source_sha256"],
            "GPU":engine.gpu,"CPU_reference_self_checks":reference_checks,
            "GPU_CPU_checks":checks,"max_absolute_feature_difference":largest,
            "registered_maps_checked":4,"synthetic_maps_checked":6,
            "C149_output_only_reuse_maps_checked":2,
            "internal_geometry_dtype":"float64","cache_dtype":"float32","channel_names":CHANNEL_NAMES,
            "network_or_training_executed":False}

def export(data,proto,resume):
    if (PART/"complete.json").exists():
        done=read(PART/"complete.json")
        assert done["protocol_sha256"]==sha(AUDIT/"protocol.json")
        assert done["features_sha256"]==sha(PART/"features.npy")
        print(json.dumps({"status":"already_complete_no_reexport","complete":str(PART/"complete.json")}));return
    v=read(AUDIT/"validation.json")
    assert v["status"]=="passed" and v["protocol_sha256"]==sha(AUDIT/"protocol.json")
    assert v["source_sha256"]==proto["source_sha256"],"Validation source identity changed"
    PART.mkdir(parents=True,exist_ok=True)
    index_path=PART/"indices.json";cache_path=PART/"features.npy";progress_path=PART/"progress.json"
    if index_path.exists():assert read(index_path)==data["ids"]
    else:write(index_path,data["ids"])
    shape=(3443,12,256,256);start=0
    if cache_path.exists():
        assert resume,"Partial cache exists; inspect and explicitly --resume"
        progress=read(progress_path);assert progress["protocol_sha256"]==sha(AUDIT/"protocol.json")
        start=int(progress["maps"]);cache=np.load(cache_path,mmap_mode="r+")
        assert cache.shape==shape and cache.dtype==np.float32
    else:cache=np.lib.format.open_memmap(cache_path,mode="w+",dtype=np.float32,shape=shape)
    engine=DiffractionFeatures()
    old_root=ROOT/"audit/c149_error_audit"
    old=read(old_root/"summary.json")
    assert old["status"]=="complete" and sha(old_root/"profiles.npy")==old["profile_cache_sha256"]
    old_protocol=read(old_root/"protocol.json");assert old_protocol["ordered_panel_ids"]==data["monitor"]
    old_profiles=np.load(old_root/"profiles.npy",mmap_mode="r",allow_pickle=False)
    old_lookup={i:j for j,i in enumerate(data["monitor"])}
    from src import cc_tracking
    run=cc_tracking.init(name="C150 deterministic diffraction features",campaign="C150",
      config={"run_kind":"analysis.features","protocol":proto,"_cc_dashboard":{"kind":"analysis","scope":"3443fixed FIT+monitor inputs; no labels/masks/models"},"GPU":engine.gpu},
      description="Continuous-triangle Fresnel integral, classical3edge and P526 Bullington; no solver rerun.",
      tags=["analysis","input-only","deterministic-features"])
    write(AUDIT/"tracking.json",{"run_id":run.id,"dir":str(run.dir)})
    before=time.monotonic();minimum=np.full(12,np.inf);maximum=np.full(12,-np.inf)
    try:
        if start:
            for j in range(start):
                minimum=np.minimum(minimum,np.asarray(cache[j]).min((1,2)))
                maximum=np.maximum(maximum,np.asarray(cache[j]).max((1,2)))
        for j in range(start,len(data["ids"])):
            i=data["ids"][j];r=data["rows"][i]
            reuse=None if i not in old_lookup else np.array(old_profiles[old_lookup[i],[1,2]],copy=True)
            value=engine(data["heights"][int(r["hm_row"])],int(r["tx_row"]),int(r["tx_col"]),reuse)
            cache[j]=value
            minimum=np.minimum(minimum,value.min((1,2)));maximum=np.maximum(maximum,value.max((1,2)))
            if (j+1)%16==0 or j+1==len(data["ids"]):
                cache.flush();elapsed=time.monotonic()-before
                progress={"stage":"feature_export","maps":j+1,"total":len(data["ids"]),"elapsed_seconds":elapsed,
                          "eta_seconds":elapsed/max(1,j+1-start)*(len(data["ids"])-j-1),
                          "protocol_sha256":sha(AUDIT/"protocol.json")}
                write(progress_path,progress);print(json.dumps(progress),flush=True)
                cc_tracking.log({"progress/maps":j+1,"progress/total_maps":len(data["ids"]),
                  "time/elapsed_seconds":elapsed,"time/eta_seconds":progress["eta_seconds"]},step=j+1)
        done={"status":"complete","maps":3443,"shape":list(shape),"dtype":"float32","channel_names":CHANNEL_NAMES,
              "features_sha256":sha(cache_path),"indices_sha256":sha(index_path),
              "ordered_ids_sha256":proto["ordered_ids_sha256"],"protocol_sha256":sha(AUDIT/"protocol.json"),
              "source_sha256":proto["source_sha256"],"geometry_convention":proto["geometry_convention"],
              "channel_minimum":minimum,"channel_maximum":maximum,"GPU":engine.gpu,
              "C149_output_reuse":{"maps":engine.reused_maps,"max_nu_lambda1_difference":engine.max_reused_nu_difference,
                                  "max_t_difference":engine.max_reused_t_difference,
                                  "source_cache_sha256":old["profile_cache_sha256"],
                                  "internal_Deygout_nodes_reused":False},
              "elapsed_seconds":time.monotonic()-before,"labels_read":False,"masks_read":False,"C82_read":False,
              "network_or_training_executed":False,"fixed_solver_features_unchanged":True,
              "tracking":{"run_id":run.id,"dir":str(run.dir)}}
        write(PART/"complete.json",done);write(AUDIT/"complete.json",done)
        cc_tracking.log({"completion/maps":3443,"completion/elapsed_seconds":done["elapsed_seconds"]},step=3444)
        print(json.dumps(clean({"status":"complete","maps":3443,"features_sha256":done["features_sha256"],"elapsed_seconds":done["elapsed_seconds"]})),flush=True)
    finally:cc_tracking.finish()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight",action="store_true")
    group.add_argument("--validate",action="store_true",help="Small authorized GPU/CPU fixtures only")
    group.add_argument("--export",action="store_true",help="Full input-only feature export; requires separate launch authorization")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args();data=inputs();AUDIT.mkdir(parents=True,exist_ok=True)
    proto=protocol(data)
    if (AUDIT/"protocol.json").exists():
        assert read(AUDIT/"protocol.json")==proto,"Frozen source/protocol changed"
    else:write(AUDIT/"protocol.json",proto)
    if args.preflight:
        result={"status":"CPU_preflight_passed","protocol_sha256":sha(AUDIT/"protocol.json"),**coordinate_preflight(data)}
        write(AUDIT/"preflight.json",result);print(json.dumps(result));return
    if args.validate:
        coordinate_preflight(data);engine=DiffractionFeatures();result=validate(engine,data,proto)
        write(AUDIT/"validation.json",result)
        print(json.dumps({"status":result["status"],"checks":len(result["GPU_CPU_checks"]),
                          "max_absolute_feature_difference":result["max_absolute_feature_difference"],
                          "protocol_sha256":result["protocol_sha256"]}));return
    assert read(AUDIT/"preflight.json")["protocol_sha256"]==sha(AUDIT/"protocol.json")
    export(data,proto,args.resume)

if __name__=="__main__":main()
