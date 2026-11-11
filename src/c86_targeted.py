"""Use the original solver for exact global-lattice subqueries, including repeated RX surface crossings."""
import argparse,ctypes,csv,importlib,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from c79_sionna import meshfile,mi,rt,dr
from c66_lattice import Mesh
D=R.parent/'data';A=R/'audit/c79_receiver';C=R/'audit/c78_clear'
LIB=ctypes.CDLL(str(R/'src/libc86_directions.so'));UP=ctypes.POINTER(ctypes.c_uint32)
LIB.directions86.argtypes=[np.ctypeslib.ndpointer(dtype='float64',flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_longlong,ctypes.POINTER(UP),ctypes.POINTER(ctypes.c_longlong)]
LIB.directions86.restype=ctypes.c_int;LIB.directions86_free.argtypes=[UP]
def indices(verts,pad,N,alpha):
    ctr=verts.mean(1);rad=np.linalg.norm(verts-ctr[:,None],axis=-1).max(1)+pad+.003
    length=np.linalg.norm(ctr,axis=1);delta=np.arcsin(np.minimum(.99,rad/length));theta=np.arccos(np.clip(ctr[:,2]/length,-1,1))
    uc=(np.arctan2(ctr[:,1],ctr[:,0])-alpha)/(2*np.pi)
    du=np.arcsin(np.minimum(.999,np.sin(delta)/np.sin(theta)))/(2*np.pi)
    vl=.5*(1-np.cos(np.maximum(theta-delta,0)));vh=.5*(1-np.cos(np.minimum(theta+delta,np.pi)))
    boxes=np.ascontiguousarray(np.stack([uc-du-2e-6,uc+du+2e-6,vl-2e-6,vh+2e-6],1))
    ptr=UP();n=ctypes.c_longlong();rc=LIB.directions86(boxes,len(boxes),N,ctypes.byref(ptr),ctypes.byref(n))
    if rc:raise RuntimeError(f'direction enumeration {rc}')
    try:return np.ctypeslib.as_array(ptr,shape=(n.value,)).copy()
    finally:LIB.directions86_free(ptr)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--of',type=int,default=3)
    ap.add_argument('--limit',type=int,default=0);args=ap.parse_args();dr.set_thread_count(6)
    assert rt.__version__=='1.1.0' and mi.variant()=='llvm_ad_mono_polarized'
    module=importlib.import_module('sionna.rt.radio_map_solvers.radio_map_solver')
    rows=list(csv.DictReader((D/'train_index.csv').open()));height=np.load(D/'train_hm.npy',mmap_mode='r');panel=np.load(C/'train_panel.npz')
    config=json.load(open(A/'c86_protocol.json'))['maps'];assigned=list(enumerate(config))[args.shard::args.of]
    if args.limit:assigned=assigned[:args.limit]
    out=A/f'targeted{args.shard}';out.mkdir(exist_ok=True)
    mesh=Mesh(1,0);idx=np.load(R/'src/c70_centroid_lookup0.npz');t0=time.monotonic()
    for kn,(seq,rec) in enumerate(assigned):
        path=out/f"{rec['terrain']}.npz"
        if path.exists():continue
        rr=rows[rec['index']];h=height[int(rr['hm_row'])];tr,tc=int(rr['tx_row']),int(rr['tx_col'])
        tx=np.array([tc-128.,127.-tr,float(np.floor(h[min(tr+1,255),tc])+3)])
        vert=np.column_stack([mesh.xy,h.ravel()]).astype(np.float32);up=vert.copy();up[:,2]+=1
        faces=mesh.faces[:,[0,2,1]]
        meshfile(out/'ground.ply',vert,faces);meshfile(out/'rx.ply',up,faces)
        ground=mi.load_dict({'type':'ply','filename':str(out/'ground.ply'),'face_normals':False,'bsdf':{'type':'holder-material'}})
        mat=rt.RadioMaterial('regolith',relative_permittivity=3.411,conductivity=.0003562,thickness=10.,scattering_coefficient=.632,xpd_coefficient=.1,scattering_pattern='lambertian')
        scene=rt.load_scene();scene.edit(add=rt.SceneObject(mi_mesh=ground,name='terrain',radio_material=mat))
        scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V');scene.rx_array=scene.tx_array
        scene.add(rt.Transmitter('tx',position=tx.tolist()));scene.frequency=415e6
        rx=mi.load_dict({'type':'ply','filename':str(out/'rx.ply'),'face_normals':True})
        jp=np.array(rec['locations']);pixel=panel['pixels'][rec['pos'],jp]
        ids=idx['ids'][pixel];w=idx['weights'][pixel];unique=np.unique(ids[w>1e-7])
        v=up[faces[unique]].astype('float64')-tx
        norm=(4*np.pi*415e6/299792458)**2*np.square(up[faces].mean(1)-tx).sum(1).clip(.0625)
        solver=rt.RadioMapSolver();results=[];num=[]
        modes=[(4_000_000,0.,.65),(4_000_000,.013,.65),(4_000_000_000,0.,.65),(4_000_000_000,.013,.65)]
        if seq%4==0:modes.append((4_000_000_000,0.,1.35))
        for N,alpha,pad in modes:
            selected=indices(v,pad,N,alpha);pwr=np.zeros(len(faces),'float64')
            for st in range(0,len(selected),1_000_000):
                rays=selected[st:st+1_000_000];M=len(rays)
                def lattice(request_size):
                    assert request_size==M
                    ns=mi.Float64(mi.UInt(rays));golden=(1.+dr.sqrt(mi.Float64(5.)))/2.
                    x=ns/golden;x=x-dr.floor(x)
                    x=x+alpha/(2*np.pi);x=x-dr.floor(x)
                    return mi.Point2f(x,ns/(N-1))
                module.fibonacci_lattice=lattice
                radio=solver(scene,measurement_surface=rx,samples_per_tx=M,max_depth=0,los=True,specular_reflection=False,diffuse_reflection=False,refraction=False,seed=13)
                pwr+=radio.path_gain.numpy()[0].astype('float64')*(M/N)
            rel=pwr*norm;gain=(rel[ids]*w).sum(1);hit=((pwr[ids]>0)*w).sum(1)
            results.append(np.stack([gain,hit],1));num.append(len(selected))
        assert all(np.isfinite(z).all() for z in results)
        tmp=path.with_suffix('.part.npz');np.savez(tmp,values=np.stack(results),directions=num,modes=modes,pixel=pixel,loc=jp,pos=rec['pos'],index=rec['index'])
        tmp.replace(path);print('map',kn+1,len(assigned),'target cells',len(pixel),'selected rays',num,'seconds',round(time.monotonic()-t0,1),flush=True)
    if not args.limit:(out/'complete.json').write_text(json.dumps(dict(maps=len(assigned),seconds=time.monotonic()-t0))+'\n')
if __name__=='__main__':main()
