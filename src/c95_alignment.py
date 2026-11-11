"""Original solver, one-factor scene controls. Inputs H/TX only; never reads PL, masks, or predictions."""
import argparse,importlib,json,sys,time,os
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from c86_targeted import indices,meshfile,mi,rt,dr
A=R/'audit/c95_alignment';P=R/'audit/c93_joint_miss';N=4_000_000_000;C=(4*np.pi*415e6/299792458)**2
SOLVER_CACHE={}
def variant_solver(cfg):
    step=cfg.get('step')
    if not step:return rt.RadioMapSolver(),importlib.import_module('sionna.rt.radio_map_solvers.radio_map_solver')
    if step not in SOLVER_CACHE:
        from importlib.util import spec_from_file_location,module_from_spec
        name='sionna.rt.radio_map_solvers.c95_'+step
        spec=spec_from_file_location(name,A/'solver_variants'/('radio_map_solver_'+step+'.py'))
        mod=module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);SOLVER_CACHE[step]=mod
    mod=SOLVER_CACHE[step];solver=mod.RadioMapSolver()
    solver.c95_origin=mi.Point3f(*cfg.get('offset_origin',[0,0,0]))
    solver.c95_position_scaled=cfg.get('position_scaled',True)
    solver.c95_epsilon=cfg.get('epsilon',8.940696716308594e-05)
    return solver,mod


def faces(mode,xy,order=0):
    a=np.arange(65536).reshape(256,256)[:-1,:-1].ravel()
    if mode=='delaunay':
        seq=np.arange(65536).reshape(256,256)
        if order&4:seq=seq.T
        if order&1:seq=seq[::-1]
        if order&2:seq=seq[:,::-1]
        seq=seq.ravel();f=seq[Delaunay(xy[seq]).simplices]
        cross=np.cross(xy[f[:,1]]-xy[f[:,0]],xy[f[:,2]]-xy[f[:,0]])
        f[cross<0]=f[cross<0][:,[0,2,1]]
        return f.astype('int32')
    if mode=='anti':can=np.concatenate([np.stack([a,a+1,a+256],1),np.stack([a+257,a+256,a+1],1)])
    else:can=np.concatenate([np.stack([a,a+1,a+257],1),np.stack([a,a+257,a+256],1)])
    return can[:,[0,2,1]]

def support(xy,ff,config):
    if not any(k in config for k in ('rx','grid','rxoffset','pixel','centermode','frame')):
        v=np.load(R/'src/c70_centroid_lookup0.npz');return v['ids'],v['weights'],np.ones(65536,bool)
    ctr=xy[ff].astype('float64').mean(1)
    if 'pixel' in config:ctr=xy[ff[:,[0,2,1]]].mean(1)
    if config.get('centermode')=='sionna':
        flat=ff.ravel();cs=[]
        for a in (0,1):
            tensor=dr.reshape(mi.TensorXf,mi.Float(xy[flat,a]),[len(ff),3]);cs.append(dr.mean(tensor,axis=1).numpy())
        ctr=np.stack(cs,axis=1)
    tri=Delaunay(ctr)
    gx,gy=np.meshgrid(np.linspace(ctr[:,0].min(),ctr[:,0].max(),256),np.linspace(ctr[:,1].max(),ctr[:,1].min(),256))
    qp=np.stack([gx.ravel(),gy.ravel()],1).astype('float64')
    if 'pixel' in config:
        pp=config['pixel']
        if isinstance(pp,list):qp+=np.array(pp)[None,:]
        elif pp=='vertices':
            qx,qy=np.meshgrid(np.linspace(-128,128,256),np.linspace(128,-128,256));qp=np.stack([qx.ravel(),qy.ravel()],1)
        elif pp=='integer':
            qx,qy=np.meshgrid(np.arange(256)-128,127-np.arange(256));qp=np.stack([qx.ravel(),qy.ravel()],1)
        elif pp=='half':
            qx,qy=np.meshgrid(np.arange(256)-127.5,127.5-np.arange(256));qp=np.stack([qx.ravel(),qy.ravel()],1)
    ss=tri.find_simplex(qp);valid=ss>=0;ss=ss.clip(0)
    tt=tri.transform[ss];w=np.einsum('nij,nj->ni',tt[:,:2],qp-tt[:,2]);w=np.column_stack([w,1-w.sum(1)])
    w[~valid]=0
    return tri.simplices[ss],w,valid

def geometry(h,rr,cfg,templates):
    r,c=int(rr['tx_row']),int(rr['tx_col'])
    axis=np.linspace(-128,128,256,dtype=np.float32);xx,yy=np.meshgrid(axis,-axis)
    if cfg.get('grid')=='integer':xx,yy=np.meshgrid(np.arange(256,dtype=np.float32)-128,127-np.arange(256,dtype=np.float32))
    xy=np.stack([xx.ravel(),yy.ravel()],1)
    if cfg.get('frame') in ('positive256','positive255'):
        aa=np.linspace(0,256,256,dtype=np.float32) if cfg['frame']=='positive256' else np.arange(256,dtype=np.float32)
        xx,yy=np.meshgrid(aa,aa[::-1]);xy=np.stack([xx.ravel(),yy.ravel()],1)
    fg=faces(cfg.get('ground','main'),xy,cfg.get('vertex_order',0));fr=faces(cfg.get('rx','main'),xy,cfg.get('vertex_order',0))
    vert=np.column_stack([xy,h.ravel()]).astype('float32')
    tx=np.array([c-128.,127.-r,float(np.floor(h[min(r+1,255),c])+3)])
    if cfg.get('frame') in ('positive256','positive255'):
        tx[:2]=[c,255.-r]
        if cfg['frame']=='positive255':tx[2]=np.floor(h[r,c])+3
    if cfg.get('txxy')=='vertex':tx[:2]=[xx[r,c],yy[r,c]]
    if cfg.get('txxy')=='half':tx[:2]=[c-127.5,127.5-r]
    if cfg.get('txz')=='row':tx[2]=np.floor(h[r,c])+3
    if cfg.get('txz')=='continuous':tx[2]=h[min(r+1,255),c]+3
    if cfg.get('txz')=='surface':
        u=(tx[0]+128)*255/256;v=(128-tx[1])*255/256;cx=min(254,int(u));cy=min(254,int(v));u-=cx;v-=cy
        if u>=v:z=(1-u)*h[cy,cx]+(u-v)*h[cy,cx+1]+v*h[cy+1,cx+1]
        else:z=(1-v)*h[cy,cx]+u*h[cy+1,cx+1]+(v-u)*h[cy+1,cx]
        tx[2]=np.floor(z)+3
    if 'txdelta' in cfg:tx+=np.asarray(cfg['txdelta'])
    rv=vert.copy();rv[:,2]+=cfg.get('rxheight',1.)
    if cfg.get('rxoffset')=='normal':
        dy,dx=np.gradient(h.astype('float64'),float(xx[0,1]-xx[0,0]))
        normal=np.stack([-dx,dy,np.ones_like(dx)],-1);normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
        rv=vert+normal.reshape(-1,3).astype('float32')
    key=json.dumps(cfg,sort_keys=True)
    if cfg.get('rxoffset')=='normal':key+=str(rr['hm_row'])
    if key not in templates:templates[key]=support(rv[:,:2],fr,cfg)
    look=templates[key]
    if 'world_shift' in cfg:
        shift=np.asarray(cfg['world_shift'],dtype='float32');vert=vert+shift;rv=rv+shift;tx=tx+shift
    return vert,rv,fg,fr,tx,look

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['screen','extension','confirm','cases','padding'],default='screen');ap.add_argument('--shard',type=int,required=True);ap.add_argument('--of',type=int,default=6)
    ap.add_argument('--configs',nargs='*');ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args();assert mi.variant()=='llvm_ad_mono_polarized' and rt.__version__=='1.1.0'
    dr.set_thread_count(4)
    plan=json.loads((A/'plan.json').read_text());cfgs=args.configs or list(plan['configs'])
    if args.stage=='confirm':
        frozen=json.loads((A/'freeze.json').read_text());assert set(cfgs)<=set(frozen['configs'])
    import csv
    rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));hm=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
    f=np.load(P/'patch_features.npz');grid=f['pixels']
    out=A/args.stage;out.mkdir(exist_ok=True);work=A/'isolated_work'/args.stage/f'worker{args.shard}_pid{os.getpid()}';work.mkdir(parents=True,exist_ok=False)
    tasks=[(e,c) for e in plan[args.stage] for c in cfgs][args.shard::args.of]
    if args.limit:tasks=tasks[:args.limit]
    module=importlib.import_module('sionna.rt.radio_map_solvers.radio_map_solver');templates={};t0=time.monotonic()
    for j,(rec,cfgname) in enumerate(tasks):
        cfg=plan['configs'][cfgname];i=rec['index'];dest=out/f'{i}_{cfgname}.npz'
        if dest.exists():continue
        h=hm[int(rows[i]['hm_row'])];vert,rv,fg,fr,tx,look=geometry(h,rows[i],cfg,templates)
        meshfile(work/'ground.ply',vert,fg);meshfile(work/'rx.ply',rv,fr)
        ground=mi.load_dict({'type':'ply','filename':str(work/'ground.ply'),'face_normals':False,'bsdf':{'type':'holder-material'}})
        mat=rt.RadioMaterial('regolith',relative_permittivity=3.411,conductivity=.0003562,thickness=10.,scattering_coefficient=.632,xpd_coefficient=.1,scattering_pattern='lambertian')
        scene=rt.load_scene();scene.edit(add=rt.SceneObject(mi_mesh=ground,name='terrain',radio_material=mat))
        scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
        scene.rx_array=scene.tx_array;scene.add(rt.Transmitter('tx',position=tx.tolist()));scene.frequency=415e6
        rx=mi.load_dict({'type':'ply','filename':str(work/'rx.ply'),'face_normals':True})
        px=grid[rec['patch_map']].reshape(16,49)[rec['windows']].ravel();ids,w,valid=look;ids3=ids[px];w3=w[px]
        unique=np.unique(ids3[w3>1e-7]);alpha=cfg.get('phase',0.)
        selected=indices(rv[fr[unique]].astype('float64')-tx,cfg.get('padding',1.35),N,alpha)
        solver,module=variant_solver(cfg);power=np.zeros(len(fr),'float64')
        for st in range(0,len(selected),1_000_000):
            rays=selected[st:st+1_000_000];M=len(rays)
            def lattice(n):
                assert n==M
                ns=mi.Float64(mi.UInt(rays));gold=(1+dr.sqrt(mi.Float64(5.)))/2
                u=ns/gold;u-=dr.floor(u)
                if alpha:
                    u+=alpha/(2*np.pi);u-=dr.floor(u)
                return mi.Point2f(u,ns/(N-1))
            module.fibonacci_lattice=lattice
            res=solver(scene,measurement_surface=rx,samples_per_tx=M,max_depth=0,los=True,specular_reflection=False,diffuse_reflection=False,refraction=False,seed=13)
            power+=res.path_gain.numpy()[0].astype('float64')*(M/N)
        direct=power[ids3]*C;average=(direct*w3).sum(1)
        center=(rv[fr[ids3]].astype('float64').mean(2)*w3[:,:,None]).sum(1)
        invref=1/np.square(center-tx).sum(1).clip(.0625)
        ref=average/invref
        assert np.isfinite(direct).all() and np.all(direct>=0)
        tmp=dest.with_suffix('.part.npz')
        np.savez(tmp,pixel=px,native=direct,weights=w3,ids=ids3,valid=valid[px],average=average,ratio=ref,tx=tx,center=center,directions=len(selected),config=cfgname)
        tmp.replace(dest)
        print(args.stage,j+1,len(tasks),i,cfgname,'rays',len(selected),'seconds',round(time.monotonic()-t0,1),flush=True)
        del scene,ground,rx,solver,res
    (A/f'{args.stage}_shard{args.shard}_done.json').write_text(json.dumps(dict(total=len(tasks),seconds=time.monotonic()-t0,configs=cfgs))+'\n')
if __name__=='__main__':main()
