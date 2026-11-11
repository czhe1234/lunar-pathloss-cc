"""Original Sionna mesh-receiver direct-only replay on independently selected contiguous TRAIN patches."""
import argparse,csv,importlib,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from c86_targeted import indices,meshfile,mi,rt,dr
A=R/'audit/c93_joint_miss';D=R.parent/'data'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--of',type=int,default=3);ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
    assert mi.variant()=='llvm_ad_mono_polarized' and rt.__version__=='1.1.0'
    dr.set_thread_count(4);module=importlib.import_module('sionna.rt.radio_map_solvers.radio_map_solver');array=np.load(R/'src/c70_centroid_lookup0.npz')
    scene_ids=array['ids'];weights=array['weights'];axis=np.linspace(-128,128,256,dtype=np.float32);x,y=np.meshgrid(axis,-axis)
    cell=np.arange(65536).reshape(256,256)[:-1,:-1].ravel()
    faces=np.concatenate([np.stack([cell,cell+257,cell+1],1),np.stack([cell,cell+256,cell+257],1)])
    rows=list(csv.DictReader(open(D/'train_index.csv')));height=np.load(D/'train_hm.npy',mmap_mode='r')
    rec=json.load(open(A/'exact_windows.json'))['maps'];p=np.load(A/'patch_features.npz');ids=p['ids'];grid=p['pixels']
    dest=A/(f'native{a.shard}' if a.native_only else f'exact{a.shard}');dest.mkdir(exist_ok=True);start=time.monotonic()
    tasks=list(enumerate(rec))[a.shard::a.of]
    for j,(k,e) in enumerate(tasks):
        i=e['index'];outfile=dest/f'{i}.npz'
        if outfile.exists():continue
        row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);h=height[int(row['hm_row'])]
        tx=np.array([tc-128.,127.-tr,float(np.floor(h[min(tr+1,255),tc])+3)])
        vertices=np.stack([x,y,h],2).reshape(-1,3).astype('float32');up=vertices.copy();up[:,2]+=1
        meshfile(dest/'ground.ply',vertices,faces);meshfile(dest/'rx.ply',up,faces)
        ground=mi.load_dict({'type':'ply','filename':str(dest/'ground.ply'),'face_normals':False,'bsdf':{'type':'holder-material'}})
        mat=rt.RadioMaterial('regolith',relative_permittivity=3.411,conductivity=.0003562,thickness=10.,
                scattering_coefficient=.632,xpd_coefficient=.1,scattering_pattern='lambertian')
        s=rt.load_scene();s.edit(add=rt.SceneObject(mi_mesh=ground,name='terrain',radio_material=mat))
        s.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V');s.rx_array=s.tx_array
        s.add(rt.Transmitter('tx',position=tx.tolist()));s.frequency=415e6
        rx=mi.load_dict({'type':'ply','filename':str(dest/'rx.ply'),'face_normals':True})
        assert ids[e['patch_map']]==i
        px=grid[e['patch_map']].reshape(16,49)[e['windows']].ravel();ids3=scene_ids[px];w=weights[px]
        uf=np.unique(ids3[w>1e-7]);v=up[faces[uf]].astype('float64')-tx
        dist=np.square(up[faces].mean(1).astype('float64')-tx).sum(1).clip(.0625);norm=(4*np.pi*415e6/299792458)**2*dist
        solver=rt.RadioMapSolver();results=[];nray=[];modes=[(0.,1.35),(.013,1.35),(-.017,1.35)]
        if a.native_only:modes=[(0.,1.35)]
        elif k%5==0:modes.append((0.,2.4))
        N=4_000_000_000
        native=[]
        for alpha,pad in modes:
            selected=indices(v,pad,N,alpha);pwr=np.zeros(len(faces),'float64')
            for st in range(0,len(selected),1_000_000):
                rays=selected[st:st+1_000_000];M=len(rays)
                def lattice(n):
                    assert n==M
                    ns=mi.Float64(mi.UInt(rays));gold=(1+dr.sqrt(mi.Float64(5.)))/2
                    u=ns/gold;u=u-dr.floor(u);u+=alpha/(2*np.pi);u-=dr.floor(u)
                    return mi.Point2f(u,ns/(N-1))
                module.fibonacci_lattice=lattice
                signal=solver(s,measurement_surface=rx,samples_per_tx=M,max_depth=0,los=True,specular_reflection=False,diffuse_reflection=False,refraction=False,seed=13)
                pwr+=signal.path_gain.numpy()[0].astype('float64')*(M/N)
            native.append((pwr[ids3]*(4*np.pi*415e6/299792458)**2).copy())
            rel=(pwr*norm)[ids3];gain=(rel*w).sum(1);hit=((pwr[ids3]>0)*w).sum(1)
            direct=(pwr[ids3]*w).sum(1)*(4*np.pi*415e6/299792458)**2
            results.append(np.stack([gain,hit,direct],1));nray.append(len(selected))
        assert np.isfinite(results).all() and (np.asarray(results)>=0).all()
        tmp=outfile.with_suffix('.partial.npz');np.savez(tmp,values=np.stack(results),modes=modes,directions=nray,pixel=px,index=i,native_inv_square=np.stack(native),native_ids=ids3,weights=w);tmp.replace(outfile)
        print('map',j+1,len(tasks),'rays',nray,'seconds',round(time.monotonic()-start,1),flush=True)
    (dest/'done.json').write_text(json.dumps(dict(maps=len(tasks),seconds=time.monotonic()-start,backend=mi.variant(),solver=rt.__version__))+'\n')
if __name__=='__main__':main()
