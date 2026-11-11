"""TRAIN-side multibounce probe using the organizer's solver, in an isolated CPU process."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')
import drjit as dr
import sionna.rt as rt
from scipy.ndimage import uniform_filter

R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver'
def meshfile(path, vertices, faces):
    typ=np.dtype([('n','u1'),('idx','<i4',(3,))])
    data=np.empty(len(faces),dtype=typ);data['n']=3;data['idx']=faces
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n'
    with path.open('wb') as f:
        f.write(header.encode('ascii'));vertices.astype('<f4').tofile(f);data.tofile(f)

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--of',type=int,default=3)
    p.add_argument('--stage',default='screen');p.add_argument('--rays',type=int,default=4000000)
    a=p.parse_args();dr.set_thread_count(8)
    assert rt.__version__=='1.1.0' and mi.variant()=='llvm_ad_mono_polarized'
    plan=json.load(open(A/'protocol.json'))['panel']
    # Select by terrain names and fixed RNG from the pre-written partition; no outcome-based selection.
    rng=np.random.default_rng(79435)
    records=[]
    for group,limit in [('fit',48),('select',32),('confirm',64)]:
        sub=[v for v in plan if v['group']==group];per=rng.permutation(len(sub))[:limit]
        if (group=='confirm')==(a.stage=='confirm'):records.extend(sub[k] for k in per)
    records=records[a.shard::a.of]
    panel=np.load(R/'audit/c78_clear/train_panel.npz')
    rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));hh=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
    idx=np.load(R/'src/c70_centroid_lookup0.npz')
    ids=idx['ids'];weights=idx['weights'];support=weights.sum(1)>.5
    ya,xa=np.mgrid[:256,:256];axis=np.linspace(-128,128,256,dtype='float32')
    cell=(ya[:-1,:-1]*256+xa[:-1,:-1]).ravel()
    # World y decreases with image row. Winding is chosen to retain the upward geometric normal.
    faces=np.concatenate([np.stack([cell,cell+257,cell+1],1),np.stack([cell,cell+256,cell+257],1)],0)
    vertices=np.stack([axis[xa],-axis[ya],xa*0],-1).astype('float32').reshape(-1,3)
    out=A/('rt_'+a.stage+str(a.shard));out.mkdir(exist_ok=True)
    meta=dict(rays_each=a.rays,solver=rt.__version__,backend=mi.variant(),registration='world x=col-128, y=127-row, z=floor(H[row+1,col])+3, 256/255 vertex spacing, RX z+1, upward faces',
              bands=['415','58'],depth=100,seeds=[13,47],primary='2 seeds share the same primary lattice; independent interactions after first hit',
              features='baseband array per valid H-TX point: direct reference, each-band multibounce sum, positive-excess over direct, coverage and log-domain interpolation at several support scales')
    solver=rt.RadioMapSolver();t0=time.monotonic()
    for j,rec in enumerate(records):
        dest=out/(str(rec['pos'])+'.npz')
        if dest.exists():continue
        row=rows[rec['index']];h=hh[int(row['hm_row'])];tr,tc=int(row['tx_row']),int(row['tx_col'])
        z=np.floor(h[min(tr+1,255),tc])+3;tx=np.array([tc-128,127-tr,z])
        vertices[:,2]=h.ravel();meshfile(out/'ground.ply',vertices,faces)
        rv=vertices.copy();rv[:,2]+=1.;meshfile(out/'receiver.ply',rv,faces)
        mat=rt.RadioMaterial('regolith',relative_permittivity=3.411,conductivity=.0003562,thickness=10.,scattering_coefficient=.632,xpd_coefficient=.1,scattering_pattern='lambertian')
        ground=mi.load_dict({'type':'ply','filename':str(out/'ground.ply'),'face_normals':False,'bsdf':{'type':'holder-material'}})
        obj=rt.SceneObject(mi_mesh=ground,name='terrain',radio_material=mat)
        scene=rt.load_scene();scene.edit(add=obj)
        scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
        scene.rx_array=scene.tx_array;scene.add(rt.Transmitter('tx',position=tx.tolist()))
        rx=mi.load_dict({'type':'ply','filename':str(out/'receiver.ply'),'face_normals':True})
        cent=rv[faces].mean(1);r2=np.square(cent-tx).sum(1).clip(.0625)
        pix=panel['pixels'][rec['pos']];res=[];all_count=[]
        for b,(freq,sigma) in enumerate([(415e6,.0003562),(5.775e9,.009324)]):
            scene.frequency=freq;mat.conductivity=sigma
            norm=(4*np.pi*freq/299792458.)**2*r2
            if b==0:
                ro=solver(scene,measurement_surface=rx,samples_per_tx=a.rays,max_depth=0,los=True,specular_reflection=False,diffuse_reflection=False,refraction=False,seed=13)
                direct=ro.path_gain.numpy()[0].astype('float64')*norm
                assert np.all(direct>=0) and np.isfinite(direct).all()
            sols=[]
            for seed in (13,47):
                rm=solver(scene,measurement_surface=rx,samples_per_tx=a.rays,max_depth=100,los=True,specular_reflection=True,diffuse_reflection=True,refraction=True,seed=seed)
                sols.append(rm.path_gain.numpy()[0].astype('float64')*norm)
            total=(sols[0]+sols[1])*.5
            assert (total>=0).all() and np.isfinite(total).all()
            extra=np.maximum(0,total-direct);all_count.append([float((x>0).mean()) for x in (direct,total,extra)])
            col=[];w=support.reshape(256,256).astype(float)
            for field,tag in [(direct,'direct'),(total,'total'),(extra,'extra'),(sols[0],'seedA'),(sols[1],'seedB')]:
                power=(field[ids]*weights).sum(1).reshape(256,256)
                hit=((field[ids]>0)*weights).sum(1).reshape(256,256)
                for size in (1,3,7,15):
                    den=uniform_filter(w,size=size,mode='nearest').clip(.02)
                    ep=uniform_filter(power,size=size,mode='nearest')/den
                    en=uniform_filter(hit,size=size,mode='nearest')/den
                    col.extend([(-10*np.log10(np.maximum(ep,.00001))).clip(-35,50).ravel()[pix],en.ravel()[pix]])
            res.append(np.stack(col,1).astype('float32'))
        np.savez(dest,features=np.stack(res),coverage=all_count)
        if j%5==0 or j+1==len(records):print(j+1,len(records),'elapsed',round(time.monotonic()-t0,1),'coverage',np.round(all_count,3).tolist(),flush=True)
        del scene,ground,obj,rx,rm,ro
    (out/'complete.json').write_text(json.dumps(dict(meta,records=records,seconds=time.monotonic()-t0),indent=2)+'\n')
if __name__=='__main__': main()
