"""Vectorized individual transparent-receiver trajectories. No target power or labels."""
import csv,json,os,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from c95_alignment import geometry,meshfile,mi,dr
A=R/'audit/c95_alignment';W=A/f'trace_work_pid{os.getpid()}';W.mkdir()
rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));i=20603
hm=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');h=hm[int(rows[i]['hm_row'])]
verts,rv,fg,fr,tx,look=geometry(h,rows[i],{}, {})
origin=next((R/'audit/c93_joint_miss').glob('native[0-9]/20603.npz'));case=np.load(origin)
ids=case['native_ids'];px=case['pixel'];local=np.flatnonzero((abs(px//256-206)<=2)&(abs(px%256-218)<=2))
fids=np.unique(ids[local].ravel());targets=rv[fr[fids]].astype('float64').mean(1)
dirs=targets-tx;dirs/=np.linalg.norm(dirs,axis=1,keepdims=True);directions=mi.Vector3f(dirs.T.astype('float32'))
def track(shift,compensate):
    t=np.asarray(shift,dtype='float32');tag='_'.join(map(str,shift))+str(compensate)
    meshfile(W/(tag+'_g.ply'),verts+t,fg);meshfile(W/(tag+'_r.ply'),rv+t,fr)
    g=mi.load_dict({'type':'ply','filename':str(W/(tag+'_g.ply'))})
    r=mi.load_dict({'type':'ply','filename':str(W/(tag+'_r.ply')),'face_normals':True})
    scene=mi.load_dict({'type':'scene','ground':g,'receiver':r})
    ray=mi.Ray3f(mi.Point3f((tx+t).tolist()),directions);active=mi.Bool(np.ones(len(fids),bool))
    paths=[[] for _ in fids];remaining=[]
    for k in range(128):
        si=scene.ray_intersect(ray,active=active);dr.eval(si)
        yes=active&si.is_valid()&(si.shape==r);valid=yes.numpy()
        if not valid.any():break
        p=si.p.numpy().T.astype(float);fid=si.prim_index.numpy()
        if compensate:
            mag=(1+dr.max(dr.abs(si.p-mi.Point3f(t.tolist()))))*8.940696716308594e-05
            origin=dr.fma(dr.detach(dr.mulsign(mag,dr.dot(si.n,ray.d))),dr.detach(si.n),si.p)
            new=mi.Ray3f(origin,ray.d)
        else:new=si.spawn_ray(ray.d)
        dr.eval(new);delta=(new.o-si.p).numpy().T.astype(float)
        for j in np.flatnonzero(valid):
            rr,cc=divmod(int(fid[j])%65025,255)
            paths[j].append(dict(face=int(fid[j]),row=rr,col=cc,p_local=(p[j]-t).tolist(),offset_norm_m=float(np.linalg.norm(delta[j]))))
        ray=new;active=yes;dr.eval(active)
        remaining.append(int(valid.sum()))
    else:raise RuntimeError('trajectory did not terminate within128receiver crossings')
    print('trace',shift,compensate,'iterations',len(remaining),'hit counts',remaining,flush=True)
    return paths
base=track((0,0,0),False);moved=track((128,128,0),False);fix=track((128,128,0),True)
records=[]
for j in range(len(fids)):
    bs={v['face'] for v in base[j]};ms={v['face'] for v in moved[j]};fs={v['face'] for v in fix[j]}
    records.append(dict(target_face=int(fids[j]),base=base[j],translated=moved[j],compensated=fix[j],
                   changed_faces=len(bs^ms),remaining_changed=len(bs^fs)))
chosen=max(records,key=lambda x:(x['changed_faces']-x['remaining_changed'],len(x['base'])))
out=dict(sample='05489_00',rays=len(records),changed=sum(x['changed_faces']>0 for x in records),
    compensated_changed=sum(x['remaining_changed']>0 for x in records),example=chosen,
    source='Fixed directions to RX face centers. Same H/TX-relative geometry, translated together. No measured power used. Not a map score.')
(A/'ray_trajectory.json').write_text(json.dumps(out,indent=2)+'\n')
print('rays',len(records),'changed',out['changed'],'compensated_changed',out['compensated_changed'],flush=True)
print('example',[(n,[(v['row'],v['col'],round(v['offset_norm_m'],6)) for v in chosen[n]]) for n in ('base','translated','compensated')],flush=True)
