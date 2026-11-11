"""Wave-dependent geometric inputs, on the prediction grid. No power labels, masks, or parent model output."""
import json,sys,csv
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss';D=R.parent/'data'
def read_compact():
    dst={}
    for k in ('c73_compact','c73_dev_extra_compact'):
        for path in sorted((R/'features'/k).glob('part*')):
            if not (path/'complete.json').exists():continue
            v=np.load(path/'features.npy',mmap_mode='r')
            dst.update({n:(v,j) for j,n in enumerate(json.load(open(path/'indices.json')))})
    return dst
def read_flow(stage):
    dst={}
    for p in sorted(A.glob(stage+'_flow[0-9]')):
        rows=json.load(open(p/'done.json'))['ids'];v=np.load(p/'field.npy',mmap_mode='r')
        dst.update({n:(v,j) for j,n in enumerate(rows)})
    return dst
def terrain_view(h,r,c,px,compact,flow,rot=False):
    """Angles in the raster coordinate system; source positions in the registered native metric grid."""
    y,x=np.divmod(px,256);y=y.astype('float64');x=x.astype('float64');sy=y-r;sx=x-c
    distxy=np.maximum(.25,np.hypot(sx,sy));ux=sx/distxy;uy=sy/distxy
    if rot:ux,uy=-uy,ux
    z=float(np.floor(h[min(r+1,255),c])+3);h0=h.reshape(-1)[px];dz=h0+1-z
    dist=np.hypot(distxy,dz);fs=20*np.log10(dist.clip(.25)[:,None]*np.array([415e6,5.775e9])[None,:]*(4*np.pi/299792458))
    # Only the pre-existing geometry columns are selected, not any learned power map.
    co=compact.reshape(45,-1)[:,px].T.astype(np.float32);vis=co[:,16]
    state=np.where(vis>.99,0,np.where(vis>=.01,1,2))
    cols={'range':np.log1p(dist),'vis':vis,'state':state/2,'relz':np.arcsinh(dz/8),'source_slope':np.arcsinh(dz/distxy*10),
          'profile_obstacle':co[:,8],'required':co[:,9],'where':co[:,10],'under':co[:,11],'crossings':co[:,13],
          'prior_incidence':co[:,17],'expected':co[:,18],'span':np.abs(co[:,24]-co[:,23]),
          'visible_sd':co[:,44],'hit_placeholder':co[:,43]}
    # Keep each contributing native face distinct rather than averaging the height gradient first.
    if not hasattr(terrain_view,'lookup'):
        m=np.load(R/'src/c70_centroid_lookup0.npz');terrain_view.lookup=(m['ids'],m['weights'])
    idx,w0=terrain_view.lookup;f=idx[px];w=w0[px];cell=f%65025;cy=cell//255;cx=cell%255
    a=h[cy,cx];b=h[cy,cx+1];cc=h[cy+1,cx];d=h[cy+1,cx+1];p0=f<65025
    gx=np.where(p0,b-a,d-cc);gy=np.where(p0,d-b,cc-a)
    xx=cx+np.where(p0,2/3,1/3);yy=cy+np.where(p0,1/3,2/3)
    sp=256/255;dx=xx*sp-c;dy=yy*sp-(r+1)
    if rot:dx,dy=-dy,dx
    alt=a+gx*(xx-cx)+gy*(yy-cy)+1-z
    r2=dx*dx+dy*dy+alt*alt;norm=np.sqrt(1+(gx*gx+gy*gy)/sp**2)
    cos=(alt-gx*dx/sp-gy*dy/sp)/np.sqrt(r2.clip(.0625))/norm
    count=(4e9/(4*np.pi))*.5*sp*sp*norm*abs(cos)/r2.clip(.0625)
    m=lambda v:(v*w).sum(1)
    msign=m(cos>0);gf=m(np.exp(-(.5*cos/.012)**2))
    cols.update(graze=gf,shot=m(1/(1+count/8)),small_mass=m(abs(cos)<.0075),
                angular_span=m(np.log1p(abs(cos)*100)),min_weighted=np.min(np.where(w>.05,abs(cos),1),1),
                sign_mixed=4*msign*(1-msign),cos_dir=m(np.arcsinh(cos/.025)),cos_spread=m(np.minimum(3,abs(cos-m(cos)[:,None])*40)),
                face_entropy=-m(np.log(w.clip(1e-9))),native_width=np.log1p(np.sum(w*w*count,1)))
    for tag,xx2,yy2 in [('rad',ux,uy),('side',-uy,ux)]:
        for radius in (1,3,8,24):
            hp=map_coordinates(h,[y+yy2*radius,x+xx2*radius],order=1,mode='nearest',prefilter=False)
            hm=map_coordinates(h,[y-yy2*radius,x-xx2*radius],order=1,mode='nearest',prefilter=False)
            cols[tag+'_s'+str(radius)]=np.arcsinh((hp-hm)/radius*3)
            cols[tag+'_curve'+str(radius)]=np.arcsinh((hp+hm-2*h0)/radius*5)
    # Opposing ridges seen from RX and angular margins above/below the TX->RX ray.
    tup=[]
    for vx,vy in [(-ux,-uy),(ux,uy),(-uy,ux),(uy,-ux)]:
        rr=np.array([1,2,3,5,9,16,28,48])
        query_y=(y[:,None]+vy[:,None]*rr);query_x=(x[:,None]+vx[:,None]*rr)
        inside=(query_y>=0)&(query_y<=255)&(query_x>=0)&(query_x<=255)
        hh=map_coordinates(h,[query_y,query_x],order=1,mode='nearest',prefilter=False)
        angles=(hh-h0[:,None]-1)/rr
        peak=np.maximum(-2,np.where(inside,angles,-2).max(1))
        tup.append(peak)
    cols.update(horizon_in=np.arcsinh(8*(tup[0]+dz/distxy)),horizon_out=np.arcsinh(8*tup[1]),
                horizon_wall=np.arcsinh(8*np.maximum(tup[2],tup[3])),horizon_asym=np.arcsinh(8*(tup[2]-tup[3])))
    f=flow.reshape(-1,10)[px].astype('float32')
    for j,name in enumerate(['first','unshift','shift','offset_delta','landed_count','landed_rays','first_area','weak_area','weak_area_zero','flow_change']):
        cols[name]=f[:,j]
    # rot changes projected local normals, not existing global geometry. It is only a matched local-angle control.
    return cols,fs.astype('float32')
def collect(stage,dense=False):
    plan=json.load(open(A/'patches.json')) if dense else None
    geo=np.load(A/(stage+'_geo.npz'));ids=geo['ids'];px=geo['pixels']
    if dense:
        selected=set(plan['terrain_indices']);m=np.array([int(i) in selected for i in ids]);ids=ids[m]
        px=np.array([plan['maps'][str(i)]['px'] for i in ids]);stage='confirm'
    hh=np.load(D/'train_hm.npy',mmap_mode='r');rows=list(csv.DictReader(open(D/'train_index.csv')))
    co=read_compact();fl=read_flow(stage);all_x=[];rot_x=[];f0=[]
    for k,i in enumerate(ids):
        row=rows[i];ca,cj=co[i];fo,fj=fl[i]
        h=hh[int(row['hm_row'])];args=(h,int(row['tx_row']),int(row['tx_col']),px[k],ca[cj],fo[fj])
        v,fs=terrain_view(*args);rot,_=terrain_view(*args,rot=True)
        x=np.stack(list(v.values()),-1);xx=np.stack([rot[k] for k in v],-1)
        assert np.isfinite(x).all() and np.isfinite(xx).all()
        all_x.append(x.astype('float32'));rot_x.append(xx.astype('float32'));f0.append(fs)
    tag='patch' if dense else stage
    np.savez_compressed(A/(tag+'_features.npz'),ids=ids,pixels=px,names=list(v),x=np.stack(all_x),rot=np.stack(rot_x),fs=np.stack(f0))
    print(tag,len(ids),len(px[0]),len(v))
if __name__=='__main__':collect('confirm' if sys.argv[1]=='patch' else sys.argv[1],sys.argv[1]=='patch')
