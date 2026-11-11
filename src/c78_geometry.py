"""Candidate-mesh sensitivity and conservative certificates, with no label access."""
import argparse,ctypes,csv,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,DATA,atomic_json
from src.c70_geometry import qpoints
import numpy as np
import torch
from torch.nn import functional as F
A=R/'audit/c78_clear';S=256/255
VIS=['nom9','nom64','anti64','original_txz','interp_txz','low_endpoints','high_endpoints','tx_xminus','tx_xplus','tx_yminus','tx_yplus','smooth2','smooth4']
COLUMNS=VIS+['certificate_nominal','certificate_altitude_diagonals','certificate_wide','distance_min','distance_max','nominal_distance','cos_face_abs','expected_primary_count','quad_support']
def planes(h,anti):
    a,b,c,d=(v.reshape(-1) for v in (h[:-1,:-1],h[:-1,1:],h[1:,:-1],h[1:,1:]))
    u=torch.cat((a,b+c-d)) if anti else torch.cat((a,a))
    sx=torch.cat((b-a,d-c));sy=torch.cat((c-a,d-b)) if anti else torch.cat((d-b,c-a))
    size=.5*S*S*torch.sqrt(1+(sx/S).square()+(sy/S).square())
    return torch.stack((u,sx,sy,size),1).contiguous()
def subtriangles():
    a=np.eye(3,dtype=np.float32);ab=(a[0]+a[1])/2;bc=(a[1]+a[2])/2;ca=(a[2]+a[0])/2
    return torch.tensor(np.array([[a[0],ab,ca],[ab,a[1],bc],[ca,bc,a[2]],[ab,bc,ca]]),device='cuda')
class Geometry:
    def __init__(self):
        self.ids=[];self.w=[]
        for diag in (0,1):
            data=np.load(R/f'src/c70_centroid_lookup{diag}.npz')
            self.ids.append(torch.tensor(data['ids'],dtype=torch.int32,device='cuda').contiguous())
            self.w.append(torch.tensor(data['weights'],dtype=torch.float32,device='cuda').contiguous())
        self.q=torch.cat([qpoints(i) for i in [3,8,4]]);self.sub=subtriangles()
        self.lib=ctypes.CDLL(str(R/'src/libc78_geometry.so'));self.fn=self.lib.geometry78
        P=ctypes.c_void_p;I=ctypes.c_int;T=ctypes.c_float
        self.fn.argtypes=[P]*10+[I,P,P]+[T]*6+[P,P,P];self.fn.restype=I
    @torch.no_grad()
    def compute(self,H,r,c,pixels):
        ht=torch.as_tensor(np.array(H,copy=True),device='cuda',dtype=torch.float32)
        g0,g1=planes(ht,False),planes(ht,True)
        smooth=[]
        for side in [128,64]:
            low=F.interpolate(ht[None,None],size=(side,side),mode='bilinear',align_corners=True)
            smooth.append(planes(F.interpolate(low,size=(256,256),mode='bilinear',align_corners=True)[0,0],False))
        corners=torch.stack((ht[:-1,:-1],ht[:-1,1:],ht[1:,:-1],ht[1:,1:]),-1).contiguous()
        px=torch.as_tensor(pixels.astype(np.int32),device='cuda');n=len(px);x=c*255/256;y=(r+1)*255/256
        z=float(np.floor(H[min(255,r+1),c])+3);zo=float(np.floor(H[r,c])+3)
        cx=min(254,int(x));cy=min(254,int(y));u=x-cx;v=y-cy
        zi=float(np.floor((1-v)*((1-u)*H[cy,cx]+u*H[cy,cx+1])+v*((1-u)*H[cy+1,cx]+u*H[cy+1,cx+1]))+3)
        vis=torch.empty((n,13,3),device='cuda');cert=torch.empty((n,3,2,3,4,4),device='cuda')
        params=[g0,g1,*smooth,corners,*self.ids,*self.w,px]
        err=self.fn(*[t.data_ptr() for t in params],n,self.q.data_ptr(),self.sub.data_ptr(),x,y,z,zo,zi,float(H.max()),vis.data_ptr(),cert.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(err)
        w=self.w[0][px.long()];w1=self.w[1][px.long()]
        agg=(vis*w[:,None]).sum(2);agg[:,2]=(vis[:,2]*w1).sum(1)
        agg[:,0]=(vis[:,0].half().float()*w).sum(1).half().float()
        safe=cert[...,0].amin((2,3,4))
        boundary=((px%256)==0)|((px%256)==255)|((px//256)==0)|((px//256)==255)
        safe[boundary]=0
        ranges=cert[:,2];rmax=ranges[...,1].amax((1,2,3));rmin=ranges[...,2].amin((1,2,3)).clamp_min(.001)
        fid=self.ids[0][px.long()].long();cell=fid%65025;fx=torch.where(fid<65025,2/3,1/3);fy=1-fx
        xc=cell%255+fx;yc=cell//255+fy;g=g0[fid]
        dz=g[:,:,0]+g[:,:,1]*fx+g[:,:,2]*fy+1-z;dx=(xc-x)*S;dy=(yc-y)*S;dist=(dx.square()+dy.square()+dz.square()).clamp_min(.001).sqrt()
        cos=(dz-g[:,:,1]*dx/S-g[:,:,2]*dy/S).abs()/(dist*torch.sqrt(1+(g[:,:,1]/S).square()+(g[:,:,2]/S).square()))
        vals=torch.cat((agg,safe,rmin[:,None],rmax[:,None],torch.stack(((dist*w).sum(1),(cos*w).sum(1),(4e9/(4*np.pi)*g[:,:,3]*cos/dist.square()*w).sum(1),~boundary),1)),1)
        assert torch.isfinite(vals).all()
        assert not ((safe[:,2]>.5)&(agg[:,:11].amin(1)<.999)).any(),'a certified ray was reclassified'
        assert not ((safe[:,0]>.5)&(agg[:,:2].amin(1)<.999)).any()
        return vals.cpu().numpy()
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val'],default='train');p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=3);p.add_argument('--check',action='store_true');a=p.parse_args()
    gpu=gpu_guard();eng=Geometry()
    if a.check:
        px=np.array([100*256+90,100*256+105,100*256+155,100*256+175,140*256+170],np.int32)
        flat=np.zeros((256,256),np.float32);vv=eng.compute(flat,100,100,px);assert np.allclose(vv[:,:13],1,atol=.00001) and np.all(vv[:,13:16]==1)
        ridge=np.broadcast_to(np.maximum(0,12-np.abs(np.arange(256)-128)),(256,256)).astype(np.float32)
        vv=eng.compute(ridge,100,100,px);assert (vv[:2,0]>.999).all() and (vv[2:,:11]<.01).all() and (vv[2:,13:16]==0).all()
        print('flat, known clear and blocked ridge cells verified',flush=True);return
    panel=np.load(A/f'{a.split}_panel.npz');pos=np.arange(len(panel['ids']))[a.shard::a.of];rows=list(csv.DictReader((DATA/f'{a.split}_index.csv').open()));hh=np.load(DATA/f'{a.split}_hm.npy',mmap_mode='r')
    out=A/f'{a.split}_geom_{a.shard}of{a.of}';out.mkdir(exist_ok=False);np.save(out/'positions.npy',pos)
    m=np.lib.format.open_memmap(out/'geometry.npy','w+',dtype=np.float32,shape=(len(pos),panel['pixels'].shape[1],len(COLUMNS)))
    ids=panel['ids'];pp=panel['pixels'];t=time.monotonic()
    for k,j in enumerate(pos):
        r=rows[ids[j]];m[k]=eng.compute(hh[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']),pp[j])
        if k%40==0 or k+1==len(pos):
            rr=dict(maps=k+1,total=len(pos),seconds=round(time.monotonic()-t,1));atomic_json(out/'progress.json',rr);print(rr,flush=True)
    m.flush();atomic_json(out/'complete.json',dict(gpu=gpu,columns=COLUMNS,source='H + TX only',maps=len(pos)))
if __name__=='__main__':main()
