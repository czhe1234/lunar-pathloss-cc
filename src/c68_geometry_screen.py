"""FIT-only identification of a physical geometry using per-map high-frequency and shadow-edge consistency."""
import argparse,sys,csv,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard,DATA,atomic_json
import numpy as np
import torch
import torch.nn.functional as F
from src.c67_physics import Transport
from scipy.ndimage import gaussian_filter
from src.c67_dense import frozen_maps

class Adjustable(Transport):
    def with_height(self,h,tr,tc,policy,dy,dx):
        qx=tc*255/256.;qy=(tr+1)*255/256.
        rx=np.clip(qx,0,255);ry=np.clip(qy,0,255)
        ix=int(np.floor(rx));iy=int(np.floor(ry))
        def bilinear(x,y):
            x=float(np.clip(x,0,255));y=float(np.clip(y,0,255));c=int(x);r=int(y);a=x-c;b=y-r
            return ((1-a)*(1-b)*h[r,c]+a*(1-b)*h[r,min(255,c+1)]
                +(1-a)*b*h[min(255,r+1),c]+a*b*h[min(255,r+1),min(255,c+1)])
        choices={'up2':h[min(tr+2,255),tc],'plusleft':h[min(tr+1,255),max(tc-1,0)],'plusright':h[min(tr+1,255),min(tc+1,255)],'integer':h[tr,tc],'sampled':bilinear(rx+dx,ry+dy),'nearest':h[int(np.round(ry)),int(np.round(rx))],
          'plusrow':h[min(255,tr+1),tc],'floorxy':h[iy,ix],'minuscol':h[tr,max(tc-1,0)],
          'continuous':bilinear(rx+dx,ry+dy),'nearestfloorRC':h[iy,int(np.round(rx))]}
        height=choices.get(policy,h[tr,tc]);z=height if policy=='continuous' else np.floor(height)
        s=torch.as_tensor(np.array(h,copy=True),device='cuda',dtype=torch.float32)
        if dx or dy:
            xx=self.xy[:,0]/self.spacing+dx;yy=self.xy[:,1]/self.spacing+dy
            grid=torch.stack((xx/127.5-1,yy/127.5-1),1).reshape(1,256,256,2)
            s=F.grid_sample(s[None,None],grid,align_corners=True,padding_mode='border')[0,0]
        return s,float(z+3)
    def render(self,h,tr,tc,policy='integer',dy=0.,dx=0.,dz=0.):
        s,z=self.with_height(h,tr,tc,policy,dy,dx);z+=dz
        g,vn=self.scene(s);size=len(self.faces);d=torch.empty((size,4),device='cuda');e=torch.empty_like(d)
        spec=torch.zeros((size,3),device='cuda');a=torch.empty_like(d)
        x=tc*255/256 if self.physical else tc;y=(tr+1)*255/256 if self.physical else tr
        err=self.lib.transport(g.data_ptr(),vn.data_ptr(),self.material.data_ptr(),self.primary.data_ptr(),len(self.primary),self.spec.data_ptr(),len(self.spec),
            self.dirs.data_ptr(),len(self.dirs),x,y,z,self.spacing,float(h.max())+2,self.anti,self.smooth,
            d.data_ptr(),e.data_ptr(),spec.data_ptr(),a.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(err)
        total=d[:,:2]+spec[:,:2]+a[:,:2]
        r2=(self.qx-x).square().add((self.qy-y).square()).mul(self.spacing**2)+(s.reshape(-1)+1-z).square()
        lp=torch.log10(total.clamp_min(1e-22))
        pl=-(self.interpolate(lp)+torch.log10(r2.clamp_min(.0625))[:,None])*10.
        mixed=-10*torch.log10((self.interpolate(total)*r2[:,None]).clamp_min(1e-18))
        return torch.cat((pl,mixed,self.interpolate(d[:,2:3])),1).T.reshape(5,256,256)
def test(refine=False):
    gpu=gpu_guard()
    panels=json.load(open(ROOT/'audit/c66_transport/panels.json'))
    ids=np.random.default_rng(5693).choice(panels['fit_first'],96,replace=False).tolist()
    if refine:ids=np.random.default_rng(47715).choice(sorted(set(panels['fit_first'])-set(ids)),96,replace=False).tolist()
    rows=list(csv.DictReader((DATA/'train_index.csv').open()))
    hh=np.load(DATA/'train_hm.npy',mmap_mode='r');labels=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
    masks=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']];blind=frozen_maps(ids,'fit')
    # Five stable input-only statistics. Calibrate each regression on 48 complete FIT terrains, score next 48 FIT terrains only.
    ys=[];ws=[];mm=[];base=[]
    xy=np.arange(256);yy,xx=np.meshgrid(xy,xy,indexing='ij')
    pix=np.array([np.random.default_rng(2597+i).choice(65536,3072,replace=False) for i in ids])
    for j,i in enumerate(ids):
        row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);h=hh[int(row['hm_row'])]
        fs=20*np.log10(4*np.pi*np.sqrt((xx-tc)**2+(yy-tr)**2+(h+1-(h[tr,tc]+3))**2).clip(.25)[None]*np.array([415e6,5.775e9])[:,None,None]/299792458)
        y=np.stack([labels[b][i] for b in range(2)]).astype('float32')
        mask=np.stack([np.unpackbits(m[i]).reshape(256,256) for m in masks]).astype(bool)
        p=blind[j]
        ys.append((y-p).reshape(2,-1)[:,pix[j]].T);ws.append(mask.reshape(2,-1)[:,pix[j]].T)
        mm.append((y-fs).reshape(2,-1)[:,pix[j]].T);base.append((p-fs).reshape(2,-1)[:,pix[j]].T)
    ys=np.array(ys);ws=np.array(ws);mm=np.array(mm);base=np.array(base)
    configs=[]
    for grid in [1,0]:
        for anti in [0,1]:
            for smooth in [1,0]:configs.append((grid,anti,smooth,'integer',0,0,0))
    for z in ['sampled','nearest','plusrow','floorxy','minuscol','continuous','nearestfloorRC']:
        configs.append((1,0,1,z,0,0,0))
    for delta in [-.5,-.25,.25,.5]:configs.append((1,0,1,'integer',0,0,delta))
    for dy in [-1,-.5,0,.5,1]:
        for dx in [-1,-.5,0,.5,1]:
            if dy or dx:configs.append((1,0,1,'integer',dy,dx,0))
    if refine:
        configs=[(1,0,1,'integer',0,0,0),(1,0,1,'plusrow',0,0,0)]
        for policy in ['up2','plusleft','plusright']:
            configs.append((1,0,1,policy,0,0,0))
        for dy in [-.5,0,.5]:
            for dx in [-.5,0,.5]:
                if dy or dx:configs.append((1,0,1,'plusrow',dy,dx,0))
        for dz in [-.5,-.25,-.10,.10,.25,.5]:configs.append((1,0,1,'plusrow',0,0,dz))
        configs.extend([(g,a,n,'plusrow',0,0,0) for g,a,n in [(1,0,0),(1,1,1),(1,1,0),(0,0,1),(0,1,1)]])
    tag='c70_refine' if refine else 'c68'
    # Add direct-power and high-pass reflectance terms to a common baseline control, fitted only on calibration terrains.
    engines={};results=[];start=time.monotonic()
    root=ROOT/'audit/c66_transport';values=np.lib.format.open_memmap(root/f'{tag}_screen_features.npy','w+',dtype=np.float16,shape=(len(configs),len(ids),3072,9))
    for k,(grid,anti,smooth,policy,dy,dx,dz) in enumerate(configs):
        key=(grid,anti)
        if key not in engines:engines[key]=Adjustable(physical=grid,anti=anti,smooth=smooth,direct=1,spec=2,diffuse=48)
        eng=engines[key];eng.smooth=smooth
        for j,i in enumerate(ids):
            row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);h=hh[int(row['hm_row'])]
            v=eng.render(h,tr,tc,policy,dy,dx,dz)
            filt=F.avg_pool2d(F.pad(v[:4][None],(2,2,2,2),mode='replicate'),5,stride=1)[0]
            feat=torch.cat((v[:4].clamp(-15,85),(v[:4]-filt).clamp(-24,24),v[4:]),0)
            values[k,j]=feat.reshape(9,-1)[:,pix[j]].T.half().cpu().numpy()
        f=values[k].astype('float32');scores=[];ba=[]
        for b in [0,1]:
            bp=base[:,:,b]; w=ws[:,:,b];y=ys[:,:,b];fc=f[:,:,[b,2+b,4+b,6+b,8]]
            phi=np.stack((np.ones_like(bp),bp,np.clip(bp-5,0,100),np.clip(bp-16,0,100),np.clip(bp-30,0,100)),axis=2)
            sig=1/(1+np.exp(np.clip((bp-10)/4,-30,30)))
            feat=np.concatenate((phi,fc,fc*sig[:,:,None]),2)
            rr=[]
            for xxm in [phi,feat]:
                xa=xxm[:48][w[:48]].astype('float64');ya=y[:48][w[:48]].astype('float64')
                beta=np.linalg.solve(xa.T@xa+np.eye(xa.shape[1])*10,xa.T@ya)
                e=y[48:][w[48:]]-xxm[48:][w[48:]]@beta;rr.append(float(np.sqrt((e*e).mean())))
            score=f[:,:,8][48:];clear=(mm[:,:,b][48:]<2)&w[48:];shadow=(mm[:,:,b][48:]>18)&w[48:]
            ba.append(float(((score[clear]>.5).mean()+(score[shadow]<=.5).mean())/2))
            scores.append(rr)
        results.append(dict(config=configs[k],regression_control_then_candidate=scores,proxy_los_balanced=ba))
        atomic_json(root/f'{tag}_grid_partial.json',dict(indices=ids,fit_terrains=48,holdout_terrains=48,records=results,elapsed=time.monotonic()-start))
        print(k,configs[k],scores,ba,round(time.monotonic()-start,1),flush=True)
    values.flush()
    atomic_json(root/f'{tag}_grid_screen.json',dict(indices=ids,fit_terrains=48,holdout_terrains=48,records=results,elapsed=time.monotonic()-start))
if __name__=='__main__':test('--refine' in sys.argv)
