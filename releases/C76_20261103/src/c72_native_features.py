"""Candidate receiver triangles. Keep the two triangles in each grid square separate."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import DATA,gpu_guard,atomic_json
from src.c70_geometry import GeometricMaps
from src.c69_registered import RegisteredTransport
import numpy as np
import torch
from torch.nn import functional as F

class NativeFeatures:
    def __init__(self):
        self.geo=GeometricMaps()
        self.paths=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=128)
        self.centers=self.paths.xy[self.paths.faces].mean(1)
    @torch.no_grad()
    def __call__(self,h,tr,tc):
        direct,spec,diff,t,x,y,z,emit=self.paths.raw(h,tr,tc,return_emission=True)
        geom,_=self.geo.native(t,tr,tc)
        rz=t.reshape(-1)[self.paths.faces].mean(1)+1
        d2=((self.centers[:,0]-x*self.paths.spacing).square()+(self.centers[:,1]-y*self.paths.spacing).square()+(rz-z).square()).clamp_min(.0625)
        power=torch.cat((direct[:,:2],spec[:,:2],diff[:,:2],direct[:,:2]+spec[:,:2]+diff[:,:2],emit[:,:2]),1)
        powers=torch.log10((power*d2[:,None]).clamp(1e-12,1e8))/4
        stats=torch.stack((direct[:,2],torch.log1p(direct[:,3])/3,torch.log1p(spec[:,2]),
            diff[:,0].square()/diff[:,2].clamp_min(1e-30)/len(self.paths.dirs),
            diff[:,1].square()/diff[:,3].clamp_min(1e-30)/len(self.paths.dirs),
            emit[:,2],emit[:,3],.5*torch.log(d2)/6,(rz-z)/30,
            torch.ones_like(rz)*(z-3-float(h[tr,tc]))/3),1)
        v=torch.cat((geom,powers,stats),1).clamp(-10,10)
        # First 255x255 lattice, then the other triangle of each square.
        f=v.shape[1];out=v.reshape(2,255,255,f).permute(0,3,1,2).reshape(2*f,255,255)
        return F.pad(out[None],(0,1,0,1),mode='replicate')[0].contiguous()
def main():
    p=argparse.ArgumentParser();p.add_argument('--dir',default='features/c72_native');p.add_argument('--panel',default='')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=1);a=p.parse_args();gpu=gpu_guard()
    panel=json.load(open(ROOT/'audit/c66_transport/panels.json'));ids=json.load(open(ROOT/a.panel)) if a.panel else panel['dev_maps']+panel['fit_maps'];ids=ids[a.shard::a.of]
    out=ROOT/a.dir/f'part{a.shard}of{a.of}';out.mkdir(parents=True,exist_ok=False);atomic_json(out/'indices.json',ids)
    rows=list(csv.DictReader((DATA/'train_index.csv').open()));heights=np.load(DATA/'train_hm.npy',mmap_mode='r');pred=NativeFeatures()
    t=time.monotonic()
    for j,i in enumerate(ids):
        r=rows[i];v=pred(heights[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col'])).half().cpu().numpy()
        assert np.isfinite(v).all() and v.shape==(94,256,256)
        if j==0:arr=np.lib.format.open_memmap(out/'features.npy','w+',dtype=np.float16,shape=(len(ids),*v.shape))
        arr[j]=v
        if j%80==0 or j+1==len(ids):
            h=dict(maps=j+1,total=len(ids),seconds=round(time.monotonic()-t,1));atomic_json(out/'progress.json',h);print(h,flush=True)
    arr.flush();atomic_json(out/'complete.json',dict(vars(a),gpu=gpu,num_channels=94,source='H and TX exclusively'))
if __name__=='__main__':main()
