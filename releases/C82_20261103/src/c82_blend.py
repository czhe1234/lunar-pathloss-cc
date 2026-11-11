"""Can local scale and expert disagreement predict which C76 errors to trust? TRAIN-only fitting."""
import argparse,csv,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
A=R/'audit/c79_receiver'
def smooth(x,w):
    return F.avg_pool2d(x,w,1,w//2,count_include_pad=False)
def field_features(base,a,b,fs,vis,mode,band):
    mean=.5*(a+b);res=b-a;extra=mean-fs
    soft=((extra[:,band:band+1]-3)/8).clamp(-1,2)
    shadow=1-vis;trans=4*vis*shadow
    other=1-band
    # Laplacian pyramid of corrections; the two raw expert predictions are never recomputed or shifted at inference.
    inp=torch.cat([res[:,band:band+1]/8,(mean-base)[:,band:band+1]/8,extra[:,band:band+1]/20,
                   (extra[:,band:band+1]-extra[:,other:other+1])/10],1)
    one=torch.ones_like(soft)
    if mode=='simple':return torch.cat([one,res[:,band:band+1]/8,res[:,band:band+1]/8*shadow],1)
    low=inp
    channels=[]
    for w in (3,7,15,31):
        new=smooth(inp,w);channels.append((low-new).clamp(-6,6));low=new
    channels.extend([low[:,:2].clamp(-6,6),low[:,2:].clamp(-6,6)])
    z=torch.cat(channels,1)
    terrain=torch.cat([one,soft,shadow,trans,soft*shadow],1)
    if mode=='scales':return torch.cat([terrain,z,z*soft,z*shadow],1)
    hi=mean[:,band:band+1]-smooth(mean[:,band:band+1],3)
    v=smooth(hi.square(),5).clamp_min(.00001).sqrt()
    lim=smooth(res[:,band:band+1].abs(),5)/4
    ext=extra[:,band:band+1]/10
    extrema=[]
    for w in (3,7):
        mx=F.max_pool2d(ext,w,1,w//2)-ext
        mn=-F.max_pool2d(-ext,w,1,w//2)-ext
        extrema.extend([mx,mn,mx*soft,mn*soft,mx*shadow,mn*shadow])
    return torch.cat([terrain,z,z*soft,z*shadow,z*lim.clamp(0,2),z*(v/3).clamp(0,2),*extrema],1).clamp(-14,14)
class Data:
    def __init__(self):
        p=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1'
        load=lambda f:np.load(f,mmap_mode='r')
        self.ix=json.load(open(p/'indices.json'))['index_rows']
        self.base=[load(p/f'pl{b}.npy') for b in ('415','58')]
        self.cor=load(R/'runs/c72_observed/full_internal/correction.npy')
        self.graph=load(R/'runs/c73_wide_real/full_internal/prediction.npy')
        self.y=[load(R.parent/f'data/train_pl{b}.npy') for b in ('415','58')]
        self.w=[load(R.parent/f'data/train_mask{b}.npy') for b in ('415','58')]
        self.h=load(R.parent/'data/train_hm.npy');self.rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
        names={r['terrain']:r['group'] for r in json.load(open(A/'protocol.json'))['panel']}
        self.groups=[names[self.rows[i]['terrain_id']] for i in self.ix]
        self.cv={}
        for d in ('c73_compact','c73_dev_extra_compact'):
            for s in (R/'features'/d).glob('part*'):
                ar=load(s/'features.npy');self.cv.update({i:(ar,j) for j,i in enumerate(json.load(open(s/'indices.json')))})
        self.px=np.load(R/'audit/c78_clear/train_panel.npz')['pixels']
    def batch(self,j):
        ids=[self.ix[k] for k in j];to=lambda a:torch.from_numpy(np.ascontiguousarray(a)).cuda()
        base=to(np.stack([q[j] for q in self.base],1)).float();local=base+to(self.cor[j]).float();graph=to(self.graph[j]).float()
        y=to(np.stack([q[ids] for q in self.y],1)).float()
        w=to(np.stack([np.unpackbits(q[ids],axis=1).reshape(-1,256,256) for q in self.w],1)).bool()
        vis=to(np.stack([self.cv[i][0][self.cv[i][1],16] for i in ids]))[:,None].float()
        h=to(self.h[[int(self.rows[i]['hm_row']) for i in ids]]).float()
        tr=to(np.array([int(self.rows[i]['tx_row']) for i in ids]));tc=to(np.array([int(self.rows[i]['tx_col']) for i in ids]))
        yy,xx=torch.meshgrid(torch.arange(256,device='cuda'),torch.arange(256,device='cuda'),indexing='ij')
        z=h[torch.arange(len(j),device='cuda'),(tr+1).clamp_max(255),tc].floor()+3
        d=torch.sqrt((xx-tc[:,None,None]).square()+(yy-tr[:,None,None]).square()+(h+1-z[:,None,None]).square()).clamp_min(.25)
        fs=20*torch.log10(d[:,None]*torch.tensor([415e6,5.775e9],device='cuda')[None,:,None,None]*(4*np.pi/299792458))
        return base,local,graph,fs,vis,y,w
def sse(e,w,v):
    return torch.stack([torch.stack([(e.square()*m).sum((2,3),dtype=torch.float64),m.sum((2,3))],2)
                        for m in (w,w&(v>.99),w&(v<=.99))],2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['simple','scales','full'],required=True);args=ap.parse_args()
    g=gpu_guard();torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    D=Data();hist={};co=[];ts=time.monotonic();out=A/('blend_'+args.mode);out.mkdir(exist_ok=True)
    fit=np.flatnonzero(np.array(D.groups)=='fit')
    for b in (0,1):
        gram=right=None;ct=0
        for start in range(0,len(fit),8):
            j=fit[start:start+8];base,local,graph,fs,v,y,w=D.batch(j);px=torch.tensor(D.px[j].astype(np.int64),device='cuda')
            p=(local+graph)*.5;X=field_features(base,local,graph,fs,v,args.mode,b)
            xx=X.flatten(2).gather(2,px[:,None].expand(-1,X.shape[1],-1)).transpose(1,2)
            yy=(y[:,b]-p[:,b]).flatten(1).gather(1,px);ww=w[:,b].flatten(1).gather(1,px)
            z=xx[ww].double();t=yy[ww].double()
            if gram is None:gram=z.T@z;right=z.T@t
            else:gram+=z.T@z;right+=z.T@t
            ct+=len(z)
        lam=.008*gram.diagonal().clamp_min(.0002*ct);lam[0]=.1
        co.append(torch.linalg.solve(gram+torch.diag(lam),right).float())
    atomic_json(out/'coeff.json',dict(mode=args.mode,beta=[x.cpu().tolist() for x in co],fit_maps=len(fit),GPU=g))
    for group in ('select','confirm'):
        sel=np.flatnonzero(np.array(D.groups)==group);pool=[]
        for start in range(0,len(sel),8):
            j=sel[start:start+8];base,aa,bb,fs,v,y,w=D.batch(j);p=(aa+bb)*.5
            corr=torch.cat([(field_features(base,aa,bb,fs,v,args.mode,b)*co[b][None,:,None,None]).sum(1,keepdim=True) for b in (0,1)],1).clamp(-15,15)
            eo=torch.maximum(torch.minimum(aa,bb)-y,y-torch.maximum(aa,bb)).clamp_min(0)
            pool.append(torch.stack([sse(e,w,v) for e in (p-y,p+corr-y,eo,aa-y,bb-y)],1).cpu().numpy())
        arr=np.concatenate(pool);np.savez(out/(group+'.npz'),sse=arr,terrain=[D.rows[D.ix[j]]['terrain_id'] for j in sel],indices=sel)
        a=arr.sum(0);s=np.sqrt(a[...,0]/a[...,1]);r=dict(maps=len(sel),rmse=s[1].tolist(),gain=(s[0]-s[1]).tolist(),label_assisted_convex_oracle_gain=(s[0]-s[2]).tolist())
        hist[group]=r;print(group,'maps',len(sel),'gain',np.round(s[0]-s[1],5)[:,:2].tolist(),'oracle',np.round(s[0]-s[2],4)[:,0].tolist(),'seconds',round(time.monotonic()-ts,1),flush=True)
    atomic_json(out/'result.json',hist)
if __name__=='__main__':main()
