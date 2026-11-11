"""Estimate a fixed, position-specific receiver response across independent TRAIN terrains."""
from pathlib import Path
import argparse,sys,time,csv,json
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
A=R/'audit/c88_mesh'
def smooth(x,k):
    return F.avg_pool2d(x,k,1,k//2,count_include_pad=False)
def features(p,h,r,c,b):
    yy,xx=torch.meshgrid(torch.arange(256,device='cuda'),torch.arange(256,device='cuda'),indexing='ij')
    dx=xx-c[:,None,None];dy=yy-r[:,None,None]
    z=h[torch.arange(len(h),device='cuda'),(r+1).clamp_max(255),c].floor()+3
    dz=h+1-z[:,None,None];ds=(dx.square()+dy.square()+dz.square()).clamp_min(.0625).sqrt()
    fs=20*torch.log10(ds*(4*np.pi*(415e6 if b==0 else 5.775e9)/299792458))
    pl=p[:,b:b+1];ex=(pl-fs[:,None])/8
    common=(p.mean(1,keepdim=True)-smooth(p.mean(1,keepdim=True),3))/4
    high=((p[:,1:2]-p[:,:1])-smooth(p[:,1:2]-p[:,:1],3))/8
    q=F.pad(pl,(1,1,1,1),mode='replicate')
    stencil=[(q[:,:,1+j:257+j,1+i:257+i]-pl)/4 for j,i in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]]
    h1=F.pad(h[:,None],(1,1,1,1),mode='replicate')[:,0]
    sxp=h1[:,1:257,2:258]-h;sxm=h-h1[:,1:257,:256]
    syp=h1[:,2:258,1:257]-h;sym=h-h1[:,:256,1:257]
    grazing=[]
    for gx,gy in [(sxp,syp),(sxp,sym),(sxm,syp),(sxm,sym)]:
        cos=((dz-dx*gx-dy*gy)/(ds*(1+gx.square()+gy.square()).sqrt())).abs()
        grazing.append((1/(1+(cos/.015).square()))[:,None])
    one=torch.ones_like(ex);soft=((ex-.35)*2).clamp(0,1)
    return torch.cat([one,ex.clamp(-3,9),soft,common,high,*stencil,*[v*soft for v in stencil[:4]],*grazing,*[v*soft for v in grazing]],1).clamp(-8,8)
class Fields:
    def __init__(self):
        info=json.load(open(A/'base_index.json'));self.ids=info['ids'];self.groups=np.array(info['groups'])
        self.rows=list(csv.DictReader(open(DATA/'train_index.csv')))
        self.h=np.load(DATA/'train_hm.npy',mmap_mode='r');self.p=np.load(A/'base.npy',mmap_mode='r')
        self.y=[np.load(DATA/f'train_pl{k}.npy',mmap_mode='r') for k in ('415','58')]
        self.m=[np.load(DATA/f'train_mask{k}.npy',mmap_mode='r') for k in ('415','58')]
    def batch(self,j,b):
        ids=[self.ids[k] for k in j];rows=[self.rows[i] for i in ids]
        to=lambda a:torch.from_numpy(np.ascontiguousarray(a)).cuda()
        p=to(self.p[j]);h=to(self.h[[int(v['hm_row']) for v in rows]])
        r=to(np.array([int(v['tx_row']) for v in rows]));c=to(np.array([int(v['tx_col']) for v in rows]))
        y=to(self.y[b][ids]).float();w=to(np.unpackbits(self.m[b][ids],axis=1).reshape(-1,256,256)).bool()
        return features(p,h,r,c,b),y-p[:,b],w
def solve(gram,rhs,alpha):
    n,f,_=gram.shape
    out=torch.empty((n,f),device='cuda')
    for j in range(0,n,2048):
        G=gram[j:j+2048];v=rhs[j:j+2048]
        eye=(G.diagonal(dim1=-2,dim2=-1).clamp_min(1)+1)*alpha+1.5
        out[j:j+2048]=torch.linalg.solve(G+torch.diag_embed(eye),v.unsqueeze(-1)).squeeze(-1)
    return out
def permute(w):
    # Permute cells only inside 16x16 blocks, preserving large-scale domain trends.
    ids=np.arange(65536).reshape(16,16,16,16).transpose(0,2,1,3).reshape(256,256)
    rng=np.random.default_rng(88238);perm=np.empty(65536,np.int64)
    for row in ids:perm[row]=rng.permutation(row)
    return w[:,torch.from_numpy(perm).cuda()]
def operators(gram,rhs,alpha):
    n,f,_=gram.shape
    tied=solve(gram.mean(0,keepdim=True),rhs.mean(0,keepdim=True),.008)[0]
    # Empirical-Bayes shrinkage toward one shared observation operator.
    center=rhs-gram@tied
    diag=solve(gram,center,alpha).T
    beta=tied[:,None]+diag
    rolled=tied[:,None]+permute(diag)
    return tied[:,None].expand(-1,n),beta,rolled
def score(D,b,group,weights):
    sel=np.flatnonzero(D.groups==group);res=[]
    for st in range(0,len(sel),6):
        j=sel[st:st+6];x,y,w=D.batch(j,b);x=x.flatten(2)
        corr=torch.einsum('bfn,kfn->kbn',x,weights).reshape(len(weights),len(j),256,256).clamp(-15,15)
        ss=((corr-y).square()*w).sum((2,3),dtype=torch.float64)
        res.append(torch.cat([(y.square()*w).sum((1,2),dtype=torch.float64)[None],ss,w.sum((1,2))[None]],0).T.cpu().numpy())
    arr=np.concatenate(res);return arr, np.sqrt(arr[:,:-1].sum(0)/arr[:,-1].sum())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--band',type=int,required=True);a=ap.parse_args();gpu=gpu_guard()
    b=a.band;D=Fields();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    ts=time.monotonic();fit=np.flatnonzero(D.groups=='fit');gram=rhs=None
    for st in range(0,len(fit),6):
        x,y,w=D.batch(fit[st:st+6],b);v=x.permute(2,3,0,1).reshape(65536,len(x),-1).contiguous()
        wx=v*w.permute(1,2,0).reshape(65536,len(x),1)
        gg=wx.transpose(1,2)@v;rr=(wx*y.permute(1,2,0).reshape(65536,len(x),1)).sum(1)
        if gram is None:gram=gg;rhs=rr
        else:gram+=gg;rhs+=rr
    names=['global','reg0.13','perm0.13','reg0.6','perm0.6','reg2','perm2'];ops=[]
    for a0 in (.13,.6,2.):
        tied,beta,rand=operators(gram,rhs,a0)
        if not ops:ops.append(tied)
        ops.extend([beta,rand])
    del gram,rhs
    op=torch.stack(ops);report={'band':b,'fit_maps':len(fit),'names':names,'gpu':gpu}
    arr,scores=score(D,b,'select',op);report['selection_scores']=scores.tolist()
    # Freeze variant using 48 selection terrains. Confirmation scores do not choose shrinkage.
    wi=min([1,3,5],key=lambda i:scores[i+1])
    torch.save(op[wi].cpu(),A/f'registered_{b}.pt');report['selected']=names[wi]
    atomic_json(A/f'registered_choice_{b}.json',report)
    arr,ss=score(D,b,'confirm',op[[0,wi,wi+1]])
    inds=np.flatnonzero(D.groups=='confirm')
    np.savez(A/f'registered_confirm_{b}.npz',sse_count=arr,terrain=[D.rows[D.ids[j]]['terrain_id'] for j in inds])
    report.update(confirm_map_count=len(arr),confirm_names=['C82','global',names[wi],names[wi+1]],confirm_rmse=ss.tolist(),
                  confirmed_gain=float(ss[0]-ss[2]),registration_specific_gain=float(ss[3]-ss[2]),seconds=time.monotonic()-ts)
    atomic_json(A/f'registered_{b}.json',report);print(report,flush=True)
if __name__=='__main__':main()
