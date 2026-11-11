"""Matched weak-state supervision trial, using official TRAIN only."""
import argparse,csv,json,math,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c81_weakfield import Inputs,WeakField
A=R/'audit/c79_receiver'
def mmap(p):return np.load(p,mmap_mode='r')
class Reader:
    def __init__(self):
        d=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1'
        self.ix=json.load(open(d/'indices.json'))['index_rows']
        self.base=[mmap(d/f'pl{b}.npy') for b in ('415','58')]
        self.local=mmap(R/'runs/c72_observed/full_internal/correction.npy')
        self.graph=mmap(R/'runs/c73_wide_real/full_internal/prediction.npy')
        self.hh=mmap(R.parent/'data/train_hm.npy')
        self.targets=[mmap(R.parent/f'data/train_pl{b}.npy') for b in ('415','58')]
        self.masks=[mmap(R.parent/f'data/train_mask{b}.npy') for b in ('415','58')]
        self.rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));self.engine=Inputs()
        self.group={r['terrain']:r['group'] for r in json.load(open(A/'protocol.json'))['panel']}
        self.comp={}
        for folder in ('c73_compact','c73_dev_extra_compact'):
            for part in (R/'features'/folder).glob('part*'):
                x=mmap(part/'features.npy')
                for j,i in enumerate(json.load(open(part/'indices.json'))):self.comp[i]=(x,j)
    def select(self,kind):return [j for j,i in enumerate(self.ix) if self.group[self.rows[i]['terrain_id']]==kind]
    def read(self,js):
        ids=[self.ix[j] for j in js];rows=[self.rows[i] for i in ids]
        gpu=lambda z:torch.from_numpy(np.ascontiguousarray(z)).cuda()
        hh=gpu(self.hh[[int(r['hm_row']) for r in rows]]).float()
        tr=gpu(np.array([int(r['tx_row']) for r in rows]));tc=gpu(np.array([int(r['tx_col']) for r in rows]))
        compact=gpu(np.stack([self.comp[i][0][self.comp[i][1]] for i in ids])).float()
        pred=.5*(gpu(np.stack([v[js] for v in self.base],1)).float()+gpu(self.local[js]).float()+gpu(self.graph[js]).float())
        x,fs=self.engine(hh,tr,tc,pred,compact)
        y=gpu(np.stack([v[ids] for v in self.targets],1)).float()
        w=gpu(np.stack([np.unpackbits(v[ids],axis=1).reshape(-1,256,256) for v in self.masks],1)).bool()
        clear=compact[:,16:17]>.99
        return x.half(),pred,fs,y,w,clear

class Maps:
    def __init__(self,data,js):
        self.js=list(js);self.tensors=[]
        for k in range(0,len(js),8):
            t=data.read(js[k:k+8])
            if not self.tensors:
                self.tensors=[torch.empty((len(js),*v.shape[1:]),device='cuda',dtype=v.dtype) for v in t]
            for dst,src in zip(self.tensors,t):dst[k:k+len(src)].copy_(src)
    def batch(self,ix):
        x,p,f,y,w,c=[v[ix] for v in self.tensors]
        return x.float().contiguous(memory_format=torch.channels_last),p,f,y,w,c
def metrics(net,panel,batch):
    sums=[]
    net.eval()
    with torch.inference_mode():
        for j in range(0,len(panel.js),batch):
            x,p,f,y,w,c=panel.batch(slice(j,j+batch))
            with torch.autocast('cuda',torch.bfloat16):v,logit,_,_=net(x,p,f)
            weak=y>f+3;flag=logit>0;zones=[w,w&c,w&~c]
            vals=[]
            for mask in zones:
                vals.append(torch.stack([((pred-y).square()*mask).sum((2,3),dtype=torch.float64) for pred in (p,v)]+
                         [mask.sum((2,3)),(mask&weak).sum((2,3)),(mask&weak&flag).sum((2,3)),(mask&flag).sum((2,3))],2))
            sums.append(torch.stack(vals,2).cpu().numpy())
    return np.concatenate(sums)
def get_score(rows):
    a=rows.sum(0);return np.sqrt(a[:,:,1]/a[:,:,2])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['plain','state'],required=True)
    ap.add_argument('--epochs',default=32,type=int);ap.add_argument('--batch',default=8,type=int);ap.add_argument('--confirm',action='store_true')
    a=ap.parse_args();gpu=gpu_guard();torch.manual_seed(81032);np.random.seed(81032);torch.set_num_threads(5)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
    rd=Reader();dest=R/'runs'/('c81_'+a.mode);dest.mkdir(exist_ok=True);t=time.monotonic()
    if a.confirm:
        ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=True)
        net=WeakField(ck['channels']).to('cuda',memory_format=torch.channels_last);net.load_state_dict(ck['state'])
        js=rd.select('confirm');res=[]
        for i in range(0,len(js),96):
            pp=Maps(rd,js[i:i+96]);res.append(metrics(net,pp,a.batch));del pp
            if i%384==0:print('confirmation',i+len(res[-1]),len(js),flush=True)
        res=np.concatenate(res);np.savez(dest/'confirm.npz',sse=res,indices=[rd.ix[j] for j in js],terrain=[rd.rows[rd.ix[j]]['terrain_id'] for j in js])
        b=np.sqrt(res.sum(0)[:,:,0]/res.sum(0)[:,:,2]);s=get_score(res)
        atomic_json(dest/'confirm.json',dict(terrains=484,maps=len(js),zones=['all','clear','other'],base=b.tolist(),rmse=s.tolist(),gain=(b-s).tolist(),best_epoch=ck['epoch']))
        print('confirmation gain',np.round(b-s,5).tolist(),flush=True);return
    tr=Maps(rd,rd.select('fit'));val=Maps(rd,rd.select('select'));channels=tr.tensors[0].shape[1]
    net=WeakField(channels).to('cuda',memory_format=torch.channels_last)
    h=hashlib.sha256()
    for wt in net.state_dict().values():h.update(wt.cpu().numpy().tobytes())
    atomic_json(dest/'config.json',dict(gpu=gpu,mode=a.mode,epochs=a.epochs,fit_maps=len(tr.js),select_maps=len(val.js),channels=channels,init=h.hexdigest(),seed=81032,features='H,TX,C76,compact_h_tx only',protocol=str(A/'c81_protocol.json')))
    print('loaded',len(tr.js),len(val.js),channels,'seconds',round(time.monotonic()-t,1),flush=True)
    opt=torch.optim.AdamW(net.parameters(),lr=.0013,weight_decay=.008,fused=True)
    gen=torch.Generator().manual_seed(81092);history=[];best=100
    for ep in range(1,a.epochs+1):
        net.train();order=torch.randperm(len(tr.js),generator=gen,device='cpu');lr=.0013*(.07+.93*.5*(1+math.cos(math.pi*(ep-1)/a.epochs)))
        for g in opt.param_groups:g['lr']=lr
        for start in range(0,len(order),a.batch):
            ids=order[start:start+a.batch].cuda();x,p,f,y,w,c=tr.batch(ids)
            with torch.autocast('cuda',torch.bfloat16):v,logits,strong,weak=net(x,p,f)
            ms=lambda z:((z*w).sum((0,2,3))/w.sum((0,2,3)).clamp_min(1)).mean()
            loss=ms((v-y).square())/16.
            if a.mode=='state':
                flag=y>f+3
                lc=F.binary_cross_entropy_with_logits(logits,flag.float(),reduction='none')
                la=(torch.where(flag,weak,strong)-y).square()/16.
                loss=loss+.55*ms(lc)+.15*ms(la)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
        z=metrics(net,val,a.batch);score=get_score(z);base=np.sqrt(z.sum(0)[:,:,0]/z.sum(0)[:,:,2]);criterion=float(score[:,0].mean())
        record=dict(epoch=ep,rmse=score.tolist(),gain=(base-score).tolist(),seconds=round(time.monotonic()-t,1))
        history.append(record);atomic_json(dest/'history.json',history)
        if criterion<best:
            best=criterion;torch.save(dict(state=net.state_dict(),epoch=ep,channels=channels),dest/'best.tmp')
            (dest/'best.tmp').replace(dest/'best.pt');np.save(dest/'select_sse.npy',z)
        print(ep,'gain',np.round(base-score,5)[:,:2].tolist(),'seconds',record['seconds'],flush=True)
    atomic_json(dest/'complete.json',dict(fit_maps=len(tr.js),best_score=best,epochs=a.epochs,seconds=time.monotonic()-t))
if __name__=='__main__':main()
