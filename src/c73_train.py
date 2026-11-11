"""Train matched end-to-end graph models from raw terrain/TX features and official TRAIN labels."""
import argparse,csv,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import DATA,atomic_json,gpu_guard
from src.c3_geometry import basic_inputs
from src.c73_graph import GraphField,Visibility
from src.c67_dense import frozen_maps
import numpy as np
import torch
BANDS=['415','58']
def mmap(p):return np.load(p,mmap_mode='r',allow_pickle=False)
def gpu(v):return torch.from_numpy(np.ascontiguousarray(v)).to('cuda')
class TerrainData:
    def __init__(self,ids,folders,resident=True):
        self.ids=list(ids);n=len(ids)
        rows=list(csv.DictReader((DATA/'train_index.csv').open()))
        hh=mmap(DATA/'train_hm.npy');unique,inv=np.unique([int(rows[i]['hm_row']) for i in ids],return_inverse=True)
        self.heights=torch.empty((len(unique),256,256),device='cuda',dtype=torch.float32)
        for j in range(0,len(unique),128):self.heights[j:j+128].copy_(torch.from_numpy(hh[unique[j:j+128]]))
        self.hidx=gpu(inv);self.tr=gpu(np.array([int(rows[i]['tx_row']) for i in ids]));self.tc=gpu(np.array([int(rows[i]['tx_col']) for i in ids]))
        self.terrains=[rows[i]['terrain_id'] for i in ids]
        where='cuda' if resident else 'cpu';self.x=torch.empty((n,45,256,256),device=where,dtype=torch.float16)
        self.y=torch.empty((n,2,256,256),device=where,dtype=torch.float16);self.w=torch.empty((n,2,256,256),device=where,dtype=torch.bool)
        lookup={}
        for folder in folders:
            for d in (ROOT/folder).glob('part*'):
                if not (d/'complete.json').is_file():continue
                feat=mmap(d/'features.npy');lookup.update({i:(feat,pos) for pos,i in enumerate(json.load(open(d/'indices.json')))})
        ys=[mmap(DATA/f'train_pl{b}.npy') for b in BANDS];ws=[mmap(DATA/f'train_mask{b}.npy') for b in BANDS]
        for j in range(0,n,24):
            ix=self.ids[j:j+24];sl=slice(j,j+len(ix))
            data,first=lookup[ix[0]]
            if all(lookup[i][0] is data and lookup[i][1]==first+k for k,i in enumerate(ix)):
                vv=np.array(data[first:first+len(ix)],copy=True)
            else:vv=np.stack([lookup[i][0][lookup[i][1]] for i in ix])
            self.x[sl].copy_(torch.from_numpy(vv));self.y[sl].copy_(torch.from_numpy(np.stack([m[ix] for m in ys],axis=1)))
            self.w[sl].copy_(torch.from_numpy(np.stack([np.unpackbits(m[ix],axis=1).reshape(-1,256,256) for m in ws],axis=1)).bool())
    def batch(self,ix):
        local=ix.to(self.x.device);h=self.heights[self.hidx[ix]];tr=self.tr[ix];tc=self.tc[ix]
        v=self.x[local].to('cuda',dtype=torch.float32);y=self.y[local].to('cuda',dtype=torch.float32);w=self.w[local].to('cuda')
        x,fs=basic_inputs(h,tr,tc)
        # Native dB and power interpolation are two observable candidate summaries. Clip only the initial prior.
        prior=fs+(-20*(v[:,31:33]+v[:,41:43])).clamp(-8,48)
        out=torch.cat((x,v),1).contiguous(memory_format=torch.channels_last)
        return out,h,tr,tc,prior,y,w

@torch.inference_mode()
def evaluate(net,panel,reference,batch,return_maps=False):
    allrows=[];maps=[]
    for start in range(0,len(panel.ids),batch):
        ix=torch.arange(start,min(len(panel.ids),start+batch),device='cuda')
        x,h,tr,tc,p0,y,w=panel.batch(ix)
        if net is not None:
            with torch.autocast('cuda',torch.bfloat16):p=p0+net(x,h,tr,tc).float()
        else:p=p0
        base=reference[start:start+len(ix)].to('cuda')
        se=lambda e:(e.square()*w).sum((2,3),dtype=torch.float64)
        allrows.append(torch.cat((se(p-y),se(.5*(base+p)-y),se(base-y),w.sum((2,3))),1).cpu().numpy())
        if return_maps:maps.append(p.cpu().numpy())
    rows=np.concatenate(allrows);a=rows.sum(0);summary=np.sqrt(a[:6].reshape(3,2)/a[6:]).tolist()
    return dict(standalone=summary[0],fixed_half_with_baseline=summary[1],baseline=summary[2]),rows,maps

def main():
    p=argparse.ArgumentParser();p.add_argument('--shuffle',action='store_true');p.add_argument('--name',required=True)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch',type=int,default=8);p.add_argument('--width',type=int,default=40)
    p.add_argument('--check',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();g=gpu_guard();torch.set_num_threads(6);torch.manual_seed(73119);np.random.seed(73119)
    torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('high')
    if a.check:
        h=torch.zeros((1,256,256),device='cuda');r=torch.tensor([100],device='cuda');c=torch.tensor([151],device='cuda')
        ids,v=Visibility()(h,r,c);assert (ids[:,0,:,:24]==-1).all() and (ids[:,1,:,:8]>=0).float().mean()>.96
        assert ((ids<4096)&(ids>=-1)).all() and torch.isfinite(v).all()
        print('flat geometric visibility passed',g['uuid'],flush=True);return
    out=ROOT/'runs'/a.name;out.mkdir(exist_ok=a.resume)
    plan=json.load(open(ROOT/'runs/c69_registered_spatial/protocol.json'));fit=json.load(open(ROOT/'audit/c72_continuation/wide_fit.json'))
    mon=plan['monitor'];con=plan['confirmation'];t=time.monotonic()
    resident=torch.cuda.mem_get_info()[0]/2**30>67
    if not a.resume:atomic_json(out/'protocol.json',dict(vars(a),gpu=g,fit=fit,monitor=mon,confirmation=con,teacher_input=False,resident=resident,
                    selection='minimum mean of the two standalone internal-monitor band RMSEs. Fixed 50/50 blend is a separate diagnostic.'))
    print('loading',len(fit),'FIT maps',len(mon),'monitor','resident',resident,flush=True)
    train=TerrainData(fit,['features/c73_compact'],resident);dev=TerrainData(mon,['features/c73_compact'])
    reference=torch.from_numpy(frozen_maps(mon,'dev'))
    net=GraphField(56,a.width,a.shuffle).to('cuda',memory_format=torch.channels_last)
    opt=torch.optim.AdamW(net.parameters(),lr=.0012,weight_decay=.012,fused=True)
    gen=torch.Generator().manual_seed(73731);best=1e9;history=[];begin=0
    if a.resume:
        ck=torch.load(out/'last.pt',weights_only=True,map_location='cpu');net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);gen.set_state(ck['order'])
        begin=ck['epoch'];history=json.load(open(out/'history.json'))[:begin];best=min(x['mean'] for x in history)
    else:
        init,_,_=evaluate(None,dev,reference,a.batch);atomic_json(out/'initial.json',init);print('prior',init,flush=True)
    n=len(fit);steps=math.ceil(n/a.batch);model_params=sum(p.numel() for p in net.parameters())
    print('loaded seconds',round(time.monotonic()-t,1),'parameters',model_params,flush=True)
    for ep in range(begin,a.epochs):
        net.train();total=torch.zeros(4,device='cuda',dtype=torch.float64)
        for j,ix in enumerate(torch.randperm(n,generator=gen).to('cuda').split(a.batch)):
            tt=ep+(j+1)/steps;lr=.0012*min(1.,.12+tt/.45)*(.12+.88*(.5+.5*math.cos(math.pi*tt/a.epochs)))
            for group in opt.param_groups:group['lr']=lr
            x,h,tr,tc,p0,y,w=train.batch(ix)
            with torch.autocast('cuda',torch.bfloat16):res=net(x,h,tr,tc)
            err=torch.where(w,(p0+res.float()-y),0.);se=err.square().sum((0,2,3));cnt=w.sum((0,2,3))
            loss=(se/cnt.clamp_min(1)).mean()/25
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            total+=torch.cat((se.detach(),cnt))
            if (j+1)%200==0:
                atomic_json(out/'progress.json',dict(epoch=ep+1,step=j+1,total_steps=steps,seconds=round(time.monotonic()-t,1),running_fit=(total[:2]/total[2:]).sqrt().tolist()))
        net.eval();m,per,_=evaluate(net,dev,reference,a.batch);mean=np.mean(m['standalone']).item()
        rec=dict(epoch=ep+1,fit=(total[:2]/total[2:]).sqrt().tolist(),monitor=m,mean=mean,seconds=round(time.monotonic()-t,1));history.append(rec)
        atomic_json(out/'history.json',history);print(json.dumps(rec),flush=True)
        if mean<best:
            best=mean;torch.save(dict(state=net.state_dict(),width=a.width,shuffle=a.shuffle,epoch=ep+1),out/'best.pt.tmp');os.replace(out/'best.pt.tmp',out/'best.pt')
            np.save(out/'monitor_best_by_map.npy',per)
        torch.save(dict(state=net.state_dict(),optimizer=opt.state_dict(),order=gen.get_state(),epoch=ep+1),out/'last.pt.tmp');os.replace(out/'last.pt.tmp',out/'last.pt')
    ck=torch.load(out/'best.pt',weights_only=True,map_location='cpu');net.load_state_dict(ck['state']);net.eval()
    del train,dev
    confirm=TerrainData(con,['features/c73_compact']);reference=torch.from_numpy(frozen_maps(con,'dev'))
    cm,per,_=evaluate(net,confirm,reference,a.batch);np.save(out/'confirmation_by_map.npy',per)
    joint=np.vstack((np.load(out/'monitor_best_by_map.npy'),per)).sum(0);joint=np.sqrt(joint[:6].reshape(3,2)/joint[6:])
    report=dict(best_epoch=ck['epoch'],monitor=history[ck['epoch']-1]['monitor'],confirmation=cm,combined128=joint.tolist(),order=['standalone','fixed_half','C54'],
        seconds=round(time.monotonic()-t,1),official_val=False)
    atomic_json(out/'result.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
