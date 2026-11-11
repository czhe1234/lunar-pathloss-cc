"""Dense paired correction from the two frozen full prediction maps and terrain descriptors."""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard,atomic_json,DATA
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from src.c3_geometry import basic_inputs

BANDS=('415','58')
def mmap(path):
    return np.load(path,mmap_mode='r',allow_pickle=False)

def frozen_maps(indices,split):
    cache={}
    if split=='fit':
        for d in sorted([*(ROOT/'features/c66_frozen_maps').glob('fit*of2'),*(ROOT/'features/c75_fit_extension').glob('fit*')]):
            if not (d/'complete.json').is_file():continue
            assert json.loads((d/'complete.json').read_text())['complete']
            pr=mmap(d/'prediction.npy')
            for j,i in enumerate(json.loads((d/'indices.json').read_text())):
                cache[i]=(pr,j)
        return np.stack([cache[i][0][cache[i][1]] for i in indices])
    d=ROOT/'runs/c54_combo_dev/bf16_compiled/all_shard0of1'
    order={i:j for j,i in enumerate(json.loads((d/'indices.json').read_text())['index_rows'])}
    positions=[order[i] for i in indices]
    return np.stack([mmap(d/f'pl{b}.npy')[positions] for b in BANDS],axis=1)

def gpu_array(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to('cuda')

def model_inputs(h,tr,tc,pred,g3,g4,physics=None):
    basic,fs=basic_inputs(h,tr,tc)
    frac=h[torch.arange(len(h),device=h.device),tr,tc].remainder(1)[:,None,None,None]
    gap=2.5+2*torch.sinh(3*g4[:,:1].float())-(3-frac)
    extra=[(pred-fs)/20,(pred[:,1:2]-pred[:,:1]-(fs[:,1:2]-fs[:,:1]))/12,
           torch.asinh(gap)/3,frac.expand(-1,1,256,256),h[:,None].remainder(1)-.5]
    if physics is not None:extra.append(physics.float())
    return torch.cat([basic,g3.float(),g4.float()]+extra,1).clamp(-10,10)

class Panel:
    def __init__(self,indices,split,physics_dir='',host=False):
        storage='cpu' if host else 'cuda'
        self.indices=list(indices)
        rows=list(csv.DictReader((DATA/'train_index.csv').open()))
        g3=mmap(ROOT/'features/c3/train.npy');g4=mmap(ROOT/'features/c4_extra/train.npy')
        heights=mmap(DATA/'train_hm.npy')
        pl=[mmap(DATA/f'train_pl{b}.npy') for b in BANDS]
        mask=[mmap(DATA/f'train_mask{b}.npy') for b in BANDS]
        pmap=frozen_maps(indices,split)
        exclude={}
        if split=='fit':
            ii=mmap(ROOT/'features/c38_dense/fit_map_indices.npy')
            pix=mmap(ROOT/'features/c38_dense/fit_pixel.npy').reshape(len(ii),-1)
            exclude={int(i):pos for pos,i in enumerate(ii)}
        sources=[]
        for feature_dir in filter(None,physics_dir.split(',')):
            lookup={}
            for d in sorted(d for part in feature_dir.split('+') for d in (ROOT/part).glob('*')):
                if not (d/'complete.json').is_file():continue
                pr=mmap(d/'features.npy')
                for j,i in enumerate(json.loads((d/'indices.json').read_text())):lookup[i]=(pr,j)
            assert all(i in lookup for i in indices)
            sources.append(lookup)
        n=len(indices)
        self.y=torch.empty((n,2,256,256),dtype=torch.float32,device=storage)
        self.w=torch.empty((n,2,256,256),dtype=torch.bool,device=storage)
        for start in range(0,n,12):
            ix=list(indices[start:start+12]); sl=slice(start,start+len(ix))
            tr=gpu_array(np.array([int(rows[i]['tx_row']) for i in ix]))
            tc=gpu_array(np.array([int(rows[i]['tx_col']) for i in ix]))
            h=gpu_array(heights[[int(rows[i]['hm_row']) for i in ix]]).float()
            pred=gpu_array(pmap[sl]).float()
            phys=torch.cat([gpu_array(np.stack([lk[i][0][lk[i][1]] for i in ix])) for lk in sources],1) if sources else None
            x=model_inputs(h,tr,tc,pred,gpu_array(g3[ix]),gpu_array(g4[ix]),phys)
            if start==0:
                self.x=torch.empty((n,*x.shape[1:]),dtype=torch.float16,device=storage,memory_format=torch.channels_last)
            self.x[sl].copy_(x)
            valid=np.stack([np.unpackbits(m[ix],axis=1).reshape(-1,256,256) for m in mask],axis=1)
            if exclude:
                for j,i in enumerate(ix):valid[j].reshape(2,-1)[:,pix[exclude[i]]]=0
            valid_gpu=gpu_array(valid).bool()
            self.w[sl].copy_(valid_gpu)
            target=gpu_array(np.stack([m[ix] for m in pl],axis=1)).float()
            self.y[sl].copy_(torch.where(valid_gpu,target-pred,0.))
        self.terrain=[rows[i]['terrain_id'] for i in indices]
        for x,y in zip(self.x.split(12),self.y.split(12)):
            assert torch.isfinite(y).all() and torch.isfinite(x).all()

class Block(nn.Module):
    def __init__(self,width,rate,spatial):
        super().__init__()
        self.rate=rate
        self.spatial=spatial
        self.filter=nn.Conv2d(width,width,3,dilation=rate,padding=rate,groups=width,bias=False)
        self.norm=nn.LayerNorm(width)
        self.up=nn.Linear(width,2*width)
        self.down=nn.Linear(2*width,width)
        self.scale=nn.Parameter(torch.full((width,),.1))
    def forward(self,x):
        if self.spatial:
            v=self.filter(x)
        else:
            v=x*(self.filter.weight.sum((1,2,3))/3)[None,:,None,None]
        v=self.norm(v.permute(0,2,3,1))
        v=self.down(F.gelu(self.up(v)))*self.scale
        return x+v.permute(0,3,1,2)

class ResidualMap(nn.Module):
    def __init__(self,channels,width,spatial):
        super().__init__()
        self.stem=nn.Conv2d(channels,width,1)
        self.blocks=nn.Sequential(*(Block(width,r,spatial) for r in (1,2,4,2,1)))
        self.head=nn.Conv2d(width,2,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        return 4*self.head(self.blocks(self.stem(x)))

@torch.no_grad()
def score(net,panel,batch):
    rows=[]
    for lo in range(0,len(panel.indices),batch):
        sl=slice(lo,lo+batch);e=panel.y[sl].to('cuda');w=panel.w[sl].to('cuda')
        if net is not None:
            with torch.autocast('cuda',torch.bfloat16): p=net(panel.x[sl].to(device='cuda',dtype=torch.float32))
            e=e-p.float()
        rows.append(torch.cat(((e.square()*w).sum((2,3),dtype=torch.float64),w.sum((2,3))),dim=1).cpu().numpy())
    rows=np.concatenate(rows)
    return np.sqrt(rows[:,:2].sum(0)/rows[:,2:].sum(0)).tolist(), rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('center','spatial'),required=True)
    p.add_argument('--name',required=True);p.add_argument('--physics-dir',default='')
    p.add_argument('--width',type=int,default=40);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--epochs',type=int,default=18);p.add_argument('--fit-limit',type=int,default=0)
    p.add_argument('--host-panel',action='store_true');p.add_argument('--network',choices=['local','field','native','observed'],default='local')
    p.add_argument('--restart-from',default='',help='Run folder of identical training protocol. Prefer last optimizer state; legacy runs restore best weights.')
    p.add_argument('--fit-indices',default='');p.add_argument('--initialize-from',default='');p.add_argument('--initial-lr',type=float,default=.0012)
    args=p.parse_args()
    gpu=gpu_guard();torch.manual_seed(6767);np.random.seed(6767);torch.set_num_threads(5)
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('high')
    panels=json.loads((ROOT/'audit/c66_transport/panels.json').read_text())
    devrows=list(csv.DictReader((DATA/'train_index.csv').open()))
    # Freeze monitoring/confirmation independently of labels, by complete terrain.
    rng=np.random.default_rng(998671)
    mterr=set(rng.choice(panels['dev_terrains'],64,replace=False).tolist())
    mon=[i for i in panels['dev_maps'] if devrows[i]['terrain_id'] in mterr]
    con=[i for i in panels['dev_maps'] if devrows[i]['terrain_id'] not in mterr]
    fit=json.load(open(ROOT/args.fit_indices)) if args.fit_indices else panels['fit_maps']
    if args.fit_indices:
        expected=set(map(int,mmap(ROOT/'features/c38_dense/fit_map_indices.npy')))
        assert set(fit)<=expected and len(set(fit))==len(fit) and not set(fit)&set(mon+con)
    if args.fit_limit: fit=np.random.default_rng(415).choice(fit,min(args.fit_limit,len(fit)),replace=False).tolist()
    out=ROOT/'runs'/args.name;out.mkdir(exist_ok=False)
    atomic_json(out/'protocol.json',dict(vars(args),gpu=gpu,fit=fit,monitor=mon,confirmation=con,
        selection='minimum mean of two monitor RMSEs; confirmation inspected after freezing; training loss excludes fixed C38 FIT teaching pixels'))
    st=time.monotonic();train=Panel(fit,'fit',args.physics_dir,args.host_panel);dev=Panel(mon,'dev',args.physics_dir,args.host_panel)
    base,base_rows=score(None,dev,args.batch)
    print('data',len(fit),len(mon),'channels',train.x.shape[1], 'baseline',base,'elapsed',time.monotonic()-st,flush=True)
    from src.c70_field import FieldMap
    if args.network in ('native','observed'):
        from src.c72_native import NativeMap
        net=NativeMap(train.x.shape[1],args.width,args.network=='native')
    elif args.network=='field':net=FieldMap(train.x.shape[1],args.width)
    else:net=ResidualMap(train.x.shape[1],args.width,args.mode=='spatial')
    model=net.to('cuda',memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=args.initial_lr,weight_decay=.008,fused=True)
    n=len(fit);b=args.batch;best=1e9;history=[]
    gen=torch.Generator(device='cpu').manual_seed(171717)
    start_epoch=0
    if args.initialize_from:
        assert not args.restart_from
        old=ROOT/'runs'/args.initialize_from;ck=torch.load(old/'best.pt',map_location='cpu',weights_only=True)
        assert ck['network']==args.network and ck['width']==args.width and ck['channels']==train.x.shape[1]
        model.load_state_dict(ck['state'])
        initial,values=score(model,dev,b)
        atomic_json(out/'warm_start.json',dict(source=str(old),source_epoch=ck['epoch'],monitor=initial,optimizer_restored=False))
    if args.restart_from:
        import hashlib
        import shutil
        old=ROOT/'runs'/args.restart_from
        old_protocol=json.loads((old/'protocol.json').read_text())
        for key in ('mode','network','width','batch','epochs','physics_dir','fit','monitor','confirmation'):
            assert old_protocol[key]==json.loads((out/'protocol.json').read_text())[key],key
        old_best=torch.load(old/'best.pt',map_location='cpu',weights_only=True)
        have_state=(old/'last.pt').is_file()
        if have_state:
            state=torch.load(old/'last.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(state['state'])
            opt.load_state_dict(state['optimizer'])
            gen.set_state(state['shuffle'])
            start_epoch=state['epoch']
        else:
            model.load_state_dict(old_best['state'])
            start_epoch=old_best['epoch']
            for _ in range(start_epoch):torch.randperm(n,generator=gen)
        history=json.loads((old/'history.json').read_text())[:start_epoch]
        assert len(history)==start_epoch and old_best['epoch']<=start_epoch
        best=history[old_best['epoch']-1]['mean']
        shutil.copy2(old/'best.pt',out/'best.pt')
        shutil.copy2(old/'monitor_model_by_map.npy',out/'monitor_model_by_map.npy')
        atomic_json(out/'history.json',history)
        atomic_json(out/'restart.json',dict(source=str(old),model_sha256=hashlib.sha256((old/'best.pt').read_bytes()).hexdigest(),
            start_epoch=start_epoch,optimizer_restored=have_state,shuffle_resumed=True,original_history_length=len(json.loads((old/'history.json').read_text()))))
        print('restart',start_epoch,'restored optimizer',have_state,'best monitor mean',best,flush=True)
    for epoch in range(start_epoch,args.epochs):
        model.train()
        order=torch.randperm(n,generator=gen).to(train.x.device)
        sums=torch.zeros(4,device='cuda',dtype=torch.float64)
        for step,ix in enumerate(order.split(b)):
            t=(epoch+(step+1)/math.ceil(n/b))
            lr=args.initial_lr*min(1,.15+t/.5)*(.12+.88*(.5+.5*math.cos(math.pi*t/args.epochs)))
            for g in opt.param_groups:g['lr']=lr
            w=train.w[ix].to('cuda');y=train.y[ix].to('cuda');x=train.x[ix].to('cuda')
            with torch.autocast('cuda',torch.bfloat16):
                pred=model(x.float())
            err=(pred.float()-y).square()*w
            se=err.sum((0,2,3));ct=w.sum((0,2,3))
            loss=(se/ct.clamp_min(1)).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),8.)
            opt.step();sums+=torch.cat((se.detach(),ct))
        model.eval();rm,per=score(model,dev,b);total=sum(rm)/2
        hist={'epoch':epoch+1,'fit':(sums[:2]/sums[2:]).sqrt().tolist(),'monitor':rm,'mean':total,'seconds':round(time.monotonic()-st,1)}
        history.append(hist);atomic_json(out/'history.json',history);print(json.dumps(hist),flush=True)
        if total<best:
            best=total
            torch.save({'state':model.state_dict(),'width':args.width,'channels':train.x.shape[1],'mode':args.mode,'network':args.network,'epoch':epoch+1},out/'best.pt.tmp')
            os.replace(out/'best.pt.tmp',out/'best.pt')
            np.save(out/'monitor_model_by_map.npy',per)
        torch.save({'state':model.state_dict(),'optimizer':opt.state_dict(),'shuffle':gen.get_state(),'epoch':epoch+1},out/'last.pt.tmp')
        os.replace(out/'last.pt.tmp',out/'last.pt')
    ck=torch.load(out/'best.pt',weights_only=True);model.load_state_dict(ck['state'])
    del train,dev
    confirmation=Panel(con,'dev',args.physics_dir,args.host_panel)
    cb,cbr=score(None,confirmation,b);cr,crr=score(model,confirmation,b)
    np.save(out/'monitor_baseline_by_map.npy',base_rows);np.save(out/'confirmation_baseline_by_map.npy',cbr)
    np.save(out/'confirmation_model_by_map.npy',crr)
    allb=np.vstack((base_rows,cbr));allr=np.vstack((np.load(out/'monitor_model_by_map.npy'),crr))
    to_rm=lambda a:np.sqrt(a[:,:2].sum(0)/a[:,2:].sum(0)).tolist()
    report={'best_epoch':ck['epoch'],'monitor_baseline':base,'monitor':history[ck['epoch']-1]['monitor'],
            'confirmation_baseline':cb,'confirmation':cr,'combined128_baseline':to_rm(allb),'combined128':to_rm(allr),
            'elapsed':time.monotonic()-st}
    atomic_json(out/'result.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
