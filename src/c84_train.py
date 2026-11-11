"""Matched paired-TX fine-tuning of the existing true visibility graph. Official TRAIN only."""
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c73_graph import GraphField
from src.c73_train import TerrainData
A=R/'audit/c84_tx'
def mm(p):return np.load(p,mmap_mode='r')
class LocalReadout:
    def __init__(self):
        folder=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1'
        self.ids=json.load(open(folder/'indices.json'))['index_rows'];self.pos={i:j for j,i in enumerate(self.ids)}
        self.base=[mm(folder/f'pl{b}.npy') for b in ('415','58')]
        self.corr=mm(R/'runs/c72_observed/full_internal/correction.npy');self.graph=mm(R/'runs/c73_wide_real/full_internal/prediction.npy')
    def fields(self,ids):
        js=[self.pos[i] for i in ids]
        p0=np.stack([a[js] for a in self.base],1).astype('float32')+self.corr[js].astype('float32')
        return torch.as_tensor(p0,device='cuda'),torch.as_tensor(self.graph[js].astype('float32'),device='cuda')
@torch.inference_mode()
def evaluate(net,data,local,batch):
    net.eval();out=[]
    for j in range(0,len(data.ids),batch):
        ids=torch.arange(j,min(len(data.ids),j+batch),device='cuda')
        x,h,r,c,p0,y,w=data.batch(ids)
        a,g=local.fields(data.ids[j:j+len(ids)])
        with torch.autocast('cuda',torch.bfloat16):b=p0+net(x,h,r,c).float()
        pp=[.5*(a+g),.5*(a+b),g,b]
        zones=[w,w&(x[:,11+16:11+17]>.99),w&(x[:,11+16:11+17]<=.99)]
        def se(m):
            return torch.stack([((v-y).square()*m).sum((2,3),dtype=torch.float64) for v in pp]+[m.sum((2,3))],2)
        out.append(torch.stack([se(m) for m in zones],2).cpu().numpy())
    return np.concatenate(out)
def rms(rows):
    v=rows.sum(0);return np.sqrt(v[...,:4]/v[...,4:])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=['old','new','delta'])
    ap.add_argument('--epochs',default=6,type=int);ap.add_argument('--confirm',action='store_true');args=ap.parse_args()
    gpu=gpu_guard();torch.manual_seed(84440);np.random.seed(84440);torch.set_num_threads(5)
    torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('high')
    plan=json.load(open(A/'protocol.json'));num_pairs=plan['multi_pairs'];out=R/'runs'/('c84_'+args.arm);out.mkdir(exist_ok=True)
    parent=R/'runs/c73_wide_real/best.pt';local=LocalReadout()
    with parent.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==plan['start_checkpoint_sha']
    ck=torch.load(out/'best.pt' if args.confirm else parent,weights_only=True,map_location='cpu')
    net=GraphField(56,ck['width'],False).to('cuda',memory_format=torch.channels_last)
    net.load_state_dict(ck['state']);t=time.monotonic();batch=8
    if args.confirm:
        rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
        terrains={r['terrain'] for r in json.load(open(R/'audit/c79_receiver/protocol.json'))['panel'] if r['group']=='confirm'}
        allids=[i for i in local.ids if rows[i]['terrain_id'] in terrains];assert len(allids)==1934
        measurements=[]
        for j in range(0,len(allids),128):
            data=TerrainData(allids[j:j+128],['features/c73_compact','features/c73_dev_extra_compact'])
            measurements.append(evaluate(net,data,local,batch));del data
            if j%384==0:print('confirmed',j+len(measurements[-1]),len(allids),flush=True)
        z=np.concatenate(measurements);v=rms(z)
        np.savez(out/'confirm.npz',sse=z,indices=allids,terrain=[rows[i]['terrain_id'] for i in allids])
        atomic_json(out/'confirm.json',dict(terrains=len(terrains),maps=len(allids),rmse=v[:,:,1].tolist(),gain=(v[:,:,0]-v[:,:,1]).tolist(),
                    graph_gain=(v[:,:,2]-v[:,:,3]).tolist(),epoch=ck['epoch']))
        print(args.arm,'confirmation',np.round(v[:,:,0]-v[:,:,1],4).tolist(),flush=True);return
    filename='old' if args.arm=='old' else 'new';pairs=np.array(json.load(open(A/(filename+'.json'))))
    ids=pairs.flatten().tolist();assert len(set(ids))==len(ids)
    resident=torch.cuda.mem_get_info()[0]>63*2**30
    fit=TerrainData(ids,['features/c73_compact','features/c84_extra'],resident=resident)
    monitor=TerrainData(plan['monitor'],['features/c73_compact','features/c73_dev_extra_compact'])
    opt=torch.optim.AdamW(net.parameters(),lr=.00032,weight_decay=.012,fused=True)
    gen=torch.Generator().manual_seed(84660);history=[];best=1e6
    atomic_json(out/'config.json',dict(arm=args.arm,gpu=gpu,parent=plan['start_checkpoint_sha'],fit_rows=len(ids),epochs=args.epochs,
        resident=resident,batch=batch,selection='mean of 2 masked C76-blend RMSEs on 48 terrain panel',full_VAL_used=False))
    z0=evaluate(net,monitor,local,batch);r0=rms(z0);assert np.max(abs(r0[:,:,0]-r0[:,:,1]))<.025
    atomic_json(out/'initial.json',dict(metric=r0.tolist(),frozen_graph_replay_difference=(r0[:,:,0]-r0[:,:,1]).tolist()))
    print('loaded',len(ids),'maps',resident,'seconds',round(time.monotonic()-t,1),flush=True)
    for ep in range(1,args.epochs+1):
        net.train();order=torch.randperm(len(pairs),generator=gen);nstep=math.ceil(len(order)/(batch//2))
        for step in range(nstep):
            loc=order[step*(batch//2):(step+1)*(batch//2)].cuda()
            ix=torch.stack([loc*2,loc*2+1],1).flatten();x,h,r,c,p,y,w=fit.batch(ix)
            with torch.autocast('cuda',torch.bfloat16):g=p+net(x,h,r,c).float()
            err=(g-y);weight=w.sum((0,2,3)).clamp_min(1);loss=((err.square()*w).sum((0,2,3))/weight).mean()/25
            if args.arm=='delta':
                mask=(w[::2]&w[1::2])&(loc<num_pairs)[:,None,None,None]
                d=((err[::2]-err[1::2]).square()*mask).sum((0,2,3))/mask.sum((0,2,3)).clamp_min(1)
                loss=loss+.22*d.mean()/25
            lr=.00032*(.1+.9*.5*(1+math.cos(math.pi*((ep-1)+(step+.5)/nstep)/args.epochs)))
            for param in opt.param_groups:param['lr']=lr
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
            if (step+1)%160==0:
                atomic_json(out/'progress.json',dict(epoch=ep,step=step+1,total_steps=nstep,seconds=time.monotonic()-t))
        z=evaluate(net,monitor,local,batch);vals=rms(z);merit=float(vals[:,0,1].mean())
        rec=dict(epoch=ep,gain=(vals[:,:,0]-vals[:,:,1]).tolist(),rmse=vals[:,:,1].tolist(),graph_gain=(vals[:,:,2]-vals[:,:,3]).tolist(),seconds=round(time.monotonic()-t,1))
        history.append(rec);atomic_json(out/'history.json',history)
        if merit<best:
            best=merit;torch.save(dict(state=net.state_dict(),width=ck['width'],shuffle=False,epoch=ep),out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
            np.save(out/'best_sse.npy',z)
        print(ep,'gain',np.round(vals[:,:,0]-vals[:,:,1],4)[:,0].tolist(),'seconds',rec['seconds'],flush=True)
    atomic_json(out/'complete.json',dict(arm=args.arm,best=best,epochs=args.epochs,seconds=time.monotonic()-t))
if __name__=='__main__':main()
