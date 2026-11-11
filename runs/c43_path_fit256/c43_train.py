"""Matched C43 A/B/C curriculum and independent FP64 evaluation; official TRAIN only."""
import sys,os,json,time,math,argparse,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard,atomic_json,check_source_unchanged
import numpy as np
import torch
from c43_model import Model
from c43_data import Data,gather
A=R/'audit/c43_architecture_20261102'
def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def setup(seed):
    gpu=gpu_guard();torch.cuda.set_per_process_memory_fraction(.8)
    torch.set_num_threads(8);torch.manual_seed(seed);np.random.seed(seed)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    return gpu
def predict(net,dat,pixels):
    x,tx,fs,y,m=dat
    with torch.autocast('cuda',dtype=torch.bfloat16):pr=net(x,tx,pixels)
    return pr.float()*20+gather(fs,pixels),gather(y,pixels).float(),gather(m,pixels).float()
@torch.inference_mode()
def evaluate(net,data,ids,pixels,batch=4,save=None):
    was=net.training;net.eval();ss=np.zeros(2);cc=np.zeros(2);per=[];predictions=[]
    fixed=torch.as_tensor(pixels,dtype=torch.long,device='cuda')
    for start in range(0,len(ids),batch):
        sub=ids[start:start+batch];dat=data.batch(sub);B=len(sub)
        with torch.autocast('cuda',dtype=torch.bfloat16):state=net.encode(dat[0],dat[1])
        ps=[]
        for lo in range(0,len(fixed),2048):
            p=fixed[lo:lo+2048][None].expand(B,-1)
            with torch.autocast('cuda',dtype=torch.bfloat16):q=net.decode(state,p)
            ps.append(q.float()*20+gather(dat[2],p))
        pred=torch.cat(ps,1).cpu().numpy().astype(np.float32)
        target=gather(dat[3],fixed[None].expand(B,-1)).float().cpu().numpy()
        valid=gather(dat[4],fixed[None].expand(B,-1)).cpu().numpy().astype(bool)
        assert np.isfinite(pred).all()
        # Independent CPU float64 scoring, same denominator as official metric.
        for i,pr,yy,mm in zip(sub,pred,target,valid):
            e=pr.astype(np.float64)-yy.astype(np.float64)
            s=(e*e*mm).sum(0);c=mm.sum(0);ss+=s;cc+=c
            per.append({'index':int(i),'terrain':data.rows[int(i)]['terrain_id'],'SSE':s.tolist(),'count':c.tolist()})
        if save is not None:predictions.extend(pred)
    rm=np.sqrt(ss/cc);out={'RMSE':rm.tolist(),'combined':float(rm.mean()),'SSE':ss.tolist(),'count':cc.tolist(),'maps':len(ids),'pixels_per_map':len(pixels)}
    if save is not None:
        path=Path(save);np.save(path.with_suffix('.npy'),np.array(predictions,np.float32))
        atomic_json(path.with_suffix('.json'),{**out,'per_map':per,'indices':list(map(int,ids)),'pixels':'all row-major' if len(pixels)==65536 else list(map(int,pixels))})
    net.train(was);return out
@torch.inference_mode()
def mechanism(net,data,ids):
    if net.variant=='cnn':return {'attention_used':False}
    was=net.training;net.eval();dat=data.batch(ids[:4])
    pix=torch.arange(0,65536,32,device='cuda')[None].expand(min(4,len(ids)),-1)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        state=net.encode(dat[0],dat[1]);normal=net.decode(state,pix)
        altered={**state,'kv':[v.flip(-1) for v in state['kv']]}
        flipped=net.decode(altered,pix)
        variant=net.variant;net.variant='cnn';zero=net.decode(state,pix);net.variant=variant
    out={'attention_used':True,
         'spatial_KV_flip_mean_abs_delta_db':((normal-flipped).abs().mean((0,1))*20).tolist(),
         'zero_attention_mean_abs_delta_db':((normal-zero).abs().mean((0,1))*20).tolist(),
         'interpretation':'Internal branch sensitivity only; not evidence that attention recovers actual paths'}
    net.train(was);return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['cnn','grid','path'],required=True)
    p.add_argument('--stage',choices=['fit64','fit256','probe'],required=True)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--presentations',type=int)
    p.add_argument('--name');p.add_argument('--parent');p.add_argument('--seed',type=int,default=20261102)
    a=p.parse_args();gpu=setup(a.seed);check_source_unchanged()
    protocol=json.loads((A/'protocol.json').read_text())
    if a.stage=='probe':return probe(a,protocol,gpu)
    run=R/'runs'/a.name;run.mkdir(exist_ok=False)
    for src in ['c43_model.py','c43_data.py','c43_train.py']:shutil.copy2(R/'src'/src,run/src)
    trainids=protocol[a.stage+'_ids'];dev=protocol['dev64_ids']
    data=Data(trainids+dev)
    net=Model(a.variant).cuda().to(memory_format=torch.channels_last)
    initial=digest(net)
    if a.parent:
        ck=torch.load(R/'runs'/a.parent/'last.pt',map_location='cpu',weights_only=False)
        assert ck['config']['variant']==a.variant and ck['config']['stage']=='fit64'
        net.load_state_dict(ck['model'])
    presentations=a.presentations or (64000 if a.stage=='fit64' else 128000)
    assert presentations%a.batch==0
    steps=presentations//a.batch
    rng=np.random.default_rng(a.seed+(0 if a.stage=='fit64' else 1))
    order=np.concatenate([rng.permutation(trainids) for _ in range(math.ceil(presentations/len(trainids)))])[:presentations]
    np.save(run/'order.npy',order)
    generator=torch.Generator(device='cuda').manual_seed(a.seed+(0 if a.stage=='fit64' else 100))
    lrmax=2e-4*a.batch/16
    opt=torch.optim.AdamW(net.parameters(),lr=lrmax,weight_decay=1e-4)
    config={**vars(a),'gpu':gpu,'fit_maps':len(trainids),'fit_terrains':len({data.rows[i]['terrain_id'] for i in trainids}),
            'dev_maps':len(dev),'parameters':sum(p.numel() for p in net.parameters()),'presentations':presentations,'steps':steps,
            'points_per_map':2048,'initial_state_sha256':initial,'starting_state_sha256':digest(net),
            'order_sha256':hashlib.sha256(order.tobytes()).hexdigest(),'initial_lr':lrmax,
            'target':'(PL-FSPL)/20','loss':'per-band valid-pixel MSE, then band mean',
            'official_VAL_loaded':False,'augmentation':False,'source_hashes':{s:hashlib.sha256((run/s).read_bytes()).hexdigest() for s in ['c43_model.py','c43_data.py','c43_train.py']}}
    atomic_json(run/'config.json',config)
    monpixels=np.array(protocol['monitor_pixels'],np.int64)
    fitmon=protocol['monitor_fit_ids'] if a.stage=='fit64' else np.random.default_rng(9).choice(trainids,32,replace=False).tolist()
    devmon=protocol['monitor_dev_ids']
    torch.cuda.synchronize();t0=time.time();hist=[];best=float('inf');beststep=0;net.train();step_times=[]
    interval=max(100,round(8000/a.batch))
    for step in range(1,steps+1):
        ts=time.perf_counter()
        ids=order[(step-1)*a.batch:step*a.batch];dat=data.batch(ids)
        pix=torch.randint(65536,(a.batch,2048),device='cuda',generator=generator,dtype=torch.long)
        lr=lrmax*min(1,step/(steps*.05))*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/steps)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):q=net(dat[0],dat[1],pix)
        yy=(gather(dat[3],pix).float()-gather(dat[2],pix))/20;mm=gather(dat[4],pix).float()
        losses=((q.float()-yy).square()*mm).sum((0,1))/mm.sum((0,1)).clamp_min(1)
        loss=losses.mean();assert torch.isfinite(loss)
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(gn)
        opt.step()
        if step<=40 or step%100==0:torch.cuda.synchronize();step_times.append(time.perf_counter()-ts)
        if step==40:
            print(json.dumps({'event':'throughput','variant':a.variant,'stage':a.stage,'step_seconds':float(np.median(step_times[10:])),'estimated_training_minutes':float(np.median(step_times[10:])*steps/60)}),flush=True)
        if step%interval==0 or step==steps:
            fit=evaluate(net,data,fitmon,monpixels,batch=min(4,a.batch))
            val=evaluate(net,data,devmon,monpixels,batch=min(4,a.batch))
            if val['combined']<best:
                best=val['combined'];beststep=step
                torch.save({'model':net.state_dict(),'step':step,'config':config},run/'best.pt')
            rec={'step':step,'steps':steps,'presentations':step*a.batch,'FIT':fit,'DEV':val,'lr':lr,'best_step':beststep,'seconds':time.time()-t0}
            hist.append(rec);atomic_json(run/'history.json',hist);atomic_json(run/'progress.json',rec);print(json.dumps(rec),flush=True)
    torch.save({'model':net.state_dict(),'step':steps,'config':config},run/'last.pt')
    net.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model'])
    devfull=evaluate(net,data,dev,np.arange(65536),batch=2,save=run/'dev64_full')
    fitfull=evaluate(net,data,fitmon,np.arange(65536),batch=2,save=run/'fit_panel_full')
    out={'status':'complete','variant':a.variant,'stage':a.stage,'best_step':beststep,'last_step':steps,'seconds':time.time()-t0,
         'DEV64_full':devfull,'FIT_panel_full':fitfull,'C38_DEV64':protocol['C38_DEV64'],
         'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9,'official_VAL_loaded':False,'parameters':config['parameters'],'mechanism':mechanism(net,data,fitmon)}
    atomic_json(run/'result.json',out);print(json.dumps(out),flush=True)

def probe(a,p,gpu):
    data=Data(p['fit64_ids'][:16]);dat=data.batch(p['fit64_ids'][:2])
    records={};hashes=[]
    for variant in ['grid','path']:
        torch.manual_seed(a.seed);net=Model(variant).cuda().to(memory_format=torch.channels_last)
        hashes.append(digest(net));pix=torch.arange(0,65536,256,device='cuda')[None].expand(2,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(dat[0],dat[1],pix)
        loss=pred.square().mean();loss.backward()
        grads={name:float(v.grad.norm()) for name,v in net.named_parameters() if name in ['offset.weight','encoder.stem.0.weight','kv.0.weight']}
        assert all(np.isfinite(v) and v>0 for v in grads.values()),grads
        net.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            st=net.encode(dat[0],dat[1]);normal=net.decode(st,pix)
            altered={**st,'kv':[v.flip(-1) for v in st['kv']]}
            changed=net.decode(altered,pix)
        delta=float((normal-changed).abs().mean()*20)
        assert delta>1e-5
        records[variant]={'parameters':sum(x.numel() for x in net.parameters()),'gradients':grads,'KV_spatial_flip_delta_db':delta}
        del net;torch.cuda.empty_cache()
    assert len(set(hashes))==1
    # Pixel coordinate check independent of the production geometry code.
    from torch.nn import functional as F
    yy,xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij')
    ramp=(xx+2*yy)[None,None];points=torch.tensor([[[[0.,0.],[255.,255.],[31.,83.]]]],device='cuda')
    got=F.grid_sample(ramp,points/127.5-1,align_corners=True)[0,0,0]
    assert float((got-torch.tensor([0.,765.,197.],device='cuda')).abs().max())<1e-4
    bench=[]
    for bs in [4,8,16]:
        torch.manual_seed(a.seed);net=Model('path').cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=.0002)
        dat=data.batch((p['fit64_ids'][:16]*4)[:bs]);pix=torch.randint(65536,(bs,2048),device='cuda')
        elapsed=[]
        for k in range(6):
            torch.cuda.synchronize();ts=time.perf_counter();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):q=net(dat[0],dat[1],pix)
            yy=(gather(dat[3],pix).float()-gather(dat[2],pix))/20;mm=gather(dat[4],pix).float()
            loss=(((q.float()-yy).square()*mm).sum((0,1))/mm.sum((0,1))).mean()
            loss.backward();opt.step();torch.cuda.synchronize();elapsed.append(time.perf_counter()-ts)
        seconds=float(np.median(elapsed[2:]));bench.append({'batch':bs,'step_seconds':seconds,'maps_per_second':bs/seconds,'peak_GB':torch.cuda.max_memory_allocated()/1e9})
        print(json.dumps({'event':'benchmark',**bench[-1]}),flush=True)
        del opt,net,dat;torch.cuda.empty_cache()
    out={'GPU':gpu,'checks':records,'B_C_initial_weights_identical':True,'coordinate_check':True,'benchmarks':bench}
    atomic_json(A/'probe.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':main()
