"""C48 raw2D terrain residual fitting and full-data training."""
import sys,os,json,time,math,argparse,hashlib,shutil,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,check_source_unchanged
import numpy as np,torch
from src.c34_model import Model as Teacher
from src.c48_model import Model
from src.c47_model import physical_profile
from tqdm import trange
A=R/'audit/c48_fresnel_corridor_20261102'
def terrain_fields():
    h=device_array(R.parent/'data/train_hm.npy',dtype=torch.float32)
    sy,sx=torch.gradient(h,dim=(-2,-1))
    fields=torch.stack((h,sx,sy),-1).contiguous()
    rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
    info=torch.tensor([[int(r['hm_row']),int(r['tx_row']),int(r['tx_col'])] for r in rows],device='cuda',dtype=torch.long)
    del h,sx,sy;torch.cuda.empty_cache()
    print(json.dumps({'event':'raw_terrain','shape':list(fields.shape),'GB':fields.numel()*4/1e9}),flush=True)
    return fields,info
def load_teacher():
    ck=torch.load(R/'runs/c38_1024_continue/best.pt',map_location='cpu',weights_only=False)
    return ck['model'],ck['config']
def teacher_net(s,c):
    net=Teacher(c['mode'],c['members'],c['width'],s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda().eval()
    net.load_state_dict(s)
    for p in net.parameters():p.requires_grad_(False)
    return net
def device_array(p,selection=None,dtype=None):
    a=np.load(p,mmap_mode='r')
    if selection is not None:a=np.asarray(a[selection])
    shape=a.shape
    out=torch.empty(shape,dtype=dtype or torch.from_numpy(np.empty((),a.dtype)).dtype,device='cuda')
    for lo in range(0,len(a),262144):
        out[lo:lo+262144].copy_(torch.from_numpy(np.array(a[lo:lo+262144],copy=True)))
    return out
def data(part,selection=None,usecache=True,state=None,cfg=None):
    t=time.time();folder='c38_dense' if part=='fit' else 'c10_points'
    label='c38_dense' if part=='fit' else 'c6_points'
    d={'x':device_array(R/f'features/{folder}/{part}_x.npy',selection,torch.float32)}
    for k in ['y','fspl','mask']:d[k]=device_array(R/f'features/{label}/{part}_{k}.npy',selection,torch.float32)
    d['target']=(d['y']-d['fspl'])/20
    mapids=np.load(R/('features/c38_dense/fit_map_indices.npy' if part=='fit' else 'features/c6_points/dev_map_indices.npy'))
    maps=torch.repeat_interleave(torch.tensor(mapids,device='cuda',dtype=torch.long),1024 if part=='fit' else 256)
    d['map']=maps if selection is None else maps[torch.as_tensor(selection,device='cuda')]
    del maps
    if usecache:
        meta=json.loads((R/f'features/c47_teacher/{part}_meta.json').read_text())
        assert meta['complete'] and meta['input_only'] and not meta['labels_masks_read']
        d['base']=device_array(R/f'features/c47_teacher/{part}_base.npy',selection,torch.float32)
    else:
        net=teacher_net(state,cfg)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            d['base']=torch.cat([net(d['x'][lo:lo+4096]).mean(1).float() for lo in range(0,len(d['x']),4096)])
        del net;torch.cuda.empty_cache()
    assert d['base'].shape==d['y'].shape
    for lo in range(0,len(d['x']),262144):
        assert torch.isfinite(d['x'][lo:lo+262144]).all() and torch.isfinite(d['base'][lo:lo+262144]).all()
    torch.cuda.empty_cache()
    print(json.dumps({'event':'data','part':part,'rows':len(d['x']),'seconds':time.time()-t,'GPU_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
    return d
@torch.inference_mode()
def evaluate(net,d,ids=None,batch=2048,save=None):
    net.eval();s=torch.zeros(2,device='cuda',dtype=torch.float64);c=torch.zeros_like(s);pred=[]
    n=len(d['x']) if ids is None else len(ids)
    for lo in range(0,n,batch):
        j=slice(lo,min(lo+batch,n)) if ids is None else ids[lo:lo+batch]
        with torch.autocast('cuda',dtype=torch.bfloat16):q=net(d['x'][j],d['map'][j])
        out=(q.float()+d['base'][j])*20+d['fspl'][j]
        e=out.double()-d['y'][j].double();m=d['mask'][j]
        assert torch.isfinite(out).all()
        s+=(e.square()*m).sum(0);c+=m.sum(0).double()
        if save is not None:pred.append(out.cpu().numpy())
    result={'RMSE':(s/c).sqrt().cpu().tolist(),'SSE':s.cpu().tolist(),'count':c.cpu().tolist(),'rows':n}
    result['combined']=sum(result['RMSE'])/2
    if save is not None:np.save(save,np.concatenate(pred))
    net.train();return result
def tiny_selection():
    ids=np.load(R/'features/c38_dense/fit_map_indices.npy')
    rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
    positions=[];terrains=set()
    for pos,i in enumerate(ids):
        terrain=rows[int(i)]['terrain_id']
        if terrain not in terrains:positions.append(pos);terrains.add(terrain)
        if len(positions)==4:break
    return np.concatenate([np.arange(pos*1024,(pos+1)*1024) for pos in positions]),ids[positions].tolist(),sorted(terrains)
def setup(seed):
    g=gpu_guard();torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(.78)
    torch.manual_seed(seed);np.random.seed(seed);return g
def probe(gpu,s,c,fields,map_info):
    sel,ids,terrains=tiny_selection();d=data('fit',sel,usecache=False,state=s,cfg=c);records={}
    seq=physical_profile(d['x'][:16]);assert seq.shape==(16,128,18) and torch.isfinite(seq).all()
    assert (seq[:,:,0]-d['x'][:16,104:232]).abs().max().item()==0
    assert (seq[:,:,10].square()+seq[:,:,11].square()-1).abs().max().item()<2e-6
    for kind in ['strip','patch','both']:
        bench=[]
        for bs in [256,512,1024,2048]:
            torch.manual_seed(20261102);net=Model(kind,s,fields,map_info).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001)
            j=torch.arange(bs,device='cuda')%len(d['x']);x=d['x'][j];maps=d['map'][j];y=d['target'][j]-d['base'][j];m=d['mask'][j]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                assert net(x,maps).abs().max().item()==0
                info=map_info[maps[:8]];got=net.sample(info[:,0],info[:,2,None,None].float(),info[:,1,None,None].float())[:,0,0,0]
                assert (got-x[:8,31]).abs().max().item()<1e-6
            times=[]
            for it in range(7):
                torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):q=net(x,maps)
                loss=(((q.float()-y).square()*m).sum(0)/m.sum(0).clamp_min(1)).mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-t)
            grads={k:float(p.grad.norm()) for k,p in net.named_parameters() if k in ['embedding.frequencies','strip.net.0.weight','patch.net.0.weight']}
            assert all(v>0 and math.isfinite(v) for v in grads.values()),grads
            sec=float(np.median(times[3:]));bench.append({'batch':bs,'step_seconds':sec,'points_per_second':bs/sec,'gradients':grads,'parameters':sum(p.numel() for p in net.parameters())})
            print(json.dumps({'event':'probe','kind':kind,**bench[-1]}),flush=True)
            del net,opt,x,y,m;torch.cuda.empty_cache()
        records[kind]=bench
    atomic_json(A/'probe.json',{'gpu':gpu,'records':records,'tiny_map_indices':ids,'tiny_terrains':terrains,'physics_finite_and_phase_identity':True})
def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['strip','patch','both'],default='both')
    p.add_argument('--scope',choices=['tiny','full','probe'],required=True);p.add_argument('--name')
    p.add_argument('--steps',type=int);p.add_argument('--batch',type=int,default=2048);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=20261102)
    a=p.parse_args();gpu=setup(a.seed);check_source_unchanged();s,c=load_teacher();fields,map_info=terrain_fields()
    if a.scope=='probe':return probe(gpu,s,c,fields,map_info)
    run=R/'runs'/a.name;run.mkdir(exist_ok=False)
    for f in ['c48_train.py','c48_model.py','c47_model.py','c34_model.py']:shutil.copy2(R/'src'/f,run/f)
    if a.scope=='tiny':
        selection,maps,terrains=tiny_selection();fit=data('fit',selection,usecache=False,state=s,cfg=c)
        dev=fit;steps=a.steps or 1000
    else:
        if torch.cuda.mem_get_info()[0]<38*2**30:raise RuntimeError('Need38GiB free before full-data allocation; do not use another card.')
        fit=data('fit');dev=data('dev');maps=np.load(R/'features/c38_dense/fit_map_indices.npy').tolist()
        rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
        terrains=sorted({rows[i]['terrain_id'] for i in maps});assert len(terrains)==5508;steps=a.steps or 8000
        assert not set(terrains)&set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
    torch.manual_seed(a.seed);net=Model(a.kind,s,fields,map_info).cuda()
    cfg={**vars(a),'gpu':gpu,'steps':steps,'fit_rows':len(fit['x']),'dev_rows':len(dev['x']),'fit_maps':len(maps),'fit_terrains':len(terrains),
         'parameters':sum(v.numel() for v in net.parameters()),'teacher':'c38_1024_continue/best.pt','target':'(PL-C38prediction)/20',
         'official_VAL_used':False,'zero_correction_initialization':True,'source_hashes':{f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ['c48_model.py','c48_train.py','c47_model.py','c34_model.py']}}
    atomic_json(run/'config.json',cfg)
    if a.scope=='tiny':atomic_json(run/'subset.json',{'map_indices':maps,'terrain_ids':terrains,'point_selection':selection.tolist()})
    probeids=torch.from_numpy(np.linspace(0,len(fit['x'])-1,min(65536,len(fit['x'])),dtype=np.int64)).cuda()
    val=evaluate(net,dev,batch=a.batch);fp=evaluate(net,fit,probeids,batch=a.batch)
    if a.scope=='full':assert max(abs(v-r) for v,r in zip(val['RMSE'],[2.6446351803304657,2.9987403398240975]))<1e-5,val
    initial=val;best=val['combined'];beststep=0
    torch.save({'model':net.state_dict(),'step':0,'config':cfg},run/'best.pt')
    history=[{'step':0,'FIT':fp,'DEV':val,'best_step':0,'seconds':0.}];atomic_json(run/'history.json',history)
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
    gen=torch.Generator(device='cuda').manual_seed(a.seed);t0=time.time();times=[];net.train()
    interval=200 if a.scope=='tiny' else 1000
    for step in trange(1,steps+1,desc='C48 '+a.kind+' '+a.scope,mininterval=10):
        t=time.perf_counter();j=torch.randint(len(fit['x']),(a.batch,),device='cuda',generator=gen)
        lr=a.lr*min(1,step/(50 if a.scope=='tiny' else 200))*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/steps)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):q=net(fit['x'][j],fit['map'][j])
        yy=fit['target'][j]-fit['base'][j];mm=fit['mask'][j]
        loss=(((q.float()-yy).square()*mm).sum(0)/mm.sum(0).clamp_min(1)).mean()
        assert torch.isfinite(loss);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(gn);opt.step()
        if step<=30:
            torch.cuda.synchronize();times.append(time.perf_counter()-t)
        if step==30:print(json.dumps({'event':'throughput','step_seconds':float(np.median(times[10:])),'estimated_training_minutes':float(np.median(times[10:])*steps/60)}),flush=True)
        if step%interval==0 or step==steps:
            val=evaluate(net,dev,batch=a.batch);fp=evaluate(net,fit,probeids,batch=a.batch)
            if val['combined']<best:
                best=val['combined'];beststep=step;torch.save({'model':net.state_dict(),'step':step,'config':cfg},run/'best.pt')
            rec={'step':step,'steps':steps,'FIT':fp,'DEV':val,'best_step':beststep,'lr':lr,'seconds':time.time()-t0}
            history.append(rec);atomic_json(run/'history.json',history);atomic_json(run/'progress.json',rec);print(json.dumps(rec),flush=True)
            if a.scope=='full' and step>=4000 and step-beststep>=3000:break
    torch.save({'model':net.state_dict(),'step':step,'config':cfg},run/'last.pt')
    net.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model'])
    val=evaluate(net,dev,batch=a.batch,save=run/'dev_prediction.npy');fp=evaluate(net,fit,probeids,batch=a.batch)
    gain=(np.array(initial['RMSE'])-val['RMSE']).tolist()
    out={'status':'complete','scope':a.scope,'kind':a.kind,'initial_C38':initial,'DEV':val,'FIT_probe':fp,'gain_db':gain,
         'best_step':beststep,'last_step':step,'seconds':time.time()-t0,'tiny_both_le_080db':bool(max(fp['RMSE'])<=.8),
         'sampled_DEV_both_gain010db':bool(a.scope=='full' and min(gain)>=.10),'evaluation_scope':('FIT only' if a.scope=='tiny' else 'terrain-disjoint sampled DEV612'),'official_VAL_used':False,'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9}
    atomic_json(run/'result.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
