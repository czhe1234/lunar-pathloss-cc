"""Reproducible input-only predictions from a checkpoint; no target/mask arrays read."""
import os,sys,fcntl,json,csv,time,argparse,hashlib,importlib.util,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT, DATA, GPU_UUID, gpu_guard, atomic_json
from input_features import make_inputs
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--part',choices=['full','dev'],default='full');p.add_argument('--batch',type=int,default=32);p.add_argument('--fp32',action='store_true');p.add_argument('--limit',type=int,default=0);p.add_argument('--latency-samples',type=int,default=64);args=p.parse_args()
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=gpu_guard();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
run=ROOT/'runs'/args.run;config=json.loads((run/'config.json').read_text());mode=config['mode']
snapshot=run/'source/cc_core.py';spec=importlib.util.spec_from_file_location('snapshot_core',snapshot);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==config['source_sha256']['cc_core.py']
ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False);model=core.model_for(mode,config['width']);model.load_state_dict(ck['model']);model.eval()
rows=list(csv.DictReader((DATA/f'{args.split}_index.csv').open()));indices=np.arange(len(rows))
if args.part=='dev':
    assert args.split=='train'
    dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids'])
    indices=np.array([i for i,r in enumerate(rows) if r['terrain_id'] in dev],dtype=np.int64)
if args.limit:indices=indices[:args.limit]
label=f'{args.split}_{args.part}'+('_fp32' if args.fp32 else '_bf16')+(f'_first{args.limit}' if args.limit else '')
out=run/'predictions'/label
if out.exists():raise RuntimeError(f'Refuse to overwrite predictions: {out}')
out.mkdir(parents=True)
(out/'source').mkdir()
for f in [Path(__file__),ROOT/'src/input_features.py',snapshot]:shutil.copy2(f,out/'source'/f.name)
hm=np.load(DATA/f'{args.split}_hm.npy',mmap_mode='r',allow_pickle=False)
predictions=[np.lib.format.open_memmap(out/f'pl{b}.npy',mode='w+',dtype=np.float32,shape=(len(indices),256,256)) for b in ('415','58')]
atomic_json(out/'indices.json',{'source_split':args.split,'index_rows':indices.tolist(),'sample_ids':[rows[i]['sample_id'] for i in indices]})

def predict_batch(ii):
    rr=[rows[i] for i in ii]
    h=torch.from_numpy(np.array(hm[[int(r['hm_row']) for r in rr]],copy=True)).cuda()
    r=torch.tensor([int(a['tx_row']) for a in rr],device='cuda');c=torch.tensor([int(a['tx_col']) for a in rr],device='cuda')
    x=make_inputs(h,r,c,mode)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not args.fp32):pred=model(x)*core.SCALE+core.OFFSET
    return pred.float().cpu().numpy()

with torch.inference_mode():
    for _ in range(3):predict_batch(indices[:min(len(indices),args.batch)])
    start=time.perf_counter()
    for lo in range(0,len(indices),args.batch):
        pred=predict_batch(indices[lo:lo+args.batch])
        assert np.isfinite(pred).all()
        for j,arr in enumerate(predictions):arr[lo:lo+len(pred)]=pred[:,j]
    for arr in predictions:arr.flush()
    elapsed=time.perf_counter()-start
    single=[]
    for _ in range(3):predict_batch(indices[:1])
    for i in indices[:args.latency_samples]:
        start=time.perf_counter();predict_batch([i]);single.append((time.perf_counter()-start)*1000)
summary={'run':args.run,'checkpoint_epoch':ck['epoch'],'split':args.split,'part':args.part,'mode':mode,'precision':'FP32 with TF32 allowed' if args.fp32 else 'BF16 convolutions / FP32 head','maps_per_band':len(indices),'output_dtype':'float32','batch':args.batch,'batch_seconds_including_input_features_transfer_inference_output_write':elapsed,'batch_ms_per_two_band_map':elapsed/len(indices)*1000,'single_request_warm_ms':{'count':len(single),'mean':float(np.mean(single)),'p50':float(np.median(single)),'p95':float(np.percentile(single,95)),'max':float(np.max(single))},'latency_scope':'Includes input array gather, CPU-to-GPU, all derived features (including profiles when enabled), eager model forward, GPU-to-CPU. Startup/model loading excluded; single-request filesystem output excluded. Heightmap source may be in OS page cache. One request returns both bands. Hardware-specific; not the unconfirmed official hardware benchmark.','device':info,'inputs_used':[f'{args.split}_hm.npy',f'{args.split}_index.csv'],'no_target_or_validity_mask_read':True,'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'src/input_features.py',snapshot]}}
atomic_json(out/'inference.json',summary);print(json.dumps(summary),flush=True)
