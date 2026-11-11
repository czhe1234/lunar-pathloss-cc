"""C5 reproducible input-only predictor; labels and masks are never opened."""
import sys,csv,json,time,argparse,fcntl,hashlib,importlib.util,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,gpu_guard,atomic_json
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--part',choices=['dev','full'],default='full');p.add_argument('--batch',type=int,default=8);p.add_argument('--latency-samples',type=int,default=64);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=gpu_guard();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
run=ROOT/'runs'/args.run;config=json.loads((run/'config.json').read_text())
mods={}
for name in ['c3_geometry','c4_geometry','c4_model','c5_model']:
 f=run/'source'/f'{name}.py';assert hashlib.sha256(f.read_bytes()).hexdigest()==config['source_sha256'][f.name]
 spec=importlib.util.spec_from_file_location('snapshot_'+name,f);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);mods[name]=mod
ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False);model=mods['c5_model'].Model(config['width']).to('cuda',memory_format=torch.channels_last);model.load_state_dict(ck['model']);model.eval()
rows=list(csv.DictReader((DATA/f'{args.split}_index.csv').open()));indices=np.arange(len(rows))
if args.part=='dev':
 assert args.split=='train';dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids']);indices=np.array([i for i,r in enumerate(rows) if r['terrain_id'] in dev],dtype=np.int64)
if args.limit:indices=indices[:args.limit]
out=run/'predictions'/(f'{args.split}_{args.part}_bf16'+(f'_first{args.limit}' if args.limit else ''));out.mkdir(parents=True,exist_ok=False)
(out/'source').mkdir();shutil.copy2(Path(__file__),out/'source'/Path(__file__).name)
hm=np.load(DATA/f'{args.split}_hm.npy',mmap_mode='r')
arrays=[np.lib.format.open_memmap(out/f'pl{band}.npy',mode='w+',dtype=np.float32,shape=(len(indices),256,256)) for band in ['415','58']]
atomic_json(out/'indices.json',{'source_split':args.split,'index_rows':indices.tolist(),'sample_ids':[rows[i]['sample_id'] for i in indices]})
@torch.inference_mode()
def predict(ii):
 rr=[rows[i] for i in ii];h=torch.from_numpy(np.array(hm[[int(r['hm_row']) for r in rr]],copy=True)).cuda();r=torch.tensor([int(a['tx_row']) for a in rr],device='cuda');c=torch.tensor([int(a['tx_col']) for a in rr],device='cuda')
 x,fs=mods['c4_geometry'].make_inputs(h,r,c)
 with torch.autocast('cuda',dtype=torch.bfloat16):pr=model(x)+fs
 return pr.float().cpu().numpy()
start=time.perf_counter();predict(indices[:1]);cold=(time.perf_counter()-start)*1000
for _ in range(3):predict(indices[:args.batch])
start=time.perf_counter()
for lo in range(0,len(indices),args.batch):
 pr=predict(indices[lo:lo+args.batch]);assert np.isfinite(pr).all()
 for j,arr in enumerate(arrays):arr[lo:lo+len(pr)]=pr[:,j]
 if lo%(args.batch*100)==0:print(json.dumps({'event':'predict','done':lo+len(pr),'total':len(indices),'seconds':time.perf_counter()-start}),flush=True)
for arr in arrays:arr.flush()
elapsed=time.perf_counter()-start
for _ in range(3):predict(indices[:1])
lat=[]
for i in indices[:args.latency_samples]:
 t=time.perf_counter();predict([i]);lat.append((time.perf_counter()-t)*1000)
summary={'run':args.run,'checkpoint_epoch':ck['epoch'],'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),'split':args.split,'part':args.part,'precision':'BF16 convolutions / FP32 stem, geometry, final scalar head and observation mixture','maps_per_band':len(indices),'output_dtype':'float32','batch':args.batch,'batch_seconds_including_input_features_transfer_inference_output_write':elapsed,'batch_ms_per_two_band_map':elapsed/len(indices)*1000,'single_request_cold_ms':cold,'single_request_warm_ms':{'count':len(lat),'mean':float(np.mean(lat)),'p50':float(np.median(lat)),'p95':float(np.percentile(lat,95)),'max':float(np.max(lat))},'latency_scope':'Includes input gather, H2D transfer, ALL on-demand profile and co-visibility features, model, inverse transform and D2H. One request returns both bands. Model loading excluded. Warm latency uses warmed allocator/CUDA kernels. Official hardware and warm-up protocol unconfirmed.','device':info,'inputs_used':[f'{args.split}_hm.npy',f'{args.split}_index.csv'],'no_target_or_validity_mask_read':True,'source_sha256':config['source_sha256']}
atomic_json(out/'inference.json',summary);print(json.dumps(summary),flush=True)
