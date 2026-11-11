"""Input-only forward evaluation of existing checkpoints. No optimizer or backward pass."""
import os,sys,json,csv,time,hashlib,fcntl,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cc_core import gpu_guard,GPU_UUID
from input_features import make_inputs
import numpy as np
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==GPU_UUID
lock=(ROOT/'.gpu3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
device=gpu_guard();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
OUT=ROOT/'audit/geometry_model_diagnosis_20261031';sel=json.loads((OUT/'frozen_inference_selection.json').read_text());indices=sel['samples']['fit']+sel['samples']['dev'];rows=list(csv.DictReader((ROOT.parent/'data/train_index.csv').open()));H=np.load(ROOT.parent/'data/train_hm.npy',mmap_mode='r',allow_pickle=False)
folder=OUT/'frozen_predictions';folder.mkdir(exist_ok=False)
(folder/'indices.json').write_text(json.dumps({'source_split':'train','index_rows':indices,'parts':['fit']*len(sel['samples']['fit'])+['dev']*len(sel['samples']['dev'])},indent=2)+'\n')
summary={'device':device,'training':False,'optimizer_created':False,'backward_called':False,'inputs_used':['train_hm.npy','train_index.csv'],'targets_or_masks_read':False,'runs':{}}
start=time.time()
for runname in ('b0_raw_s20261031','b1_geometry_s20261031'):
 run=ROOT/'runs'/runname;config=json.loads((run/'config.json').read_text());source=run/'source/cc_core.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==config['source_sha256']['cc_core.py']
 spec=importlib.util.spec_from_file_location('frozen_core',source);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 path=run/'best.pt';before=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cuda',weights_only=False);model=core.model_for(config['mode'],config['width']);model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
 assert not any(p.requires_grad for p in model.parameters())
 for precision in ('bf16','fp32'):
  torch.backends.cuda.matmul.allow_tf32=precision=='bf16';torch.backends.cudnn.allow_tf32=precision=='bf16'
  label=runname.split('_')[0]+'_'+precision;output=np.lib.format.open_memmap(folder/(label+'.npy'),mode='w+',dtype=np.float32,shape=(len(indices),2,256,256));t0=time.time()
  with torch.inference_mode():
   for lo in range(0,len(indices),32):
    ii=indices[lo:lo+32];rr=[rows[i] for i in ii];h=torch.from_numpy(np.array(H[[int(r['hm_row']) for r in rr]],copy=True)).cuda();tr=torch.tensor([int(r['tx_row']) for r in rr],device='cuda');tc=torch.tensor([int(r['tx_col']) for r in rr],device='cuda');x=make_inputs(h,tr,tc,config['mode'])
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):prediction=model(x)*core.SCALE+core.OFFSET
    array=prediction.float().cpu().numpy();assert np.isfinite(array).all();output[lo:lo+len(ii)]=array
    if (lo+len(ii))%256==0:print(json.dumps({'phase':label,'maps':lo+len(ii),'total':len(indices),'seconds':time.time()-t0}),flush=True)
  output.flush();summary['runs'][label]={'checkpoint':str(path),'checkpoint_epoch':ck['epoch'],'checkpoint_sha256':before,'precision':precision,'tf32_allowed':precision=='bf16','seconds':time.time()-t0,'maps':len(indices),'mode':config['mode'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()};del output
 assert hashlib.sha256(path.read_bytes()).hexdigest()==before
 del model,ck;torch.cuda.empty_cache()
summary['seconds']=time.time()-start;summary['peak_allocated_gb']=torch.cuda.max_memory_allocated()/1e9
(folder/'inference.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'done':True,**summary}),flush=True)
