"""Frozen B1 symmetry diagnosis. Input-only; no optimizer and no backward."""
import os,sys,json,csv,time,hashlib,fcntl,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cc_core import gpu_guard,GPU_UUID
from input_features import make_inputs
import numpy as np
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==GPU_UUID
lock=(ROOT/'.gpu3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=gpu_guard();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
A=ROOT/'audit/geometry_model_diagnosis_20261031';F=A/'frozen_predictions';O=A/'symmetry_predictions';O.mkdir(exist_ok=False);idx=json.loads((F/'indices.json').read_text());indices=idx['index_rows'];rows=list(csv.DictReader((ROOT.parent/'data/train_index.csv').open()));H=np.load(ROOT.parent/'data/train_hm.npy',mmap_mode='r')
run=ROOT/'runs/b1_geometry_s20261031';config=json.loads((run/'config.json').read_text());source=run/'source/cc_core.py';spec=importlib.util.spec_from_file_location('symmetry_core',source);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);cp=run/'best.pt';sha=hashlib.sha256(cp.read_bytes()).hexdigest();ck=torch.load(cp,map_location='cuda',weights_only=False);model=core.model_for(config['mode'],config['width']);model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
# Old/new TX construction must be identical on actual contiguous input batches.
spec=importlib.util.spec_from_file_location('oldinputs',ROOT/'runs/b0_raw_s20261031/predictions/train_dev_bf16/source/input_features.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
outs={n:np.lib.format.open_memmap(O/(n+'.npy'),mode='w+',dtype=np.float32,shape=(len(indices),2,256,256)) for n in ('canonical','mean8','variance8')};maxinput=0.;maxrepeat=0.;inversechecks=0;start=time.time()
with torch.inference_mode():
 for lo in range(0,len(indices),32):
  ii=indices[lo:lo+32];rr=[rows[i] for i in ii];h=torch.from_numpy(np.array(H[[int(r['hm_row']) for r in rr]],copy=True)).cuda();tr=torch.tensor([int(r['tx_row']) for r in rr],device='cuda');tc=torch.tensor([int(r['tx_col']) for r in rr],device='cuda');preds=[]
  if lo==0:
   for mode in ('raw','geometry'):maxinput=max(maxinput,float((make_inputs(h,tr,tc,mode)-old.make_inputs(h,tr,tc,mode)).abs().max()))
  for flip in (False,True):
   for k in range(4):
    hh=torch.rot90(h,k,(-2,-1));r=tr.clone();c=tc.clone()
    for _ in range(k):r,c=255-c,r
    if flip:hh=hh.flip(-1);c=255-c
    x=make_inputs(hh,r,c,'geometry')
    assert torch.all(x[torch.arange(len(h),device='cuda'),1,r,c]==1)
    hhback=hh.flip(-1) if flip else hh;assert torch.equal(torch.rot90(hhback,-k,(-2,-1)),h);inversechecks+=1
    with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x)*core.SCALE+core.OFFSET
    if lo==0 and not flip and k==0:
     with torch.autocast('cuda',dtype=torch.bfloat16):p2=model(x)*core.SCALE+core.OFFSET
     maxrepeat=float((p-p2).abs().max())
    if flip:p=p.flip(-1)
    p=torch.rot90(p,-k,(-2,-1));preds.append(p.float())
  pp=torch.stack(preds);mean=pp.mean(0);var=(pp-mean).square().mean(0)
  for n,v in [('canonical',pp[0]),('mean8',mean),('variance8',var)]:outs[n][lo:lo+len(ii)]=v.cpu().numpy()
  if (lo+len(ii))%256==0:print(json.dumps({'phase':'symmetry','maps':lo+len(ii),'total':len(indices),'seconds':time.time()-start}),flush=True)
for x in outs.values():x.flush()
assert hashlib.sha256(cp.read_bytes()).hexdigest()==sha
out={'training':False,'optimizer_created':False,'backward_called':False,'targets_or_masks_read':False,'device':device,'checkpoint_sha256':sha,'checkpoint_epoch':ck['epoch'],'maps':len(indices),'orientations':8,'max_old_new_input_difference':maxinput,'max_same_process_forward_repeat_difference':maxrepeat,'height_transform_inverse_checks':inversechecks,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seconds':time.time()-start,'limits':'D4 invariance is an ideal isotropic-geometry symmetry; finite simulator sampling may not be exactly equivariant. This test measures prediction inconsistency, not label noise.'}
(O/'inference.json').write_text(json.dumps(out,indent=2)+'\n');(O/'indices.json').write_text(json.dumps(idx,indent=2)+'\n');print(json.dumps({'done':True,**out}),flush=True)
