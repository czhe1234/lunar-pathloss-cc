"""Frozen input-only diagnostic forward passes. No label/mask reads or gradients."""
import sys,json,csv,time,hashlib,fcntl,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,gpu_guard,atomic_json,check_source_unchanged
import numpy as np,torch
OUT=ROOT/'audit/c4_remaining_error_20261031';plan=json.loads((OUT/'selection.json').read_text());run=ROOT/'runs/c4_full_refit_s20261031';package=ROOT/'deliveries/c4_full_refit_s20261031'
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=gpu_guard();check_source_unchanged();cksha=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest();assert cksha==plan['frozen_checkpoint_sha256']
manifest=json.loads((package/'manifest.json').read_text())
for name in ['weights.pt','c3_geometry.py','c4_geometry.py','c4_model.py','inference.py']:assert hashlib.sha256((package/name).read_bytes()).hexdigest()==manifest['files'][name]
sys.path.insert(0,str(package));from inference import Predictor
from c3_geometry import geometry
from c4_geometry import extra_geometry,make_inputs
p=Predictor(package);assert not p.model.training
rr,cc=np.meshgrid(np.arange(4,256,8),np.arange(4,256,8),indexing='ij');rr=rr.ravel();cc=cc.ravel();gr=torch.tensor(rr,device='cuda');gc=torch.tensor(cc,device='cuda');t0=time.time()
with torch.inference_mode():
 assert not torch.is_grad_enabled()
 for split,label in [('train','matched_train'),('val','official_val')]:
  folder=OUT/label;folder.mkdir(exist_ok=False);rows=list(csv.DictReader((DATA/f'{split}_index.csv').open()));ids=plan[split+'_indices'];hm=np.load(DATA/f'{split}_hm.npy',mmap_mode='r');fields=np.lib.format.open_memmap(folder/'state_fields.npy',mode='w+',dtype=np.float16,shape=(len(ids),3,256,256));samples=np.lib.format.open_memmap(folder/'geometry_samples.npy',mode='w+',dtype=np.float16,shape=(len(ids),20,len(rr)))
  arrays=[np.lib.format.open_memmap(folder/f'pl{b}.npy',mode='w+',dtype=np.float32,shape=(len(ids),256,256)) for b in ['415','58']] if split=='train' else None
  atomic_json(folder/'indices.json',{'source_split':split,'index_rows':ids,'sample_ids':[rows[i]['sample_id'] for i in ids],'receiver_rows':rr.tolist(),'receiver_cols':cc.tolist(),'state_field_names':['E3_m','E2_m','signed_receiver_incidence_cosine'],'geometry_source':'frozen exported c4 source; 16 original normalized channels + 4 critical-height/length channels'})
  for lo in range(0,len(ids),8):
   ii=ids[lo:lo+8];rs=[rows[i] for i in ii];h=torch.from_numpy(np.array(hm[[int(r['hm_row']) for r in rs]],copy=True)).cuda();r=torch.tensor([int(a['tx_row']) for a in rs],device='cuda');c=torch.tensor([int(a['tx_col']) for a in rs],device='cuda');g=torch.cat((geometry(h,r,c),extra_geometry(h,r,c)),1);e3=(2*torch.sinh(g[:,:8].float()*3)).amax(1);e2=2*torch.sinh(g[:,8].float()*3);fields[lo:lo+len(ii)]=torch.stack((e3,e2,g[:,11].float()),1).half().cpu().numpy();samples[lo:lo+len(ii)]=g[:,:,gr,gc].cpu().numpy()
   if arrays is not None:
    x,fs=make_inputs(h,r,c,g)
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=p.model(x)+fs
    pr=pred.float().cpu().numpy();assert np.isfinite(pr).all()
    for k,a in enumerate(arrays):a[lo:lo+len(ii)]=pr[:,k]
   if lo%400==0:atomic_json(OUT/'probe_progress.json',{'stage':label,'done':lo+len(ii),'total':len(ids),'seconds':time.time()-t0});print(json.dumps({'stage':label,'done':lo+len(ii),'total':len(ids)}),flush=True)
  fields.flush();samples.flush()
  if arrays:
   for a in arrays:a.flush()
  if split=='val':
   subset=plan['precision_val_indices'];fp=np.lib.format.open_memmap(folder/'fp32_predictions.npy',mode='w+',dtype=np.float32,shape=(len(subset),2,256,256));bf=np.lib.format.open_memmap(folder/'bf16_replay.npy',mode='w+',dtype=np.float32,shape=(len(subset),2,256,256))
   for lo in range(0,len(subset),8):
    ii=subset[lo:lo+8];rs=[rows[i] for i in ii];h=torch.from_numpy(np.array(hm[[int(r['hm_row']) for r in rs]],copy=True)).cuda();r=torch.tensor([int(a['tx_row']) for a in rs],device='cuda');c=torch.tensor([int(a['tx_col']) for a in rs],device='cuda');x,fs=make_inputs(h,r,c)
    with torch.autocast('cuda',dtype=torch.bfloat16):bpred=p.model(x)+fs
    fpred=p.model(x)+fs;fp[lo:lo+len(ii)]=fpred.float().cpu().numpy();bf[lo:lo+len(ii)]=bpred.float().cpu().numpy()
   fp.flush();bf.flush()
assert hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()==cksha
atomic_json(OUT/'probe_progress.json',{'stage':'completed','seconds':time.time()-t0,'device':device,'checkpoint_sha256_before_and_after':cksha,'no_optimizer_no_backward_no_parameter_updates':True,'inputs_read':['train_hm.npy','train_index.csv','val_hm.npy','val_index.csv'],'no_target_or_mask_read':True});print('frozen probe complete',flush=True)
