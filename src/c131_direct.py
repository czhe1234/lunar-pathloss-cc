"""C131 feasibility gate: fixed nonnegative direct power implies a dB ceiling.
All model input is H/TX only; labels and masks are read after the physics call for scoring.
No learned model or oracle score is reported as deployable performance.
"""
import sys,json,time,argparse,ctypes,csv,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c69_registered import RegisteredTransport
from src.c67_physics import quadrature
import numpy as np
import torch
A=R/'audit/c131_transport'
class Direct:
 def __init__(self,n):
  self.rt=RegisteredTransport(physical=1,anti=0,smooth=1,direct=1,spec=1,diffuse=1)
  weights=self.rt.weights
  wmin=float(weights.min());werr=float((weights.sum(1)-1).abs().max())
  assert wmin>=-1e-12 and werr<2e-6,(wmin,werr)
  positive=weights.clamp_min(0);self.rt.weights=positive/positive.sum(1,keepdim=True)
  self.mapping_roundoff=dict(min_original_weight=wmin,max_sum_error=werr,max_adjustment=float((self.rt.weights-weights).abs().max()))
  self.q=quadrature(n);self.n=n
  self.lib=ctypes.CDLL(str(R/'src/libc131_direct.so'))
  self.fn=self.lib.c131_direct
  P=ctypes.c_void_p;I=ctypes.c_int;F=ctypes.c_float
  self.fn.argtypes=[P,P,P,P,I,F,F,F,F,F,P,P];self.fn.restype=I
 def faces(self,h,r,c):
  rt=self.rt;ht=torch.as_tensor(np.array(h),device='cuda',dtype=torch.float32)
  g,n=rt.scene(ht);out=torch.empty((130050,4),device='cuda')
  err=self.fn(g.data_ptr(),n.data_ptr(),rt.material.data_ptr(),self.q.data_ptr(),len(self.q),
   c*255/256,(r+1)*255/256,rt.source_height(h,r,c),rt.spacing,float(h.max()),
   out.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if err:raise RuntimeError(err)
  return out
 def pixels(self,h,r,c):
  f=self.faces(h,r,c)
  return self.rt.interpolate(f).T.reshape(4,256,256)
 def check(self):
  flat=np.zeros((256,256),np.float32);r,c=104,111
  got=self.faces(flat,r,c).cpu().numpy()
  v=self.rt.xy[self.rt.faces].cpu().double().numpy()
  q=self.q.cpu().double().numpy()
  p=np.einsum('qj,fjd->fqd',q,v)
  tx=np.array([c,r+1])
  ref=(1/np.maximum(.0625,((p-tx)**2).sum(2)+4)).mean(1)
  rel=np.max(abs(got[:,:3]-ref[:,None])/np.maximum(ref[:,None],1e-30))
  assert rel<5e-5,rel
  assert np.all(got[:,3]==1)
  ridge=np.broadcast_to(np.maximum(0,16-np.abs(np.arange(256)-128)),(256,256)).astype('float32')
  z=self.faces(ridge,100,100).cpu().numpy()
  assert np.isfinite(z).all() and np.all(z>=0)
  assert np.all(z[:,0]+1e-12>=z[:,2]) and np.all(z[:,1]+1e-12>=z[:,2])
  assert np.any(z[:,3]==0) and np.any(z[:,3]==1)
  return dict(flat_max_relative_error=float(rel),ridge_full_block_fraction=float(np.mean(z[:,3]==0)),
              quadrature_points=len(self.q),positive_direct_and_transmission=True,mapping_roundoff=self.mapping_roundoff)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,required=True);ap.add_argument('--check',action='store_true')
 ap.add_argument('--limit',type=int,default=0);ap.add_argument('--part',default='monitor',choices=['monitor','fit'])
 a=ap.parse_args();gpu=gpu_guard();torch.set_num_threads(3)
 plan=json.loads((A/'protocol.json').read_text());ids=plan[a.part]
 if a.limit:ids=ids[:a.limit]
 out=A/f'direct_n{a.n}_{a.part}';out.mkdir(exist_ok=True)
 model=Direct(a.n);check=model.check();atomic_json(out/'check.json',check)
 if a.check:print(json.dumps(check),flush=True);return
 rows=list(csv.DictReader((DATA/'train_index.csv').open()));heights=np.load(DATA/'train_hm.npy',mmap_mode='r')
 target=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')]
 mask=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
 bm=json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids']
 bi={int(i):j for j,i in enumerate(bm)};base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r')
 # Cache geometry-only power. Three channels: direct including transmission in each band, plus LoS inverse squared range.
 arr=np.lib.format.open_memmap(out/'power.npy','w+',dtype='float32',shape=(len(ids),4,256,256))
 freq=np.array([415e6,5.775e9],dtype='float64');scale=(299792458/(4*np.pi*freq))**2
 counts=[];floors=[];viol=[];c82ss=[];tailss=[];tails=[];clearss=[];clearn=[];coincident=[]
 ts=time.monotonic();times=[]
 for k,i in enumerate(ids):
  row=rows[i];h=heights[int(row['hm_row'])];tr,tc=int(row['tx_row']),int(row['tx_col'])
  torch.cuda.synchronize();start=time.monotonic()
  x=model.pixels(h,tr,tc).cpu().numpy()
  times.append(time.monotonic()-start);arr[k]=x
  assert np.isfinite(x).all() and np.all(x>=0),(i,float(np.nanmin(x)))
  y=np.stack([v[i] for v in target]).astype('float64')
  valid=np.stack([np.unpackbits(v[i],bitorder='big').reshape(256,256) for v in mask]).astype(bool)
  p=np.array(base[bi[i]],dtype='float64')
  assert p.shape==y.shape
  los=x[2].astype('float64')[None]*scale[:,None,None]
  thru=x[:2].astype('float64')*scale[:,None,None]
  # Zero direct power imposes no finite upper bound on path loss.
  ceilings=np.stack([-10*np.log10(np.maximum(z,1e-300)) for z in (los,thru)])
  err=np.maximum(y[None]-ceilings,0)*valid[None]
  n=valid.sum((1,2));counts.append(n)
  floors.append((err**2).sum((2,3)));viol.append(((err>1)&valid[None]).sum((2,3)))
  error=(p-y)**2*valid;c82ss.append(error.sum((1,2)))
  tail=valid&(abs(p-y)>=5);tails.append(tail.sum((1,2)));tailss.append((error*tail).sum((1,2)))
  # Geometry class only, computed without mask or target.
  clear=x[3]>.999;clearn.append((clear[None]&valid).sum((1,2)))
  clearss.append((error*clear).sum((1,2)))
  coincident.append((error[None]*(err>1)).sum((2,3)))
  if k%32==0 or k+1==len(ids):
   r=dict(maps=k+1,total=len(ids),seconds=time.monotonic()-ts,
    conditional_floor_rmse=np.sqrt(np.sum(floors,0)/np.sum(counts,0)).tolist(),
    C82=np.sqrt(np.sum(c82ss,0)/np.sum(counts,0)).tolist())
   atomic_json(out/'progress.json',r);print(json.dumps(r),flush=True)
 arr.flush()
 np.savez(out/'stats.npz',ids=ids,count=counts,floor_sse=floors,violation_over1_count=viol,
  baseline_sse=c82ss,tail_sse=tailss,tail_count=tails,clear_sse=clearss,clear_count=clearn,overlap_baseline_sse=coincident)
 result=dict(status='complete',part=a.part,maps=len(ids),terrains=len(set(rows[i]['terrain_id'] for i in ids)),
  quadrature_n=a.n,quadrature_points=len(model.q),GPU=gpu,check=check,seconds=time.monotonic()-ts,
  physics_seconds_median=float(np.median(times)),physics_seconds_max=float(np.max(times)),
  condition='Fixed candidate geometry and fixed direct power; arbitrary nonnegative additional power; labels used only for counterfactual scoring.',
  count=np.sum(counts,0).tolist(),C82=np.sqrt(np.sum(c82ss,0)/np.sum(counts,0)).tolist(),
  floor_rmse_los_then_through=np.sqrt(np.sum(floors,0)/np.sum(counts,0)).tolist(),
  violation_over1_fraction=(np.sum(viol,0)/np.sum(counts,0)).tolist(),
  C82_SSE_overlap_fraction=(np.sum(coincident,0)/np.sum(c82ss,0)).tolist(),
  source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'src/c131_direct.cu',R/'src/c67_transport.cu']},
  official_VAL_used=False,trained=False)
 atomic_json(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
