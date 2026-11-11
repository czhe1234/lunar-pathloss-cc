"""C135 native continuous mirror chains and coherency-preserving mixed paths."""
import sys,ctypes,time,json,csv,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
import numpy as np,torch
from src.c69_registered import RegisteredTransport
from src.c67_physics import quadrature,directions
from src.c57_transmission import slab
A=R/'audit/c135_paths'
class Paths:
 def __init__(self,source=4,receiver=1,directions_count=64,pre=6,post=3,maxsteps=128):
  self.rt=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=source,diffuse=directions_count)
  self.rt.weights=self.rt.weights.clamp_min(0);self.rt.weights/=self.rt.weights.sum(1,keepdim=True)
  self.bary=quadrature(receiver);self.pre=pre;self.post=post;self.maxsteps=maxsteps
  c=torch.linspace(0,1,4097,device='cuda')
  _,_,rs,rp=slab(c,torch.tensor([415e6,5.775e9],device='cuda'),torch.tensor([.0003562,.009324],device='cuda'))
  self.refl=torch.stack((rs.real,rs.imag,rp.real,rp.imag),-1).permute(1,0,2).contiguous()
  self.lib=ctypes.CDLL(str(R/'src/libc135_paths.so'))
  P=ctypes.c_void_p;I=ctypes.c_int;F=ctypes.c_float
  self.lib.c135_forward.argtypes=[P,P,P,P,P,I,I,I,F,F,F,F,P,P,P];self.lib.c135_forward.restype=I
  self.lib.c135_query.argtypes=[P,P,P,P,P,P,I,P,I,P,I,I,I,F,P,P,P];self.lib.c135_query.restype=I
  self.lib.c135_direct.argtypes=[P,P,P,P,I,F,F,F,F,P,P,P];self.lib.c135_direct.restype=I
  self.lib.c135_reference_spec.argtypes=[P,P,P,P,I,F,F,F,F,P,P];self.lib.c135_reference_spec.restype=I
 @torch.no_grad()
 def direct(self,s,h,row,col,pixels):
  out=torch.empty(len(s['g']),4,device='cuda');tmp=torch.empty_like(out)
  er=self.lib.c135_direct(s['g'].data_ptr(),s['n'].data_ptr(),self.rt.material.data_ptr(),self.rt.primary.data_ptr(),len(self.rt.primary),
   col*255/256,(row+1)*255/256,self.rt.source_height(h,row,col),s['top'],out.data_ptr(),tmp.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  pix=torch.as_tensor(np.array(pixels),dtype=torch.long,device='cuda')
  return self.rt.interpolate(out)[pix]
 @torch.no_grad()
 def forward(self,h,row,col):
  ht=torch.as_tensor(np.array(h),dtype=torch.float32,device='cuda');g,n=self.rt.scene(ht);nf=len(g)
  spec=torch.zeros(nf,self.pre,2,device='cuda');em=torch.zeros(self.pre+1,nf,2,2,4,device='cuda')
  er=self.lib.c135_forward(g.data_ptr(),n.data_ptr(),self.rt.material.data_ptr(),self.refl.data_ptr(),
   self.rt.spec.data_ptr(),len(self.rt.spec),self.pre,self.maxsteps,col*255/256,(row+1)*255/256,
   self.rt.source_height(h,row,col),float(np.max(h)),spec.data_ptr(),em.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  # Keep the exact existing single-reflection branch in both arms. Extra paths
  # use the Jones chain; this avoids confounding compiler-roundoff cell crossings.
  ref=torch.zeros(nf,3,device='cuda')
  er=self.lib.c135_reference_spec(g.data_ptr(),n.data_ptr(),self.rt.material.data_ptr(),self.rt.spec.data_ptr(),len(self.rt.spec),
   col*255/256,(row+1)*255/256,self.rt.source_height(h,row,col),float(np.max(h)),ref.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  spec[:,0]=ref[:,:2]
  return dict(g=g,n=n,spec=spec,em=em,top=float(np.max(h)))
 @torch.no_grad()
 def query(self,s,pixels):
  pix=torch.as_tensor(np.array(pixels),device='cuda',dtype=torch.long)
  ids=self.rt.ids[pix].int().contiguous();w=self.rt.weights[pix].contiguous()
  out=torch.empty(len(pix),8,device='cuda')
  er=self.lib.c135_query(s['g'].data_ptr(),s['n'].data_ptr(),self.rt.material.data_ptr(),self.refl.data_ptr(),
   ids.data_ptr(),w.data_ptr(),len(pix),self.bary.data_ptr(),len(self.bary),self.rt.dirs.data_ptr(),len(self.rt.dirs),
   self.pre,self.post,s['top'],s['em'].data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  spec=(s['spec'][ids.long()]*w[:,:,None,None]).sum(1)
  return torch.cat((spec,out.reshape(-1,4,2)),1)
def smoke():
 gpu=gpu_guard();torch.set_num_threads(3);engine=Paths(source=4,receiver=1,directions_count=64,pre=6,post=3)
 rng=np.random.default_rng(135041);px=rng.choice(65536,2048,replace=False)
 stats={'GPU':gpu,'geometry':'frozen C69 candidate','labels_read':False}
 flat=np.zeros((256,256),np.float32)
 s=engine.forward(flat,128,128);q=engine.query(s,px)
 d,oldspec,diff,*_,em=engine.rt.raw(flat,128,128,return_emission=True)
 spec_error=(s['spec'][:,0]-oldspec[:,:2]).abs().sum(0)/oldspec[:,:2].sum(0)
 newinitial=s['em'][0,:,:,:,:2].sum((1,3))
 em_error=(newinitial-em[:,:2]).abs().sum(0)/em[:,:2].sum(0)
 olddiff=engine.rt.interpolate(diff[:,:2])[px]
 diff_error=(q[:,6]-olddiff).abs().sum(0)/olddiff.sum(0)
 stats.update(flat_one_spec_relative_L1=spec_error.tolist(),flat_initial_diffuse_relative_L1=em_error.tolist(),
  flat_zero_post_relative_L1=diff_error.tolist(),flat_multispec_max=float(s['spec'][:,1:].max()),
  flat_mixed_max=float(q[:,7:].max()))
 assert max(stats['flat_one_spec_relative_L1'])<1e-4,stats
 assert max(stats['flat_initial_diffuse_relative_L1'])<1e-5,stats
 assert max(stats['flat_zero_post_relative_L1'])<1e-5,stats
 assert stats['flat_multispec_max']==0 and stats['flat_mixed_max']==0,stats
 rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r')
 ids=json.loads((R/'audit/c134_paired/protocol.json').read_text())['fit'][:3]
 stats['maps']=[]
 for i in ids:
  r=rows[i];h=hm[int(r['hm_row'])]
  torch.cuda.synchronize();t=time.monotonic();s=engine.forward(h,int(r['tx_row']),int(r['tx_col']));q=engine.query(s,px);torch.cuda.synchronize()
  assert torch.isfinite(q).all() and torch.all(q>=0)
  d,sp,di,*_,em=engine.rt.raw(h,int(r['tx_row']),int(r['tx_col']),return_emission=True)
  engine.maxsteps=20;reference=engine.forward(h,int(r['tx_row']),int(r['tx_col']));engine.maxsteps=128
  x=(reference['spec'][:,0]-sp[:,:2]).abs().sum(0)/sp[:,:2].sum(0).clamp_min(1e-20)
  e=(q[:,6]-engine.rt.interpolate(di[:,:2])[px]).abs().sum(0)/engine.rt.interpolate(di[:,:2])[px].sum(0).clamp_min(1e-20)
  # Area-averaged coherencies must remain positive semidefinite.
  ec=s['em'];det=ec[...,0]*ec[...,1]-ec[...,2].square()-ec[...,3].square()
  ratio=det/(ec[...,0]+ec[...,1]).square().clamp_min(1e-35)
  item=dict(map=i,seconds=time.monotonic()-t,spec_reference_20_steps_relative_L1=x.tolist(),spec_128_vs_20_relative_change=((s['spec'][:,0]-reference['spec'][:,0]).sum(0)/reference['spec'][:,0].sum(0)).tolist(),initial_diffuse_relative_L1=e.tolist(),
   power_by_component=q.sum(0).tolist(),coherency_min_relative_det=float(ratio.min()))
  stats['maps'].append(item)
  assert max(x.tolist())<2e-4 and max(e.tolist())<2e-5,item
  assert ratio.min()>-2e-6,item
 stats['status']='passed';atomic_json(A/'smoke.json',stats);print(json.dumps(stats),flush=True)
if __name__=='__main__':smoke()
