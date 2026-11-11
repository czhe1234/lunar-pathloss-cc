"""Illuminated-facet MIS stabilizes the first/last diffuse connection; based on CC C92."""
import ctypes,sys,json,csv,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c135_paths import Paths,A
from src.c67_physics import directions,quadrature
import torch,numpy as np
class MIS(Paths):
 def __init__(self,source=4,receiver=1,light=1024,angular=128):
  super().__init__(source=source,receiver=receiver,directions_count=64)
  self.nl=light;self.mdirs=directions(angular)
  self.mlib=ctypes.CDLL(str(R/'src/libc135_mis.so'));P=ctypes.c_void_p;I=ctypes.c_int;F=ctypes.c_float
  self.fn=self.mlib.c135_mis;self.fn.argtypes=[P,P,P,P,I,P,I,P,I,I,P,P,P,I,F,P,P];self.fn.restype=I
 @torch.no_grad()
 def gather(self,s,field,ids,seed):
  area=s['g'][:,3];p=(field[:,:2]+field[:,2:])*area[:,None];total=p.sum(0)
  prob=.46*(p/total.clamp_min(1e-27)).sum(1)+.08*area/area.sum();prob=(prob/prob.sum()).contiguous();cdf=prob.double().cumsum(0).float().contiguous();cdf[-1]=1
  out=torch.empty(len(ids),4,device='cuda')
  er=self.fn(s['g'].data_ptr(),s['n'].data_ptr(),self.rt.material.data_ptr(),ids.data_ptr(),len(ids),self.bary.data_ptr(),len(self.bary),
   self.mdirs.data_ptr(),len(self.mdirs),self.nl,field.data_ptr(),prob.data_ptr(),cdf.data_ptr(),seed,s['top'],out.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  return out[:,:2]
 @torch.no_grad()
 def query(self,s,pixels,seed=135079):
  q=super().query(s,pixels);pix=torch.as_tensor(np.array(pixels),device='cuda',dtype=torch.long)
  ix=self.rt.ids[pix];w=self.rt.weights[pix];uf,inv=torch.unique(ix,return_inverse=True);ids=uf.int()
  for index,em in [(6,s['em'][0]),(7,s['em'][1:].sum(0))]:
   field=em[...,:2].sum(-1).reshape(-1,4).contiguous()
   rx=self.gather(s,field,ids,seed)
   q[:,index]=(rx[inv]*w[:,:,None]).sum(1)
  return q
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=int,default=4);p.add_argument('--receiver',type=int,default=1)
 p.add_argument('--light',type=int,default=1024);p.add_argument('--angular',type=int,default=128)
 p.add_argument('--tag',default='mis');p.add_argument('--check-count',type=int,default=8);p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=1);a=p.parse_args()
 gpu=gpu_guard();torch.set_num_threads(3);engine=MIS(source=a.source,receiver=a.receiver,light=a.light,angular=a.angular)
 plan=json.loads((A/'protocol.json').read_text());rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r')
 todo=[];lookup={}
 for part in ['fit','monitor']:
  root=R/'audit/c101_residual'/part;meta=json.loads((root/'done.json').read_text());pp=meta['points_per_map'];px=np.load(root/'pixel.npy',mmap_mode='r');idx={i:j for j,i in enumerate(meta['ids'])}
  ids=plan['fit_probe' if part=='fit' else 'monitor'];ids=ids[:a.check_count] if a.check_count else ids
  for i in ids:lookup[i]=np.array(px[idx[i]*pp:(idx[i]+1)*pp]);todo.append(i)
 out=A/a.tag;out.mkdir(exist_ok=True);times=[];start=time.monotonic()
 for j,i in enumerate(todo[a.shard::a.of]):
  r=rows[i];h=hm[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);px=lookup[i];t=time.monotonic()
  s=engine.forward(h,tr,tc);q=engine.query(s,px,seed=135079+i);d=engine.direct(s,h,tr,tc,px);power=torch.cat((d[:,:2,None].transpose(1,2),q),1)
  assert torch.isfinite(power).all() and (power>=0).all(),i
  np.savez(out/f'{i}.npz',pixels=px,power=power.cpu().numpy(),direct_geometry=d[:,2:].cpu().numpy());times.append(time.monotonic()-t)
  if j%40==0:print(dict(maps=j+1,seconds=time.monotonic()-start,last=i,median=float(np.median(times))),flush=True)
 done=dict(GPU=gpu,maps=len(times),seconds=time.monotonic()-start,median_seconds=float(np.median(times)),parameters=vars(a),labels_read=False)
 atomic_json(A/f'{a.tag}_{a.shard}_done.json',done);print(done,flush=True)
if __name__=='__main__':main()
