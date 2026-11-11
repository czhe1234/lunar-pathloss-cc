"""H/TX-only candidate joint path-delay / relative-azimuth energy."""
import sys,ctypes,csv,json,time,argparse,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c69_registered import RegisteredTransport
from src.c67_physics import directions
import numpy as np,torch
A=R/'audit/c124_spectrum';SRC=R/'audit/c101_residual'
class Spectrum:
 def __init__(self,n=8192):
  self.engine=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=1)
  self.dirs=directions(n);self.lib=ctypes.CDLL(str(R/'src/libc124_spectrum.so'))
  P=ctypes.c_void_p;I=ctypes.c_int;F=ctypes.c_float
  self.lib.path_spectrum.argtypes=[P,P,P,P,I,P,P,I,F,F,F,F,F,I,I,P,P];self.lib.path_spectrum.restype=I
  lookup=np.load(R/'src/c70_centroid_lookup0.npz')
  self.ids=torch.tensor(lookup['ids'],device='cuda',dtype=torch.long)
  self.weights=torch.tensor(lookup['weights'],device='cuda',dtype=torch.float32)
 @torch.no_grad()
 def raw(self,h,row,col,faces,validate=False):
  e=self.engine
  direct,spec,unused,ht,x,y,z,em=e.raw(h,row,col,return_emission=True)
  geometry,normals=e.scene(ht)
  ids=torch.as_tensor(faces,device='cuda',dtype=torch.int32).contiguous()
  out=torch.empty((len(ids),2,8,16),device='cuda',dtype=torch.float32)
  ptr=lambda t:t.data_ptr()
  err=self.lib.path_spectrum(ptr(geometry),ptr(normals),ptr(e.material),ptr(self.dirs),len(self.dirs),ptr(em),ptr(ids),len(ids),x,y,z,e.spacing,float(np.max(h)),e.anti,e.smooth,ptr(out),torch.cuda.current_stream().cuda_stream)
  if err:raise RuntimeError(f'path_spectrum launch: {err}')
  ctr=e.xy[e.faces[ids.long()]].mean(1)
  alt=ht.reshape(-1)[e.faces[ids.long()]].mean(1)+1
  d2=((ctr[:,0]-x*e.spacing).square()+(ctr[:,1]-y*e.spacing).square()+(alt-z).square()).clamp_min(.0625)
  if validate:return out,d2
  return out*d2[:,None,None,None]
 @torch.no_grad()
 def features(self,h,row,col,pixels):
  p=torch.as_tensor(pixels,device='cuda',dtype=torch.long)
  faces,inverse=torch.unique(self.ids[p].reshape(-1),return_inverse=True)
  native=self.raw(h,row,col,faces)
  # Same fixed three-donor receiver interpolation, applied to linear power.
  values=(native[inverse].reshape(len(p),3,2,8,16)*self.weights[p,:,None,None,None]).sum(1)
  totals=values.sum((2,3))
  encoded=torch.log10(values.clamp_min(1e-38))/4
  total_encoded=torch.log10(totals.clamp_min(1e-38))/4
  return encoded.flatten(1),total_encoded
def check(a,g):
 rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));heights=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
 ids=json.load(open(SRC/'fit/done.json'))['ids'][:3]
 test=Spectrum();ref=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=8192)
 rng=np.random.default_rng(124041);faces=rng.choice(130050,2048,replace=False).astype('int32')
 reports=[]
 for i in ids:
  r=rows[i];h=heights[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col'])
  t=time.monotonic();hist,d2=test.raw(h,tr,tc,faces,True)
  direct,spec,diff,*_=ref.raw(h,tr,tc);s=hist.sum((2,3));truth=diff[torch.as_tensor(faces,device='cuda',dtype=torch.long),:2]
  rel=((s-truth).abs()/truth.abs().clamp_min(1e-35));maximum=rel.max().item()
  assert maximum<1e-4,(i,maximum)
  encoded,total=test.features(h,tr,tc,np.arange(0,65536,32,dtype='int32'))
  assert torch.isfinite(encoded).all() and torch.isfinite(total).all()
  reports.append(dict(map=i,max_relative_sum_error=maximum,seconds=time.monotonic()-t,nonzero_bin_fraction=float((hist>0).float().mean())))
 atomic_json(A/'check.json',dict(status='passed',GPU=g,reports=reports,labels_read=False,mask_read=False))
 print(reports,flush=True)
def build(a,g):
 part=a.part;src=SRC/part;d=json.load(open(src/'done.json'));maps=d['ids'];ppm=d['points_per_map']
 out=A/part/f'part{a.shard}of{a.of}';out.mkdir(parents=True,exist_ok=False)
 positions=list(range(len(maps)))[a.shard::a.of]
 atomic_json(out/'indices.json',positions)
 spec=np.lib.format.open_memmap(out/'spectrum.npy','w+',dtype=np.float16,shape=(len(positions)*ppm,256))
 total=np.lib.format.open_memmap(out/'total.npy','w+',dtype=np.float32,shape=(len(positions)*ppm,2))
 px=np.load(src/'pixel.npy',mmap_mode='r');mi=np.load(src/'mapid.npy',mmap_mode='r')
 rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));heights=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
 engine=Spectrum();start=time.monotonic()
 for j,k in enumerate(positions):
  i=maps[k];sl=slice(k*ppm,(k+1)*ppm);assert np.all(mi[sl]==i)
  r=rows[i];h=heights[int(r['hm_row'])]
  x,t=engine.features(h,int(r['tx_row']),int(r['tx_col']),np.asarray(px[sl]))
  xn=x.half().cpu().numpy();tn=t.cpu().numpy();assert np.isfinite(xn).all() and np.isfinite(tn).all()
  spec[j*ppm:(j+1)*ppm]=xn;total[j*ppm:(j+1)*ppm]=tn
  if j%64==0 or j+1==len(positions):
   p=dict(maps=j+1,total=len(positions),seconds=time.monotonic()-start);atomic_json(out/'progress.json',p);print(p,flush=True)
 spec.flush();total.flush()
 atomic_json(out/'done.json',dict(status='complete',maps=len(positions),points=len(positions)*ppm,seconds=time.monotonic()-start,GPU=g,labels_read=False,mask_read=False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--part',choices=['fit','monitor']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--of',type=int,default=1);a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(3)
 if a.check:check(a,g)
 else:build(a,g)
if __name__=='__main__':main()
