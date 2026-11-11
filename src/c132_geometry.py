"""Shared H-only surface graph; exact H/TX native first and last leg samples."""
import sys,ctypes,json,csv,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c69_registered import RegisteredTransport
from src.c67_physics import directions
import numpy as np,torch
A=R/'audit/c132_transport'
class Geometry:
 def __init__(self):
  self.rt=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=2,diffuse=1)
  ww=self.rt.weights.clamp_min(0);self.rt.weights=ww/ww.sum(1,keepdim=True)
  self.kdirs=directions(32);self.rdirs=directions(16)
  self.lib=ctypes.CDLL(str(R/'src/libc132_transport.so'));P=ctypes.c_void_p;I=ctypes.c_int;F=ctypes.c_float
  self.lib.c132_graph.argtypes=[P,P,P,P,I,F,P,P,P,P,P,P];self.lib.c132_graph.restype=I
  self.lib.c132_query.argtypes=[P,P,P,P,I,F,F,F,F,P,P,P,I,P,P,P,P,P,P,P,P,P];self.lib.c132_query.restype=I
 @torch.no_grad()
 def scene(self,h):
  ht=torch.as_tensor(np.array(h),device='cuda');g,n=self.rt.scene(ht)
  src=torch.empty(1024*32,device='cuda',dtype=torch.int32);dst=torch.empty_like(src)
  w=torch.empty(1024*32,2,device='cuda');f=torch.empty(1024*32,12,device='cuda');area=torch.empty(1024,device='cuda')
  er=self.lib.c132_graph(g.data_ptr(),n.data_ptr(),self.rt.material.data_ptr(),self.kdirs.data_ptr(),32,float(h.max()),src.data_ptr(),dst.data_ptr(),w.data_ptr(),f.data_ptr(),area.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  # Source power is total patch power, not irradiance. Bound total outgoing transport by .85.
  mass=torch.zeros(2048,2,device='cuda').index_add_(0,src.long(),w)
  w=w/torch.maximum(torch.ones_like(mass[src.long()]),mass[src.long()]/.85)
  return dict(g=g,n=n,src=src,dst=dst,k=w,kf=f,area=area,top=float(h.max()))
 @torch.no_grad()
 def query(self,h,tr,tc,pixels,scene=None):
  s=self.scene(h) if scene is None else scene
  pix=torch.as_tensor(np.array(pixels,copy=True),device='cuda',dtype=torch.long);count=len(pix);ids=self.rt.ids[pix].int().contiguous();weights=self.rt.weights[pix].contiguous()
  sq=torch.empty(2048,2,device='cuda');sf=torch.empty(1024,12,device='cuda')
  ri=torch.empty(count,48,device='cuda',dtype=torch.int32);aw=torch.empty(count,48,device='cuda')
  rq=torch.empty(count,48,2,device='cuda');qf=torch.empty(count,48,12,device='cuda');af=torch.empty_like(qf);df=torch.empty(count,3,12,device='cuda')
  err=self.lib.c132_query(s['g'].data_ptr(),s['n'].data_ptr(),self.rt.material.data_ptr(),self.rdirs.data_ptr(),16,
   tc*255/256,(tr+1)*255/256,self.rt.source_height(h,tr,tc),s['top'],s['area'].data_ptr(),ids.data_ptr(),weights.data_ptr(),count,
   sq.data_ptr(),sf.data_ptr(),ri.data_ptr(),aw.data_ptr(),rq.data_ptr(),qf.data_ptr(),af.data_ptr(),df.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if err:raise RuntimeError(err)
  # Native continuous specular rays and area-sampled direct branch; no directional binning.
  direct,spec,*_=self.rt.raw(h,tr,tc)
  ds=self.rt.interpolate(torch.cat([direct[:,:2],spec[:,:2]],1))[pix]
  df=(df*weights[:,:,None]).sum(1)
  return dict(sq=sq,sf=sf,ri=ri,aw=aw,rq=rq,qf=qf,af=af,df=df,ds=ds)
def arrays(d):return {k:v.cpu().numpy().astype(np.float16 if k in ['kf','sf','qf','af','df'] else np.int32 if k in ['src','dst','ri'] else np.float32) for k,v in d.items() if k not in ['g','n','top']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--of',type=int,default=3);ap.add_argument('--smoke',action='store_true')
 a=ap.parse_args();gpu=gpu_guard();torch.set_num_threads(3);engine=Geometry()
 plan=json.loads((A/'protocol.json').read_text());rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r')
 maps=plan['fit']+plan['monitor'];groups={}
 for i in maps:groups.setdefault(rows[i]['terrain_id'],[]).append(i)
 todo=sorted(groups)[a.shard::a.of]
 if a.smoke:todo=todo[:1]
 lookup={}
 for part in ['fit','monitor']:
  src=R/'audit/c101_residual'/part;done=json.loads((src/'done.json').read_text());pp=done['points_per_map'];px=np.load(src/'pixel.npy',mmap_mode='r')
  for j,i in enumerate(done['ids']):lookup[i]=np.asarray(px[j*pp:(j+1)*pp])
 start=time.monotonic();times=[];completed=0
 for terrain in todo:
  ids=groups[terrain];row=rows[ids[0]];h=hm[int(row['hm_row'])];t=time.monotonic();s=engine.scene(h)
  out=A/'cache'/terrain;out.mkdir(parents=True,exist_ok=True)
  np.savez(out/'graph.npz',**arrays(s))
  for i in ids:
   row=rows[i];q=engine.query(h,int(row['tx_row']),int(row['tx_col']),lookup[i],s)
   for k,v in q.items():assert torch.isfinite(v).all(),(i,k)
   for k in ['sq','rq','aw','ds']:assert torch.all(q[k]>=0),(i,k)
   np.savez(out/f'{i}.npz',pixels=lookup[i],**arrays(q));completed+=1
  torch.cuda.synchronize();times.append(time.monotonic()-t)
  if completed%32<len(ids) or a.smoke:
   p=dict(terrains=len(times),total=len(todo),maps=completed,seconds=time.monotonic()-start,last_seconds=times[-1]);atomic_json(A/f'build_{a.shard}_progress.json',p);print(p,flush=True)
  if a.smoke:
   outq=q;mass=torch.zeros(2048,2,device='cuda').index_add_(0,s['src'].long(),s['k'])
   check=dict(status='passed',GPU=gpu,max_column_power_fraction=float(mass.max()),geometry_only_K=True,query_points=len(lookup[i]),terrain=terrain,maps=len(ids),seconds=time.monotonic()-start,
    nonzero_graph_fraction=float((s['k']>0).float().mean()),positive_RX_fraction=float((outq['rq']>0).float().mean()),labels_read=False,masks_read=False)
   assert check['max_column_power_fraction']<=.85001
   atomic_json(A/'geometry_smoke.json',check);print(check,flush=True);return
 atomic_json(A/f'build_{a.shard}_done.json',dict(status='complete',GPU=gpu,terrains=len(todo),maps=completed,seconds=time.monotonic()-start))
if __name__=='__main__':main()
