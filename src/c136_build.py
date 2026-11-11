"""Cache explicit native source-to-receiver couplings, never targets/masks."""
import sys,time,json,csv,ctypes,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c135_paths import Paths
from src.c67_physics import directions
import numpy as np,torch
A=R/'audit/c136_native'
class Native(Paths):
 def __init__(self):
  super().__init__(source=8,receiver=2,directions_count=64,pre=6,post=3);self.adirs=directions(128);self.nl=256
  self.link=ctypes.CDLL(str(R/'src/libc136_links.so'));P=ctypes.c_void_p;I=ctypes.c_int;F=ctypes.c_float
  self.link.c136_spec.argtypes=[P,P,P,P,P,I,I,F,F,F,F,P,I,P,P,P,P];self.link.c136_spec.restype=I
  self.link.c136_a.argtypes=[P,P,P,P,P,I,P,I,P,I,I,I,P,P,P,I,F,I,P,P,P,P];self.link.c136_a.restype=I
 @torch.no_grad()
 def links(self,h,tr,tc,pixels,seed):
  s=self.forward(h,tr,tc);pix=torch.as_tensor(pixels,device='cuda',dtype=torch.long)
  ii=self.rt.ids[pix];wt=self.rt.weights[pix];uf,inv=torch.unique(ii,return_inverse=True);ids=uf.int()
  lookup=torch.full((len(s['g']),),-1,device='cuda',dtype=torch.int32);lookup[uf]=torch.arange(len(uf),device='cuda',dtype=torch.int32)
  cap=6000000;count=torch.zeros(1,device='cuda',dtype=torch.int32);si=torch.empty(cap,2,device='cuda',dtype=torch.int32);sw=torch.empty(cap,2,device='cuda')
  er=self.link.c136_spec(s['g'].data_ptr(),s['n'].data_ptr(),self.rt.material.data_ptr(),self.refl.data_ptr(),self.rt.spec.data_ptr(),len(self.rt.spec),self.pre,
   tc*255/256,(tr+1)*255/256,self.rt.source_height(h,tr,tc),s['top'],lookup.data_ptr(),cap,count.data_ptr(),si.data_ptr(),sw.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  ns=int(count.item());assert ns<=cap,(ns,cap);si=si[:ns].clone();sw=sw[:ns].clone()
  # Q contains polarized irradiance from direct illumination and specular prefixes.
  coh=s['em'].sum(0).contiguous();field=coh[...,:2].sum(-1)
  area=s['g'][:,3];mass=field.sum(1)*area[:,None];p=.46*(mass/mass.sum(0).clamp_min(1e-27)).sum(1)+.08*area/area.sum()
  p=(p/p.sum()).contiguous();cdf=p.double().cumsum(0).float().contiguous();cdf[-1]=1
  cap=len(uf)*len(self.bary)*(len(self.adirs)*(1+self.post)+self.nl);count.zero_()
  ai=torch.empty(cap,2,device='cuda',dtype=torch.int32);aw=torch.empty(cap,2,device='cuda')
  er=self.link.c136_a(s['g'].data_ptr(),s['n'].data_ptr(),self.rt.material.data_ptr(),self.refl.data_ptr(),ids.data_ptr(),len(ids),
   self.bary.data_ptr(),len(self.bary),self.adirs.data_ptr(),len(self.adirs),self.nl,self.post,coh.data_ptr(),p.data_ptr(),cdf.data_ptr(),seed,s['top'],
   cap,count.data_ptr(),ai.data_ptr(),aw.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  na=int(count.item());assert na<=cap,(na,cap);ai=ai[:na].clone();aw=aw[:na].clone()
  direct=torch.empty(len(s['g']),4,device='cuda');tmp=torch.empty_like(direct)
  er=self.lib.c135_direct(s['g'].data_ptr(),s['n'].data_ptr(),self.rt.material.data_ptr(),self.rt.primary.data_ptr(),len(self.rt.primary),
   tc*255/256,(tr+1)*255/256,self.rt.source_height(h,tr,tc),s['top'],direct.data_ptr(),tmp.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if er:raise RuntimeError(er)
  # Normalize by each receiver facet's neutral power to avoid small-power gradients.
  den=direct[uf,:2].double().clone();den.index_add_(0,si[:,0].long(),sw.double());den.index_add_(0,ai[:,0].long(),aw.double())
  safe=torch.where(den>0,den,torch.ones_like(den))
  sw=(sw.double()/safe[si[:,0].long()]).float();aw=(aw.double()/safe[ai[:,0].long()]).float();dw=(direct[uf,:2].double()/safe).float()
  pw=wt[:,:,None].double()*den[inv];pixden=pw.sum(1);pw=(pw/torch.where(pixden[:,None]>0,pixden[:,None],torch.ones_like(pixden[:,None]))).float()
  # Geometry and power-state covariates at native facets.
  ht=torch.as_tensor(np.array(h),device='cuda');xyz=torch.cat((self.rt.xy,ht.reshape(-1,1)),1)[self.rt.faces].mean(1)
  tx=torch.tensor([tc,(tr+1),self.rt.source_height(h,tr,tc)],device='cuda')
  delta=xyz-tx;dist=delta.norm(dim=1).clamp_min(.25);normal=torch.nn.functional.normalize(torch.cat((-s['g'][:,1:3]/self.rt.spacing,torch.ones(len(xyz),1,device='cuda')),1),dim=1)
  feat=torch.cat((delta/256,dist.log()[:,None]/6,normal,torch.log10(field.reshape(-1,4).clamp_min(1e-30))/10,(-delta/dist[:,None]*normal).sum(1)[:,None]),1)
  assert feat.shape[1]==12
  neutral=dw.double().clone();neutral.index_add_(0,si[:,0].long(),sw.double());neutral.index_add_(0,ai[:,0].long(),aw.double())
  ratio=(neutral[inv]*pw).sum(1);active=pixden>0
  assert (ratio[active]-1).abs().max()<1e-4
  df=feat[uf].clone();rxdelta=delta[uf].clone();rxdelta[:,2]+=1;rd=rxdelta.norm(dim=1).clamp_min(.25)
  df[:,:3]=rxdelta/256;df[:,3]=rd.log()/6;df[:,7:9]=torch.log10(direct[uf,:2].clamp_min(1e-30))/10
  df[:,9]=direct[uf,2];df[:,10]=torch.log1p(direct[uf,3])/4;df[:,11]=(-rxdelta/rd[:,None]*normal[uf]).sum(1)
  data=dict(df=df.half(),si=si,sw=sw,ai=ai,aw=aw,dw=dw,uf=uf.int(),inv=inv.int(),pw=pw,feat=feat.half(),active=active,txxy=torch.tensor([tc/128-1,(tr+1)/128-1],device='cuda'))
  for k,v in data.items():assert torch.isfinite(v).all(),k
  return {k:v.cpu().numpy() for k,v in data.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=3);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 gpu=gpu_guard();torch.set_num_threads(3);engine=Native();plan=json.loads((A/'protocol.json').read_text())
 rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r');todo=[];lookup={}
 for part in ['fit','monitor']:
  root=R/'audit/c101_residual'/part;meta=json.loads((root/'done.json').read_text());pp=meta['points_per_map'];px=np.load(root/'pixel.npy',mmap_mode='r');idx={i:j for j,i in enumerate(meta['ids'])}
  for i in plan[part]:lookup[i]=np.array(px[idx[i]*pp:(idx[i]+1)*pp]);todo.append(i)
 todo=todo[a.shard::a.of];todo=todo[:1] if a.smoke else todo
 out=A/'cache';out.mkdir(parents=True,exist_ok=True);times=[];start=time.monotonic()
 for j,i in enumerate(todo):
  r=rows[i];t=time.monotonic();val=engine.links(hm[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']),lookup[i],seed=136079+i)
  np.savez(out/f'{i}.npz',pixels=lookup[i],**val);times.append(time.monotonic()-t)
  if j%24==0:
   status=dict(maps=j+1,total=len(todo),seconds=time.monotonic()-start,median=float(np.median(times)),last=i,spec_links=len(val['si']),diffuse_links=len(val['ai']))
   atomic_json(A/f'build{a.shard}_progress.json',status);print(status,flush=True)
 done=dict(status='complete',GPU=gpu,maps=len(todo),seconds=time.monotonic()-start,median_seconds=float(np.median(times)),labels_read=False,mask_read=False)
 atomic_json(A/('smoke.json' if a.smoke else f'build{a.shard}_done.json'),done);print(done,flush=True)
if __name__=='__main__':main()
