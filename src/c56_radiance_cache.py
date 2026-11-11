"""Build a compact input-only multibounce radiosity bank for all released TRAIN inputs."""
import sys,time,json,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c56_radiosity import Geometry
import numpy as np,torch
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.55)
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
O=R/'features/c56_radiosity';O.mkdir(exist_ok=False)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');m=np.load(R/'features/c49_visibility/train_margins.npy',mmap_mode='r')
margins=torch.empty(m.shape,device='cuda',dtype=torch.float16)
for lo in range(0,len(m),128):margins[lo:lo+128].copy_(torch.from_numpy(np.array(m[lo:lo+128])))
groups={}
for i,row in enumerate(rows):groups.setdefault(int(row['hm_row']),[]).append(i)
geo=Geometry(side=32,neighbors=128,samples=128);out=np.lib.format.open_memmap(O/'train_logpower.npy',mode='w+',dtype=np.float16,shape=(len(rows),32,32,10))
t=time.time();capped=0;totalnodes=0;iterations=[];relative=[];maxrow=0.;completed=0
# Analytic flat-surface test: all diffuse surface-to-surface coupling must vanish.
flat=torch.zeros((1,256,256),device='cuda');K0,_,_,_=geo.build(flat);assert int(torch.count_nonzero(K0))==0
for lo in range(0,len(H),8):
 hs=list(range(lo,min(lo+8,len(H))));h=torch.tensor(np.array(H[hs]),device='cuda',dtype=torch.float32)
 K,pos,norm,raw=geo.build(h)
 ids=[groups[v] for v in hs];width=max(map(len,ids))
 padded=np.array([v+[v[0]]*(width-len(v)) for v in ids],np.int64);ri=torch.as_tensor(padded,device='cuda')
 tr=torch.tensor([[int(rows[i]['tx_row']) for i in line] for line in padded],device='cuda')
 tc=torch.tensor([[int(rows[i]['tx_col']) for i in line] for line in padded],device='cuda')
 logpower,convergence=geo.radiance(h,K,pos,norm,tr,tc,margins[ri])
 value=logpower.half().cpu().numpy()
 for b,line in enumerate(ids):out[line]=value[b,:len(line)].reshape(len(line),32,32,10)
 completed+=sum(map(len,ids));iterations.append(convergence['iterations']);relative.append(convergence['maximum_last_relative_update'])
 capped+=int((raw>.95).sum());totalnodes+=raw.numel();maxrow=max(maxrow,float(raw.max()))
 if lo%128==0 or lo+8>=len(H):
  p={'terrains':lo+len(hs),'total_terrains':len(H),'maps':completed,'total_maps':len(rows),'seconds':time.time()-t,
    'latest_iterations':convergence['iterations'],'latest_relative_update':convergence['maximum_last_relative_update']}
  atomic_json(O/'progress.json',p);print(json.dumps(p),flush=True)
out.flush()
meta={'complete':True,'input_only':True,'PL_and_validity_masks_read':False,'gpu':gpu,'shape':list(out.shape),'seconds':time.time()-t,
 'terrain_graph':'32x32nodes;128nearestneighbors;128ray samples peredge; Lambertian positive K withrowcap.95',
 'power':'B=rho*(E+K*B), rho.2/.5/.8/.95, TXheight3 and empirical raster-phase candidate; store log10B8channels+log10E2channels',
 'flat_surface_K_exactly_zero':True,'rows_capped_fraction':capped/totalnodes,'maximum_uncapped_row_sum':maxrow,
 'maximum_iterations':max(iterations),'maximum_final_relative_update':max(relative),
 'limits':'Coarse candidate surface-power operator. Not actual simulator mesh, radio material or full specular/refraction physics.'}
atomic_json(O/'meta.json',meta);print(json.dumps(meta),flush=True)
