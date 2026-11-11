"""Released H/TX-only feature production; labels/masks are never opened here."""
import sys,json,csv,argparse,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c58_exact_transmission import evaluate as exact,trace
from src.c57_transmission import slab
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--part',choices=['fit','dev'],required=True);a=p.parse_args()
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.45)
O=R/'features/c58_transmission';O.mkdir(exist_ok=True)
H=torch.tensor(np.array(np.load(R.parent/'data/train_hm.npy',mmap_mode='r')),device='cuda').flatten()
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
proto=json.loads((R/'audit/c43_architecture_20261102/protocol.json').read_text())
mapids=np.load(R/('features/c38_dense/fit_map_indices.npy' if a.part=='fit' else 'features/c6_points/dev_map_indices.npy'))
per=1024 if a.part=='fit' else 256;selected=proto['fit64_ids'] if a.part=='fit' else proto['dev64_ids'];lookup={int(v):k for k,v in enumerate(mapids)}
indices=np.concatenate([np.arange(lookup[i]*per,(lookup[i]+1)*per) for i in selected]);hm=np.repeat([int(rows[i]['hm_row']) for i in selected],per)
X=np.load(R/('features/c38_dense/fit_x.npy' if a.part=='fit' else 'features/c10_points/dev_x.npy'),mmap_mode='r')
dest=O/(a.part+'_loss.npy');assert not dest.exists()
loss=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(indices),2,2,2))
counts=np.lib.format.open_memmap(O/(a.part+'_counts.npy'),mode='w+',dtype=np.int16,shape=(len(indices),2,2))

# Independent analytic checks of the single-layer formula.
angles=torch.linspace(.05,1,50,device='cuda');freq=torch.tensor([415e6,5.775e9],device='cuda')
ts,tp,rs,rp=slab(angles,freq,torch.zeros(2,device='cuda'))
energy=max(float((ts.abs().square()+rs.abs().square()-1).abs().max()),float((tp.abs().square()+rp.abs().square()-1).abs().max()))
assert energy<3e-5,energy
ts,tp,_,_=slab(angles,freq,torch.tensor([.0003562,.009324],device='cuda'),thickness=0.)
assert max(float((ts-1).abs().max()),float((tp-1).abs().max()))<2e-6
# Flat surface and a ridge with analytically known entry/exit positions.
debugH=torch.zeros((1,256,256),device='cuda');xx=torch.arange(256,device='cuda')
debugH[0]=torch.clamp(10-(xx-128).abs(),min=0)[None]
debugX=torch.zeros((1,294),device='cuda');debugX[:,32]=100/256;debugX[:,33]=100/256;debugX[:,5]=50/128
rr,_,keep,_,cnt=trace(debugX,torch.zeros(1,device='cuda',dtype=torch.long),debugH.flatten(),False,0)
expected=torch.tensor([(125/1.04-100)/50,(131/.96-100)/50],device='cuda')
assert int(cnt)==2 and float((rr[0,:2]-expected).abs().max())<2e-5
flat=torch.zeros_like(debugH)
_,_,_,_,cnt=trace(debugX,torch.zeros(1,device='cuda',dtype=torch.long),flat.flatten(),False,0);assert int(cnt)==0
t=time.time()
for lo in range(0,len(indices),2048):
 x=torch.tensor(np.array(X[indices[lo:lo+2048]]),device='cuda')
 value,c=exact(x,torch.tensor(hm[lo:lo+2048],device='cuda'),H)
 loss[lo:lo+len(x)]=value.cpu().numpy();counts[lo:lo+len(x)]=c.cpu().numpy()
for z in [loss,counts]:z.flush()
np.save(O/(a.part+'_point_indices.npy'),indices)
meta={'complete':True,'part':a.part,'map_indices':selected,'rows':len(indices),'input_only':True,'PL_or_validity_masks_read':False,
 'max_crossings':32,'over32_crossing_fraction':float((counts>32).mean()),'lossless_energy_check_max_error':energy,
 'geometry':'Exactgrid/triangle DDA intersections, two diagonal candidates; analytic ridge/flat checks passed','material':'Releasedmetadata:eps3.411,thickness10,sigma.0003562/.009324','gpu':gpu,'seconds':time.time()-t}
atomic_json(O/(a.part+'_meta.json'),meta);print(json.dumps({k:meta[k] for k in meta if k!='map_indices'}),flush=True)
