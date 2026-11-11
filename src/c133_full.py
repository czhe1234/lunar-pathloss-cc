"""Full 128-terrain native-grid evaluation; no checkpoint selection here."""
import sys,json,csv,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c132_geometry import Geometry
from src.c133_model import ContextTransport as PowerTransport
from src.c133_geometry import Positions
import torch,numpy as np
A=R/'audit/c133_context'
def main():
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');plan=json.loads((A/'protocol.json').read_text());ids=plan['monitor'];rows=list(csv.DictReader((DATA/'train_index.csv').open()))
 hm=np.load(DATA/'train_hm.npy',mmap_mode='r');ys=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];vs=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
 base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r');bi={i:j for j,i in enumerate(json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids'])}
 modes=['direction','pooled','local'];nets={};outputs={};stats={m:[] for m in modes+['C82']};info={}
 for m in modes:
  folder=R/'runs'/f'c133_{m}';result=json.loads((folder/'result.json').read_text());name='best.pt' if result['best_epoch']>0 else 'last.pt'
  ck=torch.load(folder/name,map_location='cpu',weights_only=False);net=PowerTransport('direction' if m=='local' else m,ck['rounds']).cuda().eval();net.load_state_dict(ck['state']);nets[m]=net
  dest=folder/'full_monitor';dest.mkdir(exist_ok=False);atomic_json(dest/'ids.json',ids)
  outputs[m]=np.lib.format.open_memmap(dest/'prediction.npy','w+',dtype=np.float32,shape=(len(ids),2,256,256))
  info[m]=dict(checkpoint=name,epoch=ck['epoch'],reason='If no trained checkpoint beat C82, evaluate last trained checkpoint rather than relabel untouched C82 as a new model.')
 engine=Geometry();positions=Positions(engine);t=time.monotonic();last=None;times=[]
 with torch.inference_mode():
  for j,i in enumerate(ids):
   row=rows[i];h=hm[int(row['hm_row'])];terrain=row['terrain_id']
   if last!=terrain:
    scene=engine.scene(h);ar=np.load(R/'audit/c132_transport/cache'/terrain/'graph.npz');g={k:torch.tensor(ar[k],device='cuda')[None] for k in ['src','dst','k','kf','area']};last=terrain
   torch.cuda.synchronize();st=time.monotonic()
   raw=engine.query(h,int(row['tx_row']),int(row['tx_col']),np.arange(65536),scene)
   q={k:(v.half() if k in ['sf','qf','af','df'] else v)[None] for k,v in raw.items()}
   hitxy,rxxy=positions(scene,np.arange(65536));q['hitxy']=hitxy[None];q['rxxy']=rxxy[None];q['h']=torch.tensor(np.array(h),device='cuda')[None];q['inverse']=torch.zeros(1,device='cuda',dtype=torch.long);q['txxy']=torch.tensor([[2*int(row['tx_col'])/256-1,2*(int(row['tx_row'])+1)/256-1]],device='cuda')
   b=np.array(base[bi[i]],dtype='float32');bt=torch.tensor(b.reshape(2,-1).T,device='cuda')[None]
   y=np.stack([a[i] for a in ys]).astype('float64');v=np.stack([np.unpackbits(a[i]).reshape(256,256) for a in vs]).astype(bool);tail=v&(abs(y-b)>=5)
   def add(m,p):
    e=np.where(v,p-y,0);z=np.where(tail,e*e,0)
    stats[m].append(np.stack([(e*e).sum((1,2)),v.sum((1,2)),z.sum((1,2)),tail.sum((1,2))],1))
   add('C82',b)
   for m,net in nets.items():
    q['g0']=net.physical(g,q,neutral=True)
    p=net(g,q,bt)[0].T.reshape(2,256,256).cpu().numpy();assert np.isfinite(p).all();outputs[m][j]=p;add(m,p)
   torch.cuda.synchronize();times.append(time.monotonic()-st)
   if j%32==0 or j+1==len(ids):
    record=dict(maps=j+1,total=len(ids),seconds=time.monotonic()-t,scores={m:np.sqrt(np.sum(stats[m],0)[:,0]/np.sum(stats[m],0)[:,1]).tolist() for m in stats});atomic_json(A/'full_progress.json',record);print('FULL',json.dumps(record),flush=True)
 for x in outputs.values():x.flush()
 result=dict(status='complete',scope='original128TRAINmonitor;all628fullmaps;not officialVAL',GPU=gpu,checkpoint_info=info,seconds=time.monotonic()-t,three_models_plus_geometry_median_seconds=float(np.median(times)),results={})
 for m,stat in stats.items():
  z=np.sum(stat,0);result['results'][m]=dict(RMSE=np.sqrt(z[:,0]/z[:,1]).tolist(),SSE=z[:,0].tolist(),count=z[:,1].tolist(),tail_RMSE=np.sqrt(z[:,2]/z[:,3]).tolist());np.save(A/f'full_{m}_stats.npy',np.stack(stat))
 for m in modes:result['results'][m]['gain_vs_C82']=(np.array(result['results']['C82']['RMSE'])-result['results'][m]['RMSE']).tolist()
 atomic_json(A/'full_result.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
