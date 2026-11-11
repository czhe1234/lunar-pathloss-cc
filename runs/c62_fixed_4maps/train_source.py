"""Matched fixed1024 vs dense61440 pixels; 4096 excluded holdout pixels per FIT map."""
import sys,time,json,math,hashlib,csv,argparse,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c47_train import load_teacher,device_array,tiny_selection
from src.c61_model import Model
p=argparse.ArgumentParser();p.add_argument('--coverage',choices=['fixed','dense'],required=True);p.add_argument('--name',required=True);a=p.parse_args()
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.6);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
d=R/'features/c62_dense4';meta=json.loads((d/'meta.json').read_text());assert meta['complete'] and not meta['labels_masks_read']
data={k:device_array(d/(k+'.npy')) for k in ['x','fspl','base','frame']};ids=meta['map_indices'];sel,expected,terrains=tiny_selection();assert list(map(int,expected))==ids
pix=np.load(R/'features/c38_dense/fit_pixel.npy',mmap_mode='r');fixed=np.array(pix[sel],np.int64).reshape(4,1024)
hold=[];dense=[]
for j in range(4):
 available=np.setdiff1d(np.arange(65536),fixed[j]);hp=np.random.default_rng(20261209+j).choice(available,4096,replace=False);hold.append(hp);dense.append(np.setdiff1d(np.arange(65536),hp))
fixed=torch.tensor(fixed,device='cuda');hold=torch.tensor(np.array(hold),device='cuda');dense=torch.tensor(np.array(dense),device='cuda');pool=fixed if a.coverage=='fixed' else dense
Y=[];M=[]
for b in ['415','58']:
 y=np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r');m=np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r')
 Y.append(np.array(y[ids],np.float32).reshape(4,65536));M.append(np.unpackbits(np.array(m[ids]),axis=1,bitorder='big'))
data['y']=torch.tensor(np.stack(Y,-1),device='cuda');data['mask']=torch.tensor(np.stack(M,-1),device='cuda',dtype=torch.float32)
s,c=load_teacher();warm=torch.load(R/'runs/c52_adapter_full/best.pt',map_location='cpu',weights_only=False)['model'];torch.manual_seed(20261208);net=Model(s,'path',warm).cuda().to(memory_format=torch.channels_last)
run=R/'runs'/a.name;run.mkdir(exist_ok=False);shutil.copy2(__file__,run/'train_source.py');shutil.copy2(R/'src/c61_model.py',run/'model_source.py')
cfg={**vars(a),'variant':'path','scope':'fourFITterrains_pixelcoverage_control','gpu':gpu,'steps':1000,'seed':20261208,'holdout_seed':20261209,'pixels_per_map':int(pool.shape[1]),'holdout_pixels_per_map':4096,'batch_maps':4,'points_per_map':256,'official_VAL_used':False,'selection':'fixed1000updates; holdout never selects checkpoints'}
h=hashlib.sha256()
for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
cfg['initial_state_sha256']=h.hexdigest();atomic_json(run/'config.json',cfg)
np.savez(run/'pixel_split.npz',fixed=fixed.cpu().numpy(),holdout=hold.cpu().numpy(),dense=dense.cpu().numpy())
@torch.inference_mode()
def evaluate(pixels):
 net.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss);bs=torch.zeros_like(ss);mi=torch.arange(4,device='cuda')[:,None]
 with torch.autocast('cuda',dtype=torch.bfloat16):st=net.encode(data['frame'])
 for lo in range(0,pixels.shape[1],256):
  pp=pixels[:,lo:lo+256];x=data['x'][mi,pp]
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net.decode(st,x,pp)
  base=data['base'][mi,pp]*20+data['fspl'][mi,pp];out=base+q.float()*20;y=data['y'][mi,pp];m=data['mask'][mi,pp]
  e=out.double()-y.double();be=base.double()-y.double();ss+=(e.square()*m).sum((0,1));bs+=(be.square()*m).sum((0,1));cc+=m.sum((0,1)).double()
 net.train();return {'RMSE':(ss/cc).sqrt().cpu().tolist(),'C38_RMSE':(bs/cc).sqrt().cpu().tolist(),'SSE':ss.cpu().tolist(),'count':cc.cpu().tolist()}
probe=pool[:,np.linspace(0,pool.shape[1]-1,1024,dtype=np.int64)]
initial=evaluate(hold);history=[];t0=time.time();opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.001,weight_decay=1e-4);gen=torch.Generator(device='cuda').manual_seed(20261208)
for step in range(1,1001):
 mi=torch.randint(4,(4,),device='cuda',generator=gen);pi=torch.randint(pool.shape[1],(4,256),device='cuda',generator=gen);pp=pool[mi[:,None],pi];jj=mi[:,None]
 lr=.001*min(1,step/50)*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/1000)))
 for g in opt.param_groups:g['lr']=lr
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):q=net(data['frame'][mi],data['x'][jj,pp],pp)
 yy=(data['y'][jj,pp]-data['fspl'][jj,pp])/20-data['base'][jj,pp];mm=data['mask'][jj,pp];loss=(((q.float()-yy).square()*mm).sum((0,1))/mm.sum((0,1)).clamp_min(1)).mean()
 assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
 if step%200==0:
  rec={'step':step,'FIT_probe':evaluate(probe),'holdout4096':evaluate(hold),'seconds':time.time()-t0};history.append(rec);atomic_json(run/'progress.json',rec);atomic_json(run/'history.json',history);print(json.dumps(rec),flush=True)
torch.save({'model':net.state_dict(),'step':1000,'config':cfg},run/'best.pt')
out={'status':'complete','scope':cfg['scope'],'coverage':a.coverage,'initial_holdout':initial,'FIT_probe':evaluate(probe),'holdout4096':evaluate(hold),'all_pixels':evaluate(torch.arange(65536,device='cuda')[None].expand(4,-1)),'seconds':time.time()-t0,'official_VAL_used':False,'limit':'Only known FIT terrains; pixel holdout is not unseen-terrain validation. Fixed budget, one seed.'}
atomic_json(run/'result.json',out);print(json.dumps(out),flush=True)
