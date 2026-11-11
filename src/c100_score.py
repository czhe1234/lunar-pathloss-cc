"""Freeze three native-context controls, then score same complete and256point heldout maps."""
import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c100_native_attention import NativeField
from src.c99_data import Reader
A=R/'audit/c100_native_attention'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def rmse(ss,cc):return np.sqrt(np.asarray(ss).sum(0)/np.asarray(cc).sum(0)).tolist()
@torch.inference_mode()
def main():
 g=gpu_guard();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('high')
 plan=json.loads((A/'protocol.json').read_text());choices={};nets={}
 for mode in ['continue','local','global']:
  d=R/'runs'/('c100_'+mode);r=json.loads((d/'result.json').read_text());assert r['status']=='complete'
  cfg=json.loads((d/'config.json').read_text());assert cfg['initial_state_sha256']==plan['initial_state_sha256']
  ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False)
  assert ck['epoch']==r['best_epoch']
  choices[mode]=dict(best_epoch=r['best_epoch'],monitor_rmse=r['monitor']['rmse'],sha256=sha(d/'best.pt'))
  net=NativeField(mode).cuda().to(memory_format=torch.channels_last);net.load_state_dict(ck['state']);net.eval();nets[mode]=net
 reference=json.loads((R/'runs/c100_global/C82_reference.json').read_text())['rmse']
 monitor_mean={m:float(np.mean(c['monitor_rmse'])) for m,c in choices.items()};monitor_mean['C82']=float(np.mean(reference))
 eligible=min(monitor_mean,key=monitor_mean.get)
 frozen=dict(choices=choices,eligible=eligible,monitor_mean=monitor_mean,rule=plan['candidate_selection'])
 p=A/'selection_frozen.json'
 if p.exists():assert json.loads(p.read_text())==frozen
 else:atomic_json(p,frozen)
 reader=Reader();ids=sorted(plan['confirmation'],key=lambda i:(str(reader.lookup[i][0].filename),reader.lookup[i][1]));n=len(ids);names=['C82','continue','local','global']
 predictions={m:np.lib.format.open_memmap(A/(m+'_confirmation.npy'),mode='w+',dtype='float32',shape=(n,2,256,256)) for m in ['continue','local','global']}
 full=np.zeros((4,n,2),np.float64);point=full.copy();counts=np.zeros((n,2),np.int64);point_counts=counts.copy()
 dev_ids=np.load(R/'features/c6_points/dev_map_indices.npy');devpix=np.load(R/'features/c6_points/dev_pixel.npy',mmap_mode='r').reshape(len(dev_ids),256)
 where={int(i):j for j,i in enumerate(dev_ids)};t=time.monotonic()
 for start in range(0,n,8):
  ix=ids[start:start+8];sl=slice(start,start+len(ix))
  x,p0,y,w=reader.tensors(ix)
  pp=[torch.from_numpy(np.stack([reader.c82[reader.c82index[i]] for i in ix])).cuda().float()]
  for mode in ['continue','local','global']:
   with torch.autocast('cuda',torch.bfloat16):v=p0+nets[mode](x).float()
   pp.append(v);predictions[mode][sl]=v.cpu().numpy()
  pixels=torch.from_numpy(np.stack([devpix[where[i]] for i in ix]).astype(np.int64)).cuda()
  wp=w.flatten(2).gather(2,pixels[:,None].expand(-1,2,-1))
  counts[sl]=w.sum((2,3)).cpu().numpy();point_counts[sl]=wp.sum(2).cpu().numpy()
  for k,v in enumerate(pp):
   err=v-y;full[k,sl]=torch.where(w,err.square(),0.).sum((2,3),dtype=torch.float64).cpu().numpy()
   ep=err.flatten(2).gather(2,pixels[:,None].expand(-1,2,-1))
   point[k,sl]=torch.where(wp,ep.square(),0.).sum(2,dtype=torch.float64).cpu().numpy()
  if (start//8)%32==0:print('confirm',start+len(ix),n,'seconds',round(time.monotonic()-t,1),flush=True)
 for p in predictions.values():p.flush()
 primary={m:rmse(full[k],counts) for k,m in enumerate(names)};sampled={m:rmse(point[k],point_counts) for k,m in enumerate(names)}
 expected=json.loads((R/'audit/c88_mesh/base_index.json').read_text())['confirm_rmse']
 assert np.max(abs(np.array(primary['C82'])-expected))<1e-5,(primary['C82'],expected)
 old=json.loads((R/'audit/c96_information/summary.json').read_text())['metrics']['C82']['confirmation']
 assert np.max(abs(np.array(sampled['C82'])-old['rmse']))<2e-6
 assert point_counts.sum(0).tolist()==old['count']
 terrain=[reader.rows[i]['terrain_id'] for i in ids];unique=sorted(set(terrain));lookup={t:j for j,t in enumerate(unique)}
 tt=np.array([lookup[t] for t in terrain]);ss=np.zeros((4,len(unique),2));cc=np.zeros((len(unique),2))
 for b in [0,1]:
  cc[:,b]=np.bincount(tt,weights=counts[:,b],minlength=len(unique))
  for k in range(4):ss[k,:,b]=np.bincount(tt,weights=full[k,:,b],minlength=len(unique))
 rng=np.random.default_rng(100097);draw=rng.integers(0,len(unique),(2000,len(unique)));den=cc[draw].sum(1);ci={}
 for left,right in [('C82','continue'),('C82','local'),('C82','global'),('continue','local'),('continue','global'),('local','global')]:
  a=ss[names.index(left)][draw].sum(1);b=ss[names.index(right)][draw].sum(1)
  gains=np.sqrt(a/den)-np.sqrt(b/den);ci[left+'-minus-'+right]=np.quantile(gains,[.025,.975],axis=0).T.tolist()
 gain={m:(np.array(primary['C82'])-np.array(primary[m])).tolist() for m in ['continue','local','global']}
 promote=eligible!='C82' and bool((np.array(gain[eligible])>=.30).all() and (np.array(ci['C82-minus-'+eligible])[:,0]>0).all())
 np.savez(A/'confirmation_sse.npz',names=np.array(names),mapid=np.array(ids),terrain=np.array(terrain),full_sse=full,count=counts,point_sse=point,point_count=point_counts,terrain_sse=ss,terrain_count=cc)
 # Independent float64 scoring from saved float32 fields and original labels. No second model inference.
 check=np.zeros((3,2),np.float64);checkcount=np.zeros(2,np.int64)
 for start in range(0,n,16):
  ix=ids[start:start+16];sl=slice(start,start+len(ix))
  yy=np.stack([ar[ix] for ar in reader.y],1).astype(np.float64)
  ww=np.stack([np.unpackbits(ar[ix],axis=1).reshape(-1,256,256) for ar in reader.w],1).astype(bool)
  checkcount+=ww.sum((0,2,3))
  for k,mode in enumerate(['continue','local','global']):
   e=np.asarray(predictions[mode][sl],dtype=np.float64)-yy
   check[k]+=np.where(ww,e*e,0.).sum((0,2,3))
 independent=np.sqrt(check/checkcount)
 diff=float(np.max(abs(independent-np.array([primary[m] for m in ['continue','local','global']]))));assert diff<2e-6,diff
 summary=dict(primary_fullmap_RMSE=primary,old256pixel_RMSE=sampled,gain_vs_C82=gain,terrain_paired_95ci=ci,
   eligible_selected_by_monitor=eligible,promote=promote,promotion_rule=plan['promotion'],confirmation_terrains=len(unique),confirmation_maps=n,
   valid_pixels=counts.sum(0).tolist(),point_valid_pixels=point_counts.sum(0).tolist(),GPU=g,seconds=time.monotonic()-t,
   independent_float64_rmse=independent.tolist(),independent_max_difference_db=diff,
   confirmation_selection_frozen=True,official_val_used=False,website_submission=False)
 atomic_json(A/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
