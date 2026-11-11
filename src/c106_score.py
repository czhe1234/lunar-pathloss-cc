"""Frozen16-map candidates on same internal panels; localization and power scored independently."""
import argparse,sys,json,hashlib,time,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c105_context import create,reader,ResidualPanel
from src.c106_metrics import metrics
A=R/'audit/c106_fit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['monitor','confirm'],default='confirm');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 frozen=json.loads((A/'selection_frozen.json').read_text());point=R/'audit/c101_residual'/a.part
 meta=json.loads((point/'done.json').read_text());ids=meta['ids'];pp=meta['points_per_map'];n=len(ids)*pp
 raw=np.load(point/'target.npy');y=raw[:,:2];mask=raw[:,2:4].astype(bool);base=np.load(point/'base.npy');pixels=np.load(point/'pixel.npy').reshape(len(ids),pp)
 rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));terrain=[rows[i]['terrain_id'] for i in ids]
 data=ResidualPanel(reader(),ids,'context');pixel=torch.tensor(pixels,dtype=torch.int64,device='cuda')
 arrays={'C82':base};t=time.monotonic()
 models=frozen['candidates']
 if a.part=='confirm':
  path=R/'audit/c105_context/context_confirm_prediction.npy';assert path.exists();arrays['C105']=np.load(path)
 for name,choice in models.items():
  path=R/choice['checkpoint'];assert sha(path)==choice['sha256']
  ck=torch.load(path,map_location='cpu',weights_only=False)
  net=create().cuda().to(memory_format=torch.channels_last);net.load_state_dict(ck['state']);net.eval()
  p=np.empty((n,2),np.float32)
  for start in range(0,len(ids),16):
   stop=min(len(ids),start+16);sl=slice(start,stop);x,b,yy,ww=data.batch(sl)
   with torch.autocast('cuda',torch.bfloat16):pred=b+net(x).float()
   def take(v):return v.flatten(2).gather(2,pixel[sl,None,:].expand(-1,2,-1)).transpose(1,2).reshape(-1,2)
   pos=slice(start*pp,stop*pp)
   assert np.array_equal(take(b).cpu().numpy(),base[pos]);assert np.array_equal(take(yy).cpu().numpy(),y[pos]);assert np.array_equal(take(ww).cpu().numpy(),mask[pos])
   p[pos]=take(pred).cpu().numpy()
  arrays[name]=p;np.save(A/(name+'_'+a.part+'_prediction.npy'),p);del net;torch.cuda.empty_cache()
  print('PREDICTED',name,a.part,flush=True)
 result={};permap=[];counts=mask.reshape(len(ids),pp,2).sum(1)
 for name,p in arrays.items():
  mm=metrics(p,base,y,mask);result[name]=mm
  ss=np.where(mask,(p.astype('float64')-y.astype('float64'))**2,0).reshape(len(ids),pp,2).sum(1);permap.append(ss)
 names=list(arrays);permap=np.array(permap);unique=sorted(set(terrain));lookup={v:j for j,v in enumerate(unique)};ix=np.array([lookup[t] for t in terrain])
 ts=np.stack([np.stack([np.bincount(ix,weights=permap[k,:,b],minlength=len(unique)) for b in range(2)],1) for k in range(len(names))])
 tc=np.stack([np.bincount(ix,weights=counts[:,b],minlength=len(unique)) for b in range(2)],1)
 rng=np.random.default_rng(106097);draw=rng.integers(0,len(unique),(2000,len(unique)));den=tc[draw].sum(1);ci={};gain={}
 for name in names[1:]:
  zz=np.sqrt(ts[0][draw].sum(1)/den)-np.sqrt(ts[names.index(name)][draw].sum(1)/den)
  ci[name]=np.quantile(zz,[.025,.975],axis=0).T.tolist()
  gain[name]=(np.array(result['C82']['rmse'])-result[name]['rmse']).tolist()
 eligible=frozen['eligible'];promote=False
 if eligible in gain and a.part=='confirm':
  promote=bool((np.array(gain[eligible])>=.30).all() and (np.array(ci[eligible])[:,0]>0).all())
 summary=dict(status='complete',part=a.part,metrics=result,gain_vs_C82=gain,terrain_paired95ci=ci,eligible=eligible,promote=promote,
  selection='FITonly;checkpoint/armfrozenbeforeholdout',scope=f'{len(unique)}TRAINheldoutterrains,{len(ids)}maps,{pp}uniformpoints/map;not officialVAL',
  deep_fade_definition='truePL-C82>=5dB;location_score=predPL-C82;fixedthreshold5dB. Masks scoreonly.',
  no_threshold_tuning=True,GPU=gpu,seconds=time.monotonic()-t,official_val_used=False)
 np.savez(A/(a.part+'_sums.npz'),names=np.array(names),mapid=np.array(ids),terrain=np.array(terrain),sse=permap,count=counts)
 atomic_json(A/(a.part+'_summary.json'),summary)
 print(json.dumps({k:{'rmse':v['rmse'],'deep_fade':v['deep_fade']} for k,v in result.items()}),flush=True)
if __name__=='__main__':main()
