import sys,json,time,hashlib,argparse,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c105_context import reader,ResidualPanel
from src.c117_factorized import create
from src.c118_location import AuxField
from src.c106_metrics import metrics
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['c119'],default='c119');a=ap.parse_args()
 A=R/'audit/c119_polarity';fr=json.loads((A/'selection_frozen.json').read_text())
 g=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high');P=R/'audit/c101_residual/confirm'
 meta=json.loads((P/'done.json').read_text());ids=meta['ids'];pp=meta['points_per_map'];n=len(ids)*pp
 raw=np.load(P/'target.npy');y=raw[:,:2];mask=raw[:,2:4].astype(bool);base=np.load(P/'base.npy');pixel=torch.tensor(np.load(P/'pixel.npy').reshape(len(ids),pp),device='cuda',dtype=torch.int64)
 data=ResidualPanel(reader(),ids,'context');preds={'C82':base};start=time.monotonic()
 preds['C118parent']=np.load(R/'audit/c118_location/location_confirm_prediction.npy')
 for name,ch in fr['choices'].items():
  path=R/ch['checkpoint'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ch['sha256']
  ck=torch.load(path,map_location='cpu',weights_only=False)
  assert ck['step']==ch['step']
  if ch.get('kind')=='C82':preds[name]=base.copy();continue
  net=(create(ch['mode']) if a.family=='c117' else AuxField()).cuda().to(memory_format=torch.channels_last);net.load_state_dict(ck['state']);net.eval()
  arr=np.empty((n,2),np.float32)
  for st in range(0,len(ids),16):
   en=min(st+16,len(ids));sl=slice(st,en);x,b,yy,ww=data.batch(sl)
   with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
   def take(v):return v.flatten(2).gather(2,pixel[sl,None].expand(-1,2,-1)).transpose(1,2).reshape(-1,2).cpu().numpy()
   pos=slice(st*pp,en*pp);assert np.array_equal(take(b),base[pos]);assert np.array_equal(take(yy),y[pos]);assert np.array_equal(take(ww),mask[pos])
   arr[pos]=take(p)
  preds[name]=arr;np.save(A/(name+'_confirm_prediction.npy'),arr);del net;torch.cuda.empty_cache()
  print('predicted',name,flush=True)
 scores={};ss=[]
 for name,p in preds.items():
  scores[name]=metrics(p,base,y,mask);ss.append(np.where(mask,(p.astype('float64')-y)**2,0).reshape(len(ids),pp,2).sum(1))
 ss=np.stack(ss);count=mask.reshape(len(ids),pp,2).sum(1);rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));terr=np.array([rows[i]['terrain_id'] for i in ids])
 uniq,ix=np.unique(terr,return_inverse=True);ts=np.array([[np.bincount(ix,weights=z[:,b]) for b in range(2)] for z in ss]).transpose(0,2,1);tc=np.array([np.bincount(ix,weights=count[:,b]) for b in range(2)]).T
 rng=np.random.default_rng(117097);draw=rng.integers(0,len(uniq),(2000,len(uniq)));den=tc[draw].sum(1);ci={};gain={};names=list(preds)
 for k,name in enumerate(names[1:],1):
  ci[name]=np.quantile(np.sqrt(ts[0][draw].sum(1)/den)-np.sqrt(ts[k][draw].sum(1)/den),[.025,.975],axis=0).T.tolist()
  gain[name]=(np.array(scores['C82']['rmse'])-scores[name]['rmse']).tolist()
 eligible=fr['eligible'];promote=eligible in gain and bool((np.array(gain[eligible])>=.3).all() and (np.array(ci[eligible])[:,0]>0).all())
 np.savez(A/'confirm_sums.npz',names=np.array(names),sse=ss,count=count,terrain=terr,mapid=np.array(ids))
 atomic_json(A/'confirm_summary.json',dict(status='complete',metrics=scores,gain=gain,paired95ci=ci,eligible=eligible,promote=promote,scope='same484TRAINterrains1934maps2048pointspermap',official_val_used=False,GPU=g,seconds=time.monotonic()-start))
 print('DONE',{k:v['rmse'] for k,v in scores.items()},flush=True)
if __name__=='__main__':main()
