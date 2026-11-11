"""Independent full-native NumPy scoring of C127; raw TRAIN labels/masks only."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c127_joint';D=R.parent/'data'
def mm(p):return np.load(p,mmap_mode='r',allow_pickle=False)
def masks(ix,arrays):return np.stack([np.unpackbits(m[ix],axis=1).reshape(len(ix),256,256).astype(bool) for m in arrays],1)
def score(pred,ids,base,yarr,marr,rows):
 n=len(ids);stat=np.zeros((n,8),np.float64);sign=np.zeros(2);true=np.zeros(2);guess=np.zeros(2);correct=np.zeros(2);amp=np.zeros(2);bs=bn=0.;tailpred=[]
 for start in range(0,n,24):
  ix=ids[start:start+24];v=masks(ix,marr);y=np.stack([a[ix] for a in yarr],1).astype(np.float64)
  p=np.asarray(pred[start:start+len(ix)],dtype=np.float64);b=np.asarray(base[start:start+len(ix)],dtype=np.float64)
  e=np.where(v,p-y,0);tail=v&(np.abs(y-b)>=5);pt=v&(np.abs(p-b)>=5)
  stat[start:start+len(ix),:2]=(e*e).sum((2,3))
  stat[start:start+len(ix),2:4]=v.sum((2,3))
  stat[start:start+len(ix),4:6]=np.where(tail,e*e,0).sum((2,3))
  stat[start:start+len(ix),6:8]=tail.sum((2,3))
  sign+=(tail&((p-b)*(y-b)>0)).sum((0,2,3));true+=tail.sum((0,2,3));guess+=pt.sum((0,2,3));correct+=(tail&pt).sum((0,2,3));amp+=np.where(tail,np.abs(p-b),0).sum((0,2,3))
  both=v[:,0]&v[:,1];be=np.where(both,e[:,1]-e[:,0],0);bs+=(be*be).sum();bn+=both.sum()
 z=stat.sum(0);groups=defaultdict(list)
 for j,i in enumerate(ids):groups[rows[i]['terrain_id']].append(j)
 pair_s=np.zeros(2);pair_n=np.zeros(2);common_s=np.zeros(2);total_s=np.zeros(2)
 for g in groups.values():
  if len(g)<2:continue
  ix=[ids[j] for j in g];v=masks(ix,marr);y=np.stack([a[ix] for a in yarr],1).astype(np.float64)
  e=np.where(v,np.asarray(pred[g],dtype=np.float64)-y,0.)
  num=v.sum(0);sume=e.sum(0);sq=(e*e).sum(0)
  pair_s+=(num*sq-sume*sume).sum((1,2));pair_n+=(num*(num-1)/2).sum((1,2))
  ok=num>=2;common_s+=np.where(ok,sume*sume/np.maximum(num,1),0).sum((1,2));total_s+=np.where(ok,sq,0).sum((1,2))
 result=dict(rmse=np.sqrt(z[:2]/z[2:4]).tolist(),valid=z[2:4].astype(int).tolist(),tail_rmse=np.sqrt(z[4:6]/z[6:8]).tolist(),tail_count=z[6:8].astype(int).tolist(),
  tail_sign_accuracy=(sign/np.maximum(true,1)).tolist(),tail_precision_at_fixed5dB=(correct/np.maximum(guess,1)).tolist(),
  tail_recall_at_fixed5dB=(correct/np.maximum(true,1)).tolist(),tail_mean_abs_correction=(amp/np.maximum(true,1)).tolist(),
  band_difference_rmse=float(np.sqrt(bs/bn)),TX_difference_rmse=np.sqrt(pair_s/np.maximum(pair_n,1)).tolist(),
  multiTX_common_error_fraction=(common_s/np.maximum(total_s,1e-30)).tolist())
 return result,stat
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arms',nargs='*');args=ap.parse_args()
 plan=json.loads((A/'protocol.json').read_text());ids=plan['monitor'];rows=list(csv.DictReader(open(D/'train_index.csv')))
 ya=[mm(D/f'train_pl{b}.npy') for b in ['415','58']];ma=[mm(D/f'train_mask{b}.npy') for b in ['415','58']]
 ib=json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids'];lookup={int(i):j for j,i in enumerate(ib)}
 base=np.array(mm(R/'audit/c88_mesh/base.npy')[[lookup[i] for i in ids]],np.float32)
 result=dict(scope='628maps/128 original TRAIN-held-out terrains; full native valid pixels; not officialVAL',official_val_used=False,models={})
 result['models']['C82'],baseline=score(base,ids,base,ya,ma,rows)
 terrains=sorted(set(rows[i]['terrain_id'] for i in ids));ti={t:j for j,t in enumerate(terrains)};loc=np.array([ti[rows[i]['terrain_id']] for i in ids])
 stats={'C82':baseline};grouped={}
 arms=args.arms or plan['arms']
 for name in arms:
  out=R/'runs'/('c127_'+name)
  saved=json.loads((out/'result.json').read_text());assert saved['status']=='complete'
  assert json.loads((out/'indices.json').read_text())==ids
  pred=mm(out/'monitor_prediction.npy')
  v,s=score(pred,ids,base,ya,ma,rows)
  assert np.max(np.abs(np.array(v['rmse'])-saved['monitor']['rmse']))<1e-5,(name,v['rmse'],saved['monitor']['rmse'])
  v.update(best_epoch=saved['best_epoch'],last_epoch=saved['last_epoch'])
  v['gain_vs_C82']=(np.array(result['models']['C82']['rmse'])-v['rmse']).tolist()
  result['models'][name]=v;stats[name]=s
  print('SCORE',name,json.dumps(v),flush=True)
 for name,s in stats.items():
  g=np.zeros((len(terrains),8),np.float64);np.add.at(g,loc,s);grouped[name]=g
 rng=np.random.default_rng(127181);draw=rng.integers(len(terrains),size=(3000,len(terrains)))
 def draws(name):
  z=grouped[name][draw].sum(1);return np.sqrt(z[:,:2]/z[:,2:4])
 base_draw=draws('C82')
 for name in arms:
  gain=base_draw-draws(name);result['models'][name]['gain_CI95']=np.quantile(gain,[.025,.975],axis=0).T.tolist()
 comparisons={}
 for route in ['direct','residual']:
  a=route+'_independent';b=route+'_joint'
  if a in stats and b in stats:
   gain=np.array(result['models'][a]['rmse'])-result['models'][b]['rmse']
   comparisons[route]=dict(joint_gain_vs_independent=gain.tolist(),CI95=np.quantile(draws(a)-draws(b),[.025,.975],axis=0).T.tolist())
 result['joint_effect']=comparisons
 result['monitor_gate_passed']=[name for name in arms if min(result['models'][name]['gain_vs_C82'])>=.20]
 np.savez(A/'terrain_sse.npz',**grouped)
 (A/'score.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('DONE',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
