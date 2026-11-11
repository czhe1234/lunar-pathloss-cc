"""Independent CPU statistics of frozen predictions. No torch or fitting."""
from pathlib import Path
import json,csv,time,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit/c4_remaining_error_20261031';DATA=ROOT.parent/'data';bands=['415','58'];rr,cc=np.indices((256,256));allresults={};t0=time.time()
for split,label in [('train','matched_train'),('val','official_val')]:
 folder=OUT/label;mapping=json.loads((folder/'indices.json').read_text());ids=mapping['index_rows'];rows=list(csv.DictReader((DATA/f'{split}_index.csv').open()));hm=np.load(DATA/f'{split}_hm.npy',mmap_mode='r');state=np.load(folder/'state_fields.npy',mmap_mode='r')
 predroot=folder if split=='train' else ROOT/'runs/c4_full_refit_s20261031/predictions/val_full_bf16';pr=[np.load(predroot/f'pl{b}.npy',mmap_mode='r') for b in bands];ys=[np.load(DATA/f'{split}_pl{b}.npy',mmap_mode='r') for b in bands];ms=[np.load(DATA/f'{split}_mask{b}.npy',mmap_mode='r') for b in bands]
 zones=['all','height_sensitive','strict_clear','deep_blocked','remaining'];zstat={z:{'sse':np.zeros(2),'n':np.zeros(2,dtype=np.int64),'bias_sum':np.zeros(2),'label_excess_sum':np.zeros(2),'model_excess_sum':np.zeros(2)} for z in zones};matrix_sse=np.zeros((2,3,3));matrix_n=np.zeros((2,3,3),dtype=np.int64);tails={str(v):{'sse':np.zeros(2),'n':np.zeros(2,dtype=np.int64)} for v in [1,3,5,10,20]};blocks={str(v):np.zeros(2) for v in [4,8,16]};common=np.zeros(3);permap=[];geometry_class={key:np.zeros((2,4),dtype=np.int64) for key in ['height3','height2','model_6.5']}
 for lo in range(0,len(ids),8):
  ii=ids[lo:lo+8];n=len(ii);rs=[rows[i] for i in ii];h=np.array(hm[[int(r['hm_row']) for r in rs]],dtype=float);tr=np.array([int(r['tx_row']) for r in rs]);tc=np.array([int(r['tx_col']) for r in rs]);ht=h[np.arange(n),tr,tc];distance=np.sqrt((rr[None]-tr[:,None,None])**2+(cc[None]-tc[:,None,None])**2+(h+1-ht[:,None,None]-3)**2);fs=np.stack([20*np.log10(4*np.pi*distance*f/299792458.) for f in [415e6,5.775e9]],1)
  pred=np.stack([p[lo:lo+n] for p in pr],1).astype(float);y=np.stack([a[ii] for a in ys],1).astype(float);valid=np.stack([np.unpackbits(a[ii],axis=1,bitorder='big').reshape(n,256,256) for a in ms],1).astype(bool);err=pred-y;square=err**2;st=state[lo:lo+n].astype(float);sens=(st[:,0]<0)&(st[:,1]>=0);clear=(st[:,1]<-.5)&(np.abs(st[:,2])>.03);deep=st[:,0]>2;rest=~(sens|clear|deep);sels=[np.ones_like(sens),sens,clear,deep,rest];ey=y-fs;ep=pred-fs
  for zone,sel in zip(zones,sels):
   v=valid&sel[:,None];z=zstat[zone];z['sse']+=np.where(v,square,0).sum((0,2,3));z['n']+=v.sum((0,2,3));z['bias_sum']+=np.where(v,err,0).sum((0,2,3));z['label_excess_sum']+=np.where(v,ey,0).sum((0,2,3));z['model_excess_sum']+=np.where(v,ep,0).sum((0,2,3))
  cy=np.digitize(ey,[3,10]);cp=np.digitize(ep,[3,10])
  for k in range(2):
   for i in range(3):
    for j in range(3):
     v=valid[:,k]&(cy[:,k]==i)&(cp[:,k]==j);matrix_sse[k,i,j]+=square[:,k][v].sum();matrix_n[k,i,j]+=v.sum()
   actualclear=valid[:,k]&(ey[:,k]<3);actualblocked=valid[:,k]&(ey[:,k]>10)
   for name,positive in [('height3',st[:,0]>=0),('height2',st[:,1]>=0),('model_6.5',ep[:,k]>=6.5)]:
    geometry_class[name][k]+=np.array([(positive&actualblocked).sum(),((~positive)&actualblocked).sum(),((~positive)&actualclear).sum(),(positive&actualclear).sum()])
  for th,z in tails.items():
   v=valid&(np.abs(err)>float(th));z['sse']+=np.where(v,square,0).sum((0,2,3));z['n']+=v.sum((0,2,3))
  for bs,total in blocks.items():
   b=int(bs);e=np.where(valid,err,0).reshape(n,2,256//b,b,256//b,b).sum((3,5));den=valid.reshape(n,2,256//b,b,256//b,b).sum((3,5));total+=np.where(den>0,e**2/np.maximum(den,1),0).sum((0,2,3))
  joint=valid.all(1);c=(err[:,0]+err[:,1])/2;d=(err[:,1]-err[:,0])/2;common+=np.array([(c[joint]**2).sum(),(d[joint]**2).sum(),joint.sum()])
  ss=np.where(valid,square,0).sum((2,3));ct=valid.sum((2,3))
  for j,i in enumerate(ii):permap.append({'index_row':i,'terrain_id':rs[j]['terrain_id'],'group':rs[j]['group'],'relief_bin':int(np.searchsorted([10,30,60,120],np.ptp(h[j]),side='right')),'sse':ss[j].tolist(),'count':ct[j].tolist()})
 total=zstat['all']['sse'];count=zstat['all']['n']
 for z in zstat.values():
  z['rmse']=np.sqrt(z['sse']/z['n']);z['sse_fraction']=z['sse']/total;z['valid_pixel_fraction']=z['n']/count;z['bias']=z['bias_sum']/z['n'];z['perfect_fix_here_remaining_rmse']=np.sqrt(np.maximum(total-z['sse'],0)/count)
  for k,v in z.items():z[k]=v.tolist()
 assert np.allclose(np.sum([zstat[k]['sse'] for k in zones[1:]],axis=0),total,rtol=1e-10)
 for z in tails.values():
  z['sse_fraction']=(z['sse']/total).tolist();z['valid_pixel_fraction']=(z['n']/count).tolist();z['sse']=z['sse'].tolist();z['n']=z['n'].tolist()
 classes={}
 for name,a in geometry_class.items():classes[name]={'counts_TP_FN_TN_FP':a.tolist(),'balanced_accuracy':(.5*(a[:,0]/(a[:,0]+a[:,1])+a[:,2]/(a[:,2]+a[:,3]))).tolist()}
 result={'maps_per_band':len(ids),'terrains':len({r['terrain_id'] for r in permap}),'rmse':np.sqrt(total/count).tolist(),'zones':zstat,'large_error_tails':tails,'target_and_predicted_excess_classes':{'thresholds_db':[3,10],'counts':matrix_n.tolist(),'sse':matrix_sse.tolist(),'mismatched_class_sse_fraction':((matrix_sse.sum((1,2))-np.trace(matrix_sse,axis1=1,axis2=2))/total).tolist(),'both_high_sse_fraction':(matrix_sse[:,2,2]/total).tolist()},'block_mean_projection_sse_fraction':{k:(v/total).tolist() for k,v in blocks.items()},'joint_band_common_error_sse_fraction':float(common[0]/(common[0]+common[1])),'joint_valid_pixels':int(common[2]),'strong_state_classification':classes,'permap':permap}
 (folder/'statistics.json').write_text(json.dumps(result,indent=2)+'\n');allresults[label]={k:v for k,v in result.items() if k!='permap'};print(label,result['rmse'],flush=True)
# Group by terrain so 51 TX maps do not count as 51 independent scenes.
rng=np.random.default_rng(20261101);boot={}
for label in ['matched_train','official_val']:
 rows=json.loads((OUT/label/'statistics.json').read_text())['permap'];terr={}
 for r in rows:
  a=terr.setdefault(r['terrain_id'],{'group':r['group'],'bin':r['relief_bin'],'sse':np.zeros(2),'n':np.zeros(2) });a['sse']+=r['sse'];a['n']+=r['count']
 groups={}
 for a in terr.values():groups.setdefault((a['group'],a['bin']),[]).append(a)
 ss=np.zeros((1000,2));nn=np.zeros_like(ss)
 for group in groups.values():
  sel=rng.integers(0,len(group),size=(1000,len(group)));a=np.array([x['sse'] for x in group]);b=np.array([x['n'] for x in group]);ss+=a[sel].sum(1);nn+=b[sel].sum(1)
 boot[label]=np.sqrt(ss/nn)
gap=boot['official_val']-boot['matched_train'];allresults['terrain_stratified_bootstrap']={'replicates':1000,'strata':'single/multi and height-range bins; independently resample terrains in each split','validation_minus_training_rmse_interval_95':np.quantile(gap,[.025,.975],axis=0).tolist()}
(OUT/'statistics_summary.json').write_text(json.dumps(allresults,indent=2)+'\n');print('statistics complete',time.time()-t0,flush=True)
