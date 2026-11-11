"""Known-terrain held-out TX interpolation; never uses query labels as inputs."""
import csv,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path.cwd();D=R.parent/'data';A=R/'audit/c111_TXcoverage';A.mkdir(exist_ok=True)
t=time.monotonic();plan=json.loads((R/'audit/c108_diagnosis/protocol.json').read_text());rows=list(csv.DictReader(open(D/'train_index.csv')))
by=defaultdict(list)
for i,r in enumerate(rows):by[r['terrain_id']].append(i)
yy=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ww=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
hm=np.load(D/'train_hm.npy',mmap_mode='r');idx=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());index={i:j for j,i in enumerate(idx)}
bb=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
gridy,gridx=np.mgrid[:256,:256];K=20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458.)
train=plan['sets']['128'];terrains=train['terrains'];trainids=np.array(train['fit']).reshape(128,3);queries=train['same_terrain_newTX']
names=['C82']+[f'{method}{n}' for n in [3,8,40] for method in ['nearest','IDW']]
record=[];rng=np.random.default_rng(111009)
def fs(h,ids):
 rr=np.array([int(rows[i]['tx_row']) for i in ids]);cc=np.array([int(rows[i]['tx_col']) for i in ids])
 z=np.floor(h[(rr+1).clip(max=255),cc])+3
 dist=np.sqrt((gridx[None]-cc[:,None,None])**2+(gridy[None]-rr[:,None,None])**2+(h[None]+1-z[:,None,None])**2).clip(.25)
 return 20*np.log10(dist),rr,cc
for j,(terrain,q) in enumerate(zip(terrains,queries)):
 first=trainids[j].tolist();available=[i for i in by[terrain] if i!=q and i not in first];rng.shuffle(available)
 donors=first+available;assert len(donors)==50 and q not in donors and len(set(donors))==50
 donors=donors[:40];h=np.array(hm[int(rows[q]['hm_row'])],np.float64)
 lf,rr,cc=fs(h,donors);lq,rq,cq=fs(h,[q]);distTX=np.sqrt((rr-rq[0])**2+(cc-cq[0])**2)
 target=np.stack([np.array(v[q],np.float64) for v in yy]);mask=np.stack([np.unpackbits(v[q]).reshape(256,256).astype(bool) for v in ww])
 value=np.stack([np.array(v[donors],np.float64) for v in yy]);valid=np.stack([np.unpackbits(v[donors],axis=1).reshape(40,256,256).astype(bool) for v in ww])
 excess=value-(lf[None]+K[:,None,None,None]);base=np.array(bb[index[q]],np.float64)
 common=mask&valid[:,:3].any(1);count=common.sum((1,2));weak=common&((target-base)>=5)
 predictions={'C82':base};nearestTX={}
 for n in [3,8,40]:
  w=valid[:,:n]/(distTX[None,:n,None,None]**2+1);den=w.sum(1);ex=np.where(valid[:,:n],excess[:,:n],0)
  mean=np.sum(w*ex,axis=1)/den.clip(1e-30);nn=w.argmax(1)
  nearest=np.take_along_axis(excess[:,:n],nn[:,None],axis=1)[:,0]
  predictions[f'IDW{n}']=lq[0][None]+K[:,None,None]+mean
  predictions[f'nearest{n}']=lq[0][None]+K[:,None,None]+nearest
  nearestTX[str(n)]=float(distTX[:n].min())
 stats={}
 for name,p in predictions.items():
  e=np.where(common,p-target,0);wp=weak&((p-base)>=5);fp=common&~weak&((p-base)>=5)
  stats[name]=dict(sse=(e**2).sum((1,2)).tolist(),weak_sse=np.where(weak,e**2,0).sum((1,2)).tolist(),tp=wp.sum((1,2)).tolist(),fp=fp.sum((1,2)).tolist())
 record.append(dict(terrain=terrain,query=q,donors=donors,count=count.tolist(),target_valid=mask.sum((1,2)).tolist(),weak_count=weak.sum((1,2)).tolist(),
  nearestTX_distance=nearestTX,metrics=stats))
def summarize(records):
 count=np.array([z['count'] for z in records]).sum(0);total=np.array([z['target_valid'] for z in records]).sum(0);wc=np.array([z['weak_count'] for z in records]).sum(0)
 out={}
 for name in names:
  s=np.array([z['metrics'][name]['sse'] for z in records]).sum(0);ws=np.array([z['metrics'][name]['weak_sse'] for z in records]).sum(0)
  tp=np.array([z['metrics'][name]['tp'] for z in records]).sum(0);fp=np.array([z['metrics'][name]['fp'] for z in records]).sum(0)
  out[name]=dict(rmse=np.sqrt(s/count).tolist(),weak_power_rmse=np.sqrt(ws/wc).tolist(),
   precision=np.divide(tp,tp+fp,out=np.zeros(2),where=(tp+fp)>0).tolist(),recall=(tp/wc).tolist())
 return dict(count=count.tolist(),coverage_of_valid=(count/total).tolist(),weak_count=wc.tolist(),metrics=out,
  median_nearestTX_m={str(n):float(np.median([z['nearestTX_distance'][str(n)] for z in records])) for n in [3,8,40]})
result=dict(scope='KnownterrainTRAINdonorlookup,3/8/40nestedTXmaps;queryTXexcluded. Notanunseenterrainpredictor;notpromotion evidence.',
 targets='Same128queryTXasC108;common8subsetalsoidenticalC108samegeometrynewTXpanel',
 interpolation='labeldonorFSPLexcessdB;inverseTX-distance-squaredweightsorclosestvaliddonor;fixedbeforequeryevaluation',
 mask='scoreonlycommonquery-validpixelswithatleastonevaliddonoramongtheoriginal3;donoravailabilityisobservedTRAINcontext',
 all128=summarize(record),common8=summarize(record[:8]),perterrain=record,official_VAL_used=False,seconds=time.monotonic()-t)
(A/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['scope','all128','common8','seconds']}))
