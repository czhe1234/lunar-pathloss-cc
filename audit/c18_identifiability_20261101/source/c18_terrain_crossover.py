"""Paired heldout-TX cross-over: same terrain support versus other terrain support."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='8';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,time,hashlib,shutil
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[1];O=R/'audit/c18_identifiability_20261101/terrain_crossover';O.mkdir(exist_ok=False);(O/'source').mkdir()
for p in [Path(__file__),O.parent/'terrain_crossover_protocol.json']:shutil.copy2(p,O/'source'/p.name)
D=R/'features/c6_points';rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));ids=np.load(D/'dev_map_indices.npy')
X=np.load(R/'features/c10_points/dev_x.npy',mmap_mode='r');Y=np.load(D/'dev_y.npy',mmap_mode='r');M=np.array(np.load(D/'dev_mask.npy',mmap_mode='r'),bool)
P=np.column_stack([np.load(R/f'runs/c10_points_phase_geometry_s20261101/dev_pred{b}.npy') for b in ['415','58']]).astype(float)
byterr={}
for j,i in enumerate(ids):
 r=rows[int(i)]
 if r['group']=='multi':byterr.setdefault(r['terrain_id'],{}).setdefault((int(r['tx_row']),int(r['tx_col'])),j)
terr=sorted(byterr);assert len(terr)==39;rng=np.random.default_rng(20261104);rng.shuffle(terr);fold={t:int(i>=20) for i,t in enumerate(terr)}
supports=[[],[]];queries=[];qterrain=[];qfold=[];assign={}
for t in terr:
 maps=sorted(byterr[t].values());rng.shuffle(maps);cut=len(maps)//2;sup=maps[:cut];qry=maps[cut:];supports[fold[t]].extend(sup);queries.extend(qry);qterrain.extend([t]*len(qry));qfold.extend([fold[t]]*len(qry))
 assign[t]={'fold':fold[t],'support_index_rows':ids[sup].tolist(),'query_index_rows':ids[qry].tolist()}
 assert {(rows[int(ids[j])]['tx_row'],rows[int(ids[j])]['tx_col']) for j in sup}.isdisjoint({(rows[int(ids[j])]['tx_row'],rows[int(ids[j])]['tx_col']) for j in qry})
def points(maps):return (np.asarray(maps)[:,None]*256+np.arange(256)[None]).ravel()
si=[points(v) for v in supports];n=min(map(len,si))
for f in [0,1]:
 if len(si[f])>n:si[f]=np.sort(rng.choice(si[f],n,replace=False))
qi=points(queries);qt=np.repeat(qterrain,256);qf=np.repeat(qfold,256);tcode=np.array([{t:i for i,t in enumerate(terr)}[t] for t in qt])
manifest={'terrain_assignments':assign,'support_points_equalized':n,'query_maps':len(queries),'query_points':len(qi),'fold_terrain_counts':[20,19],'query_index_rows':ids[queries].tolist()}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');np.save(O/'query_point_indices.npy',qi)
params=json.loads((O.parent/'residual_probe/manifest.json').read_text())['parameters'];names=json.loads((R/'features/c10_points/metadata.json').read_text())['features']
same=P[qi].copy();other=P[qi].copy();start=time.time();fitstats={}
for f in [0,1]:
 A=np.array(X[si[f]]);yy=np.array(Y[si[f]],float);pp=P[si[f]];mm=M[si[f]]
 fitstats[str(f)]={}
 for k,b in enumerate(['415','58']):
  m=mm[:,k];model=lgb.train(params,lgb.Dataset(A[m],label=(yy-pp)[m,k],feature_name=names),num_boost_round=400)
  correction=model.predict(np.array(X[qi]),num_threads=8);same[qf==f,k]+=correction[qf==f];other[qf!=f,k]+=correction[qf!=f]
  model.save_model(str(O/f'fold{f}_{b}.txt'))
  fitres=pp[m,k]+model.predict(A[m],num_threads=8)-yy[m,k]
  fitstats[str(f)][b]={'support_valid':int(m.sum()),'support_RMSE':float(np.sqrt(np.mean(fitres**2)))}
  print(json.dumps({'fold':f,'band':b,'seconds':time.time()-start}),flush=True)
  del model
 del A
vX=np.array(X[qi]);e3=(2*np.sinh(3*vX[:,11:19])).max(1);e2=2*np.sinh(3*vX[:,19]);z={'all':np.ones(len(qi),bool),'occluded_proxy':e2>=0,'height_sensitive':(e3<0)&(e2>=0),'nominal_blocked':e3>=0}
preds={'C10':P[qi],'same_terrain_other_TX_support':same,'different_terrain_support':other};target=np.array(Y[qi],float);mask=M[qi];stats={};sse={}
rng=np.random.default_rng(20261105);weights=rng.multinomial(39,np.full(39,1/39),size=1000)
for pname,pr in preds.items():
 np.save(O/f'{pname}_query_prediction.npy',pr.astype(np.float32));stats[pname]={};sse[pname]={}
 for zn,zone in z.items():
  stats[pname][zn]={};sse[pname][zn]={}
  for k,b in enumerate(['415','58']):
   m=mask[:,k]&zone;err=pr[:,k]-target[:,k];ss=np.bincount(tcode,weights=err*err*m,minlength=39);cc=np.bincount(tcode,weights=m.astype(float),minlength=39);sse[pname][zn][b]=ss
   base=sse['C10'][zn][b] if pname!='C10' else ss
   stats[pname][zn][b]={'valid_pixels':int(cc.sum()),'RMSE':float(np.sqrt(ss.sum()/cc.sum())),'SSE_reduction_percent_vs_C10':float(100*(1-ss.sum()/base.sum())),
    'terrain_bootstrap95':np.percentile(100*(1-(weights@ss)/(weights@base)),[2.5,97.5]).tolist()}
comparison={};decision={}
for zn in z:
 comparison[zn]={}
 for b in ['415','58']:
  a=sse['different_terrain_support'][zn][b];v=sse['same_terrain_other_TX_support'][zn][b]
  comparison[zn][b]={'SSE_reduction_same_vs_other_percent':float(100*(1-v.sum()/a.sum())),'terrain_bootstrap95':np.percentile(100*(1-(weights@v)/(weights@a)),[2.5,97.5]).tolist()}
for b in ['415','58']:
 v=stats['same_terrain_other_TX_support']['occluded_proxy'][b];c=comparison['occluded_proxy'][b]
 decision[b]=bool(v['SSE_reduction_percent_vs_C10']>=5 and v['terrain_bootstrap95'][0]>0 and c['terrain_bootstrap95'][0]>0)
out={'state':'completed','scope':'39 internalDEV multi-TX terrains, exact same heldout query TX/pixels compared under two support conditions; diagnostic only',
 'statistics':stats,'same_vs_other_terrain_support':comparison,'scene_conditioned_learnability_supported':decision,'fit_stats':fitstats,'seconds':time.time()-start,
 'limitations':['Same-terrain labels are unavailable for a novel competition terrain. These scores are not deployable or official.',
 'A hidden fixed per-terrain parameter can also be learned from same-terrain support labels; positive transfer is not proof that raw H is sufficient.',
 'This follow-up was chosen after the main residual-probe result; it is exploratory, with fixed architecture and no query-based model selection.']}
(O/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':'completed','decision':decision,'occluded':{n:v['occluded_proxy'] for n,v in stats.items()},'paired':comparison['occluded_proxy'],'seconds':out['seconds']}),flush=True)
