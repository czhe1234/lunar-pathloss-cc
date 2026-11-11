"""CPU-only error subdivisions, same-terrain TX contrasts and exact-input checks."""
from pathlib import Path
import json,csv,collections,itertools,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';OUT=ROOT/'audit/c4_remaining_error_20261031';bands=['415','58'];t0=time.time();output={}
for split,label in [('train','matched_train'),('val','official_val')]:
 folder=OUT/label;mapping=json.loads((folder/'indices.json').read_text());ids=mapping['index_rows'];rows=list(csv.DictReader((DATA/f'{split}_index.csv').open()));rs=[rows[i] for i in ids];hm=np.load(DATA/f'{split}_hm.npy',mmap_mode='r');state=np.load(folder/'state_fields.npy',mmap_mode='r');predroot=folder if split=='train' else ROOT/'runs/c4_full_refit_s20261031/predictions/val_full_bf16';ps=[np.load(predroot/f'pl{b}.npy',mmap_mode='r') for b in bands];ys=[np.load(DATA/f'{split}_pl{b}.npy',mmap_mode='r') for b in bands];ms=[np.load(DATA/f'{split}_mask{b}.npy',mmap_mode='r') for b in bands]
 names=['height_sensitive','clear_near_boundary','clear_grazing','clear_robust','shallow_blocked','deep_blocked'];stats={z:{'n':np.zeros(2,dtype=np.int64),'sse':np.zeros(2)} for z in names}
 rr=np.array(mapping['receiver_rows']);cc=np.array(mapping['receiver_cols']);Y=np.empty((len(ids),2,len(rr)));P=np.empty_like(Y);M=np.empty_like(Y,dtype=bool);E=np.empty((len(ids),len(rr)));FS=np.empty_like(Y)
 for lo in range(0,len(ids),8):
  ii=ids[lo:lo+8];n=len(ii);st=state[lo:lo+n].astype(float);e3,e2,cos=st[:,0],st[:,1],st[:,2];sens=(e3<0)&(e2>=0);near=(e3<0)&(e2<0)&(e2>=-.5);grazing=(e2<-.5)&(np.abs(cos)<=.03);clear=(e2<-.5)&(np.abs(cos)>.03);shallow=(e3>=0)&(e3<=2);deep=e3>2;sels=[sens,near,grazing,clear,shallow,deep];assert np.all(np.sum(sels,axis=0)==1)
  y=np.stack([a[ii] for a in ys],1).astype(float);p=np.stack([a[lo:lo+n] for a in ps],1).astype(float);m=np.stack([np.unpackbits(a[ii],axis=1,bitorder='big').reshape(n,256,256) for a in ms],1).astype(bool);err=(p-y)**2
  for name,sel in zip(names,sels):
   v=m&sel[:,None];stats[name]['n']+=v.sum((0,2,3));stats[name]['sse']+=np.where(v,err,0).sum((0,2,3))
  Y[lo:lo+n]=y[:,:,rr,cc];P[lo:lo+n]=p[:,:,rr,cc];M[lo:lo+n]=m[:,:,rr,cc];E[lo:lo+n]=e3[:,rr,cc]
  for j in range(n):
   r=rs[lo+j];tr=int(r['tx_row']);tc=int(r['tx_col']);h=hm[int(r['hm_row'])];d=np.sqrt((rr-tr)**2+(cc-tc)**2+(h[rr,cc]+1-h[tr,tc]-3)**2);FS[lo+j]=np.array([20*np.log10(4*np.pi*d*f/299792458.) for f in [415e6,5.775e9]])
 total=np.sum([s['sse'] for s in stats.values()],0);count=np.sum([s['n'] for s in stats.values()],0)
 for s in stats.values():
  s['rmse']=np.sqrt(s['sse']/np.maximum(s['n'],1));s['sse_fraction']=s['sse']/total;s['pixel_fraction']=s['n']/count
  for k,v in s.items():s[k]=v.tolist()
 groups=collections.defaultdict(list)
 for j,r in enumerate(rs):
  if r['group']=='multi':groups[r['terrain_id']].append(j)
 pairset=set();distances=[]
 for group in groups.values():
  xy=np.array([[int(rs[j]['tx_row']),int(rs[j]['tx_col'])] for j in group]);ds=np.sum((xy[:,None]-xy[None,:])**2,axis=-1).astype(float);ds[ds==0]=np.inf
  for a,b in enumerate(ds.argmin(1)):pairset.add(tuple(sorted([group[a],group[b]])))
 pairs=sorted(pairset);contrasts={}
 for a,b in pairs:
  ar,br=rs[a],rs[b];distances.append(np.hypot(int(ar['tx_row'])-int(br['tx_row']),int(ar['tx_col'])-int(br['tx_col'])))
 for state_name in ['all_common_valid','geometric_deep_blocked_both','target_high_excess_both','deep_and_high_excess_both']:
  contrasts[state_name]={}
  for k,band in enumerate(bands):
   real=[];model=[];real_ex=[];model_ex=[];used_terr=set();used_pairs=0
   for a,b in pairs:
    v=M[a,k]&M[b,k];deep=(E[a]>2)&(E[b]>2);high=(Y[a,k]-FS[a,k]>10)&(Y[b,k]-FS[b,k]>10)
    if state_name=='geometric_deep_blocked_both':v&=deep
    elif state_name=='target_high_excess_both':v&=high
    elif state_name=='deep_and_high_excess_both':v&=deep&high
    if not v.any():continue
    used_pairs+=1;used_terr.add(rs[a]['terrain_id']);dy=(Y[b,k]-Y[a,k])[v];dp=(P[b,k]-P[a,k])[v];df=(FS[b,k]-FS[a,k])[v];real.append(dy);model.append(dp);real_ex.append(dy-df);model_ex.append(dp-df)
   def summarize(a,b):
    a=np.concatenate(a);b=np.concatenate(b)
    return {'n_pair_receiver_observations':len(a),'target_delta_std':float(np.std(a)),'predicted_delta_std':float(np.std(b)),'predicted_to_target_std_ratio':float(np.std(b)/np.std(a)),'delta_rmse':float(np.sqrt(np.mean((b-a)**2))),'correlation':float(np.corrcoef(a,b)[0,1]),'delta_bias':float(np.mean(b-a))}
   contrasts[state_name][band]={'terrains':len(used_terr),'pairs':used_pairs,'pathloss_db':summarize(real,model),'excess_over_fspl_db':summarize(real_ex,model_ex)}
 output[label]={'subzones':stats,'same_terrain_nearest_tx':{'terrains':len(groups),'unique_undirected_pairs':len(pairs),'tx_displacement_m_quantiles_10_50_90':np.quantile(distances,[.1,.5,.9]).tolist(),'receiver_points_per_map':len(rr),'conditions':contrasts},'note':'Pair-receiver observations are correlated within terrain; not independent sample counts. High-excess classes are attenuation proxies, not ray-traced physical states.'}
 print(label, json.dumps(output[label]),flush=True)
# Exact identical provided inputs: same stored terrain and same integer TX coordinates.
repeat={}
for split in ['train','val']:
 rows=list(csv.DictReader((DATA/f'{split}_index.csv').open()));groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[(int(r['hm_row']),int(r['tx_row']),int(r['tx_col']))].append(i)
 groups=[g for g in groups.values() if len(g)>1];details=[];ss=np.zeros(2);nn=np.zeros(2,dtype=np.int64)
 y=[np.load(DATA/f'{split}_pl{b}.npy',mmap_mode='r') for b in bands];m=[np.load(DATA/f'{split}_mask{b}.npy',mmap_mode='r') for b in bands]
 for g in groups:
  for a,b in itertools.combinations(g,2):
   z={'index_rows':[a,b],'sample_ids':[rows[a]['sample_id'],rows[b]['sample_id']],'bands':{}}
   for k,band in enumerate(bands):
    ma=np.unpackbits(m[k][a],bitorder='big').reshape(256,256).astype(bool);mb=np.unpackbits(m[k][b],bitorder='big').reshape(256,256).astype(bool);v=ma&mb;d=y[k][a].astype(float)-y[k][b].astype(float);s=float(np.sum(d[v]**2));n=int(v.sum());ss[k]+=s;nn[k]+=n;z['bands'][band]={'common_valid_pixels':n,'masks_identical':bool(np.array_equal(ma,mb)),'pair_rmse':float(np.sqrt(s/n)),'max_abs_valid_difference':float(np.max(np.abs(d[v])))}
   details.append(z)
 repeat[split]={'duplicate_input_groups':len(groups),'pairs':details,'pooled_pair_rmse':[float(np.sqrt(s/n)) if n else None for s,n in zip(ss,nn)],'pooled_common_valid_pixels':nn.tolist()}
repeat['limitation']='Repeated coordinates are not proven independent ray-tracing replicates; they may share deterministic seeds or source outputs. Cannot infer a global conditional-noise floor or full input sufficiency.'
(OUT/'structure_audit.json').write_text(json.dumps(output,indent=2)+'\n');(OUT/'identical_input_recheck.json').write_text(json.dumps(repeat,indent=2)+'\n');print('complete',time.time()-t0,flush=True)
