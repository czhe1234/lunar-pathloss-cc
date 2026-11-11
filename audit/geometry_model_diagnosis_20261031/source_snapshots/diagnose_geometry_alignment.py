"""CPU data/geometry consistency diagnostics, independent of any new model fit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';OUT=ROOT/'audit/geometry_model_diagnosis_20261031';OUT.mkdir(exist_ok=True)
rows=list(csv.DictReader((DATA/'train_index.csv').open()));H=np.load(DATA/'train_hm.npy',mmap_mode='r',allow_pickle=False)
Y=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')];M=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')]
PRED=ROOT/'runs/b0_raw_s20261031/predictions/train_dev_bf16';indices=json.loads((PRED/'indices.json').read_text())['index_rows'];P=[np.load(PRED/f'pl{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')]
meta=json.loads((DATA/'metadata.json').read_text());wavelength=np.array([299792458./meta['bands_hz'][b] for b in ('415','58')])
for name,ref in json.loads((ROOT/'audit/raw_audit.json').read_text())['files'].items():
 st=(DATA/name).stat();assert st.st_size==ref['size_bytes'] and st.st_mtime_ns==ref['mtime_ns']

def write(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def add(a,b):
 for k,v in b.items():
  if isinstance(v,dict):add(a.setdefault(k,{}),v)
  else:a[k]=a.get(k,0)+v
 return a

def anchor(i):
 r=rows[i];tr=int(r['tx_row']);tc=int(r['tx_col']);result=[]
 for b in (0,1):
  valid=np.unpackbits(M[b][i],bitorder='big').reshape(256,256).astype(bool);label=np.asarray(Y[b][i],np.float32)
  a=np.where(valid,label,np.inf);rr,cc=np.unravel_index(np.argmin(a),a.shape)
  result.append([float(np.hypot(rr-tr,cc-tc)),int(rr-tr),int(cc-tc),float(a[rr,cc])])
 return result

def interpolate(h,rr,cc,triangle=0):
 if triangle==0:return map_coordinates(h,np.stack((rr,cc)),order=1,mode='nearest',prefilter=False)
 rr=np.clip(rr,0,255);cc=np.clip(cc,0,255);r0=np.minimum(np.floor(rr).astype(int),254);c0=np.minimum(np.floor(cc).astype(int),254);fy=rr-r0;fx=cc-c0
 h00=h[r0,c0];h01=h[r0,c0+1];h10=h[r0+1,c0];h11=h[r0+1,c0+1]
 if triangle==1:return np.where(fy>=fx,(1-fy)*h00+(fy-fx)*h10+fx*h11,(1-fx)*h00+(fx-fy)*h01+fy*h11)
 return np.where(fx+fy<=1,(1-fx-fy)*h00+fx*h01+fy*h10,(1-fy)*h01+(1-fx)*h10+(fx+fy-1)*h11)

def profile(h,tr,tc,rr,cc,*,height_shift=(0,0),rx_shift=(0,0),rx_height=1.,tx_height=3.,triangle=0,n=192):
 rr=rr.astype(np.float32)+rx_shift[0];cc=cc.astype(np.float32)+rx_shift[1]
 sr,sc=height_shift;t=np.arange(1,n+1,dtype=np.float32)/(n+1)
 txz=float(interpolate(h,np.array([tr+sr]),np.array([tc+sc]),triangle)[0])+tx_height
 rz=interpolate(h,rr+sr,cc+sc,triangle)+rx_height
 gy=tr+(rr-tr)[:,None]*t;gx=tc+(cc-tc)[:,None]*t
 terrain=interpolate(h,gy+sr,gx+sc,triangle)
 e=(terrain-(txz+(rz[:,None]-txz)*t)).max(1)
 distance=np.sqrt((rr-tr)**2+(cc-tc)**2+(rz-txz)**2)
 return e,distance

rr,cc=np.meshgrid(np.arange(4,256,8),np.arange(4,256,8),indexing='ij');rr=rr.ravel();cc=cc.ravel()
def transforms(h):
 return {'identity':h,'flip_row':h[::-1],'flip_col':h[:,::-1],'rotate180':h[::-1,::-1],'transpose':h.T,'rotate90':np.rot90(h,1),'rotate270':np.rot90(h,3),'anti_transpose':h.T[::-1,::-1]}

def geometry_one(item):
 cohort,i=item;r=rows[i];h=H[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);e0,dist=profile(h,tr,tc,rr,cc)
 labels=np.stack([a[i,rr,cc].astype(np.float64) for a in Y]);valid=np.stack([np.unpackbits(a[i],bitorder='big').reshape(256,256)[rr,cc].astype(bool) for a in M])
 normalized_loss=labels-20*np.log10(4*np.pi*dist[None]/wavelength[:,None])
 both=valid.all(0)&(np.hypot(rr-tr,cc-tc)>=8)
 clear_proxy=(normalized_loss.max(0)<=3)&(normalized_loss.min(0)>=-6)&both
 blocked_proxy=(normalized_loss.min(0)>=10)&both
 variants={k:(hh,{}) for k,hh in transforms(h).items()}
 variants['wrong_terrain_control']=(H[(int(r['hm_row'])+1)%len(H)],{})
 for dy,dx in [(dy,dx) for dy in (-1.,-.5,0.,.5,1.) for dx in (-1.,-.5,0.,.5,1.) if (dy,dx)!=(0,0)]:variants[f'hshift_{dy}_{dx}']=(h,{'height_shift':(dy,dx)})
 for dy,dx in [(-.5,0),(.5,0),(0,-.5),(0,.5)]:variants[f'rxshift_{dy}_{dx}']=(h,{'rx_shift':(dy,dx)})
 for height in (.5,1.5,2.):variants[f'rxheight_{height}']=(h,{'rx_height':height})
 for height in (2.,4.):variants[f'txheight_{height}']=(h,{'tx_height':height})
 for tri in (1,2):variants[f'triangle_{tri}']=(h,{'triangle':tri})
 output={}
 for name,(hh,kwargs) in variants.items():
  e=e0 if name=='identity' else profile(hh,tr,tc,rr,cc,**kwargs)[0]
  gclear=e<0;safe=e<-.5
  output[name]={'proxy_clear_n':int(clear_proxy.sum()),'proxy_clear_hit':int((clear_proxy&gclear).sum()),'proxy_blocked_n':int(blocked_proxy.sum()),'proxy_blocked_hit':int((blocked_proxy&~gclear).sum()),'joint_valid_n':int(both.sum()),'safe_clear_n':int((safe&both).sum()),'safe_clear_both_loss_above_6db':int((safe&both&(normalized_loss.min(0)>6)).sum()),'classification_disagrees_identity':int(((e>0)!=(e0>0))[both].sum())}
 return {'cohort':cohort,'index_row':i,'terrain_id':r['terrain_id'],'variants':output}

def features_one(j):
 i=indices[j];r=rows[i];h=H[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);E,d=profile(h,tr,tc,rr,cc)
 sy,sx=np.gradient(h);dx=cc-tc;dy=rr-tr;dz=h[rr,cc]+1.-(float(h[tr,tc])+3.)
 cosine=np.abs(-sx[rr,cc]*dx-sy[rr,cc]*dy+dz)/(np.sqrt(1+sx[rr,cc]**2+sy[rr,cc]**2)*d)
 return {'E':E.astype(np.float32),'distance':d.astype(np.float32),'cosine':cosine.astype(np.float32),'labels':np.stack([a[i,rr,cc] for a in Y]).astype(np.float32),'predictions':np.stack([a[j,rr,cc] for a in P]).astype(np.float32),'valid':np.stack([np.unpackbits(a[i],bitorder='big').reshape(256,256)[rr,cc] for a in M]).astype(bool),'slope':np.hypot(sx[rr,cc],sy[rr,cc]).astype(np.float32)}

def main():
 start=time.time();anchors=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for k,a in enumerate(pool.map(anchor,range(len(rows)))):
   anchors.append(a)
   if (k+1)%5000==0:print(json.dumps({'phase':'anchors','maps':k+1,'seconds':time.time()-start}),flush=True)
 anchors=np.array(anchors);np.save(OUT/'train_tx_anchor_checks.npy',anchors)
 summary={}
 for k,b in enumerate(('415','58')):
  v=anchors[:,k,0];summary[b]={'maps':len(v),'distance_quantiles_m':{str(q):float(np.quantile(v,q)) for q in (.5,.9,.99,1)},'within_3m_fraction':float((v<=3).mean()),'over_3m_original_rows':np.flatnonzero(v>3).tolist(),'mean_peak_offset_rc':anchors[:,k,1:3].mean(0).tolist()}
 write('tx_anchor_summary.json',summary);print(json.dumps({'phase':'anchors_done','summary':summary}),flush=True)
 by={}
 for i in indices:by.setdefault(rows[i]['terrain_id'],[]).append(i)
 rng=np.random.default_rng(2026103104);terrains=rng.choice(sorted(by),128,replace=False);selected=[int(rng.choice(by[t])) for t in terrains]
 items=[('discovery' if k<64 else 'confirmation',i) for k,i in enumerate(selected)];write('geometry_selection.json',{'seed':2026103104,'items':[{'cohort':c,'index_row':i,'terrain_id':rows[i]['terrain_id']} for c,i in items]})
 totals={};records=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for k,result in enumerate(pool.map(geometry_one,items)):
   records.append(result);add(totals.setdefault(result['cohort'],{}),result['variants'])
   if (k+1)%16==0:print(json.dumps({'phase':'geometry_hypotheses','maps':k+1,'total':len(items),'seconds':time.time()-start}),flush=True)
 for cohort,variants in totals.items():
  for name,s in variants.items():
   s['proxy_balanced_accuracy']=.5*(s['proxy_clear_hit']/s['proxy_clear_n']+s['proxy_blocked_hit']/s['proxy_blocked_n'])
   s['safe_clear_above_6db_fraction']=s['safe_clear_both_loss_above_6db']/s['safe_clear_n'] if s['safe_clear_n'] else None
 write('geometry_hypotheses.json',{'scope':'128 distinct development terrains; 64 discovery/64 confirmation; label power classes are proxies, not true LOS labels','variants':totals,'per_map':records})
 print(json.dumps({'phase':'geometry_done','identity':{c:v['identity'] for c,v in totals.items()},'top5':{c:sorted([(name,s['proxy_balanced_accuracy']) for name,s in v.items()],key=lambda x:-x[1])[:5] for c,v in totals.items()}}),flush=True)
 features=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for k,result in enumerate(pool.map(features_one,range(len(indices)))):
   features.append(result)
   if (k+1)%400==0:print(json.dumps({'phase':'full_development_geometry_cache','maps':k+1,'seconds':time.time()-start}),flush=True)
 arrays={k:np.stack([f[k] for f in features]) for k in features[0]};arrays.update({'original_rows':np.array(indices,dtype=np.int32),'receiver_rows':rr,'receiver_cols':cc})
 np.savez_compressed(OUT/'development_geometry_samples.npz',**arrays)
 write('geometry_completion.json',{'done':True,'uses_gpu':False,'model_training':False,'anchor_maps':len(rows),'geometry_hypothesis_maps':128,'development_maps':len(indices),'receivers_per_map':len(rr),'seconds':time.time()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sources':['train_index.csv','train_hm.npy','train_pl415.npy','train_pl58.npy','train_mask415.npy','train_mask58.npy','CC B0 saved development predictions']})
 print(json.dumps({'done':True,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
