"""Independent, CPU-only audit. Only reads release-level data; never imports project code."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv, hashlib, json, os, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'data'
OUT=ROOT/'audit'
OUT.mkdir(exist_ok=True)
t0=time.time()
report={'started_utc':datetime.now(timezone.utc).isoformat(),'source':str(DATA),'workdir':str(ROOT),'python':sys.version,'numpy':np.__version__,'scope':'release-level arrays, indices, metadata, manifest, submission example only; no other project code/results/recovered data read','files':{},'splits':{},'issues':[]}
meta=json.loads((DATA/'metadata.json').read_text())
indices={}
terrain_hashes={}
allstats=[]
Q=[0,0.01,0.1,0.5,0.9,0.99,1]
def quant(v):
 v=np.asarray(v,dtype=np.float64); v=v[np.isfinite(v)]
 return dict(zip(map(str,Q),map(float,np.quantile(v,Q)))) if len(v) else {}
def filehash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   block=f.read(8*1024*1024)
   if not block: break
   h.update(block)
 return h.hexdigest()
def histogram_summary(hist):
 values=np.arange(65536,dtype=np.uint16).view(np.float16).astype(np.float64)
 keep=(hist>0)&np.isfinite(values)
 values=values[keep]; counts=hist[keep]
 order=np.argsort(values); values=values[order]; counts=counts[order]
 n=int(counts.sum())
 if not n: return {'count':0}
 ranks=np.cumsum(counts)
 def at(k): return float(values[np.searchsorted(ranks,k+1)])
 qs={}
 for q in Q:
  rank=q*(n-1); lo=int(np.floor(rank)); hi=int(np.ceil(rank))
  qs[str(q)]=at(lo)*(hi-rank)+at(hi)*(rank-lo) if hi>lo else at(lo)
 mean=float(np.dot(values,counts)/n)
 return {'count':n,'mean':mean,'std':float(np.sqrt(max(0,np.dot(values*values,counts)/n-mean*mean))),'quantiles':qs}
for split in ['train','val']:
 with (DATA/f'{split}_index.csv').open() as f: rows=list(csv.DictReader(f))
 indices[split]=rows
 counts=Counter(r['terrain_id'] for r in rows)
 sid=[r['sample_id'] for r in rows]
 hm_to_terrain=defaultdict(set); terrain_to_hm=defaultdict(set)
 bad=[]
 for i,r in enumerate(rows):
  hm_to_terrain[int(r['hm_row'])].add(r['terrain_id']); terrain_to_hm[r['terrain_id']].add(int(r['hm_row']))
  expected=f"{int(r['terrain_id']):05d}_{int(r['tx_index']):02d}"
  if r['sample_id']!=expected or not (0<=int(r['tx_row'])<256 and 0<=int(r['tx_col'])<256): bad.append(i)
  if r['group'] not in ['single','multi'] or counts[r['terrain_id']] != (1 if r['group']=='single' else 51): bad.append(i)
 split_report={'samples':len(rows),'terrains':len(counts),'sample_groups':dict(Counter(r['group'] for r in rows)),'terrain_groups':dict(Counter(next(r['group'] for r in rows if r['terrain_id']==tid) for tid in counts)),'duplicate_sample_ids':len(sid)-len(set(sid)),'invalid_index_rows':sorted(set(bad)),'hm_row_to_multiple_terrains':sum(len(v)!=1 for v in hm_to_terrain.values()),'terrain_to_multiple_hm_rows':sum(len(v)!=1 for v in terrain_to_hm.values()),'bands':{}}
 report['splits'][split]=split_report
 if len(rows)!=meta['splits'][split]['samples_per_band'] or len(counts)!=meta['splits'][split]['terrains']: report['issues'].append(f'{split}: metadata count mismatch')
 h=np.load(DATA/f'{split}_hm.npy',mmap_mode='r',allow_pickle=False)
 if set(hm_to_terrain)!=set(range(len(h))): report['issues'].append(f'{split}: incomplete heightmap row mapping')
 hmins=[]; hmaxs=[]; hrms=[]; hashes=defaultdict(list); terrain_rows=[]
 for i in range(len(h)):
  a=np.asarray(h[i]); fin=np.isfinite(a)
  sha=hashlib.sha256(a.tobytes()).hexdigest()
  tid=next(iter(hm_to_terrain[i])); hashes[sha].append(tid)
  mn=float(a.min()); mx=float(a.max()); hmins.append(mn);hmaxs.append(mx)
  grad=np.hypot(np.diff(a,axis=0)[:,:-1],np.diff(a,axis=1)[:-1,:]); rms=float(np.sqrt(np.mean(grad.astype(np.float64)**2)));hrms.append(rms)
  terrain_rows.append({'split':split,'terrain_id':tid,'hm_row':i,'min_m':mn,'max_m':mx,'slope_rms':rms,'nonfinite':int((~fin).sum()),'sha256_array':sha})
 terrain_hashes[split]=hashes
 split_report['heightmaps']={'shape':list(h.shape),'dtype':str(h.dtype),'min_distribution':quant(hmins),'relief_distribution':quant(hmaxs),'slope_rms_distribution':quant(hrms),'nonfinite_pixels':sum(x['nonfinite'] for x in terrain_rows),'constant_maps':sum(x['max_m']==x['min_m'] for x in terrain_rows),'duplicate_hash_groups':[v for v in hashes.values() if len(v)>1]}
 with (OUT/f'{split}_terrain_stats.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(terrain_rows[0]));w.writeheader();w.writerows(terrain_rows)
 print(f'{split}: indices and {len(h)} heightmaps checked',flush=True)
 band_arrays={b:np.load(DATA/f'{split}_pl{b}.npy',mmap_mode='r',allow_pickle=False) for b in ['415','58']}
 mask_arrays={b:np.load(DATA/f'{split}_mask{b}.npy',mmap_mode='r',allow_pickle=False) for b in ['415','58']}
 bandstats={b:[] for b in band_arrays}
 hists={b:{'valid':np.zeros(65536,np.int64),'invalid':np.zeros(65536,np.int64)} for b in band_arrays}
 pair={'n':0,'sx':0.,'sy':0.,'sxx':0.,'syy':0.,'sxy':0.,'sd':0.,'sdd':0.}
 for start in range(0,len(rows),32):
  end=min(start+32,len(rows)); values={}; masks={}
  for b in ['415','58']:
   y=np.asarray(band_arrays[b][start:end]); m=np.unpackbits(mask_arrays[b][start:end],axis=1,bitorder='big').reshape(y.shape).astype(bool)
   values[b]=y.astype(np.float32);masks[b]=m
   fin=np.isfinite(y); good=m&fin; bad=(~m)&fin
   hists[b]['valid']+=np.bincount(y.view(np.uint16)[good],minlength=65536)
   hists[b]['invalid']+=np.bincount(y.view(np.uint16)[bad],minlength=65536)
   axes=(1,2); vf=values[b]
   mins=np.min(vf,axis=axes,where=good,initial=np.inf); maxs=np.max(vf,axis=axes,where=good,initial=-np.inf)
   nums=good.sum(axis=axes); means=np.sum(vf,axis=axes,where=good,dtype=np.float64)/np.maximum(nums,1)
   stds=np.sqrt(np.maximum(0,np.sum(vf*vf,axis=axes,where=good,dtype=np.float64)/np.maximum(nums,1)-means**2))
   for j,r in enumerate(rows[start:end]):
    row={'split':split,'band':b,'sample_id':r['sample_id'],'terrain_id':r['terrain_id'],'group':r['group'],'valid_count':int(m[j].sum()),'finite_valid_count':int(nums[j]),'nonfinite_count':int((~fin[j]).sum()),'nonpositive_count':int((vf[j]<=0).sum()),'valid_min_db':float(mins[j]),'valid_max_db':float(maxs[j]),'valid_mean_db':float(means[j]),'valid_std_db':float(stds[j]),'valid_eq145_count':int((good[j]&(y[j]==145)).sum()),'invalid_eq145_count':int(((~m[j])&(y[j]==145)).sum()),'invalid_not145_count':int(((~m[j])&(y[j]!=145)).sum()),'tx_valid':int(m[j,int(r['tx_row']),int(r['tx_col'])]),'tx_pathloss_db':float(vf[j,int(r['tx_row']),int(r['tx_col'])])}
    bandstats[b].append(row)
  common=masks['415']&masks['58']&np.isfinite(values['415'])&np.isfinite(values['58'])
  x=values['415'][common].astype(np.float64); y=values['58'][common].astype(np.float64);d=y-x
  pair['n']+=len(x)
  for key,a in [('sx',x),('sy',y),('sxx',x*x),('syy',y*y),('sxy',x*y),('sd',d),('sdd',d*d)]: pair[key]+=float(a.sum())
  if start%2048==0: print(f'{split}: both bands {end}/{len(rows)} maps scanned; elapsed {time.time()-t0:.1f}s',flush=True)
 for b,stats in bandstats.items():
  stat={'shape':list(band_arrays[b].shape),'dtype':str(band_arrays[b].dtype),'mask_shape':list(mask_arrays[b].shape),'mask_dtype':str(mask_arrays[b].dtype),'total_pixels':len(rows)*65536,'valid_pixels':sum(s['valid_count'] for s in stats),'nonfinite_pixels':sum(s['nonfinite_count'] for s in stats),'nonpositive_pixels':sum(s['nonpositive_count'] for s in stats),'empty_valid_maps':sum(s['valid_count']==0 for s in stats),'constant_valid_maps':sum(s['valid_std_db']==0 for s in stats),'valid_fraction_distribution':quant([s['valid_count']/65536 for s in stats]),'map_valid_mean_distribution':quant([s['valid_mean_db'] for s in stats]),'map_valid_std_distribution':quant([s['valid_std_db'] for s in stats]),'valid_eq145_pixels':sum(s['valid_eq145_count'] for s in stats),'invalid_eq145_pixels':sum(s['invalid_eq145_count'] for s in stats),'invalid_not145_pixels':sum(s['invalid_not145_count'] for s in stats),'valid_pixel_distribution':histogram_summary(hists[b]['valid']),'invalid_pixel_distribution':histogram_summary(hists[b]['invalid']),'groups':{}}
  stat['valid_fraction']=stat['valid_pixels']/stat['total_pixels']
  lo,hi=meta['pathloss_range_db_per_band'][b]; vals=np.arange(65536,dtype=np.uint16).view(np.float16).astype(np.float32)
  stat['valid_pixels_outside_metadata_bounds']=int(hists[b]['valid'][(vals<lo)|(vals>hi)].sum())
  for group in ['single','multi']:
   ss=[s for s in stats if s['group']==group]
   stat['groups'][group]={'maps':len(ss),'valid_pixels':sum(s['valid_count'] for s in ss),'valid_fraction':sum(s['valid_count'] for s in ss)/(len(ss)*65536),'map_valid_mean_distribution':quant([s['valid_mean_db'] for s in ss])}
  split_report['bands'][b]=stat
  allstats.extend(stats)
  if stat['nonfinite_pixels'] or stat['nonpositive_pixels'] or stat['empty_valid_maps']:report['issues'].append(f'{split}/{b}: invalid numerical values or empty mask')
  if b=='58' and stat['invalid_not145_pixels']:report['issues'].append(f'{split}/58: masked-out pixels differ from 145 dB')
 n=pair['n']; mx=pair['sx']/n;my=pair['sy']/n;md=pair['sd']/n
 split_report['cross_band_common_valid']={'pixels':n,'pearson_all_pixels':(pair['sxy']/n-mx*my)/np.sqrt((pair['sxx']/n-mx*mx)*(pair['syy']/n-my*my)),'delta_58_minus_415_mean_db':md,'delta_58_minus_415_std_db':np.sqrt(max(0,pair['sdd']/n-md*md)),'interpretation':'descriptive correlation; shared distance/terrain can drive it; not evidence for shared-model improvement'}
 print(f'{split}: full scan complete',flush=True)
(OUT/'numeric_scan_checkpoint.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
with (OUT/'map_stats.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(allstats[0]));w.writeheader();w.writerows(allstats)
tr={r['terrain_id'] for r in indices['train']}; va={r['terrain_id'] for r in indices['val']}
report['split_checks']={'overlap_terrain_ids':sorted(tr&va),'overlap_sample_ids':sorted({r['sample_id'] for r in indices['train']}&{r['sample_id'] for r in indices['val']}),'exact_duplicate_heightmaps_across_splits':[{'train_ids':terrain_hashes['train'][h],'val_ids':terrain_hashes['val'][h]} for h in terrain_hashes['train'].keys()&terrain_hashes['val'].keys()]}
frozen={f'{int(x):05d}' for line in (DATA/'val_terrains.txt').read_text().splitlines() for x in line.split('#',1)[0].split()}
report['split_checks']['val_terrains_txt_matches_index']=frozen==va
report['test_inputs_present']=[p.name for p in DATA.glob('test*') if p.is_file()]
with (OUT/'map_stats.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(allstats[0]));w.writeheader();w.writerows(allstats)
# Check submission indexing without producing or uploading any predictions.
val_lookup={r['sample_id']:i for i,r in enumerate(indices['val'])}; subsets=Counter();badids=0;invalidrows=0;duplicate_ids=0;seen=set();nrows=0
vm={b:np.load(DATA/f'val_mask{b}.npy',mmap_mode='r',allow_pickle=False) for b in ['415','58']}
with (DATA/'sample_submission.csv').open() as f:
 reader=csv.DictReader(f); report['submission_columns']=reader.fieldnames
 for row in reader:
  nrows+=1; sid=row['ID'];duplicate_ids+=sid in seen;seen.add(sid)
  try:
   prefix,terrain,tx,pixel=sid.split('_');b=prefix[2:];sample=f'{terrain}_{tx}'; k=int(pixel);i=val_lookup[sample]
   if b not in vm or not 0<=k<65536: raise ValueError()
   subsets[(b,sample)]+=1
   invalidrows+=not bool((int(vm[b][i,k//8])>>(7-k%8))&1)
  except (ValueError,KeyError,IndexError):badids+=1
report['submission_check']={'rows':nrows,'duplicates':duplicate_ids,'unmatched_ids':badids,'rows_on_mask0':invalidrows,'maps_per_band':dict(Counter(k[0] for k in subsets)),'rows_per_map_distribution':dict(Counter(subsets.values()))}
# Byte hashes fingerprint this local source, not proof of identity to official release.
release_names=['README.md','metadata.json','manifest.json','train_index.csv','val_index.csv','val_terrains.txt','sample_submission.csv','subset_index.csv']+[f'{s}_{f}.npy' for s in ['train','val'] for f in ['hm','pl415','pl58','mask415','mask58']]
for name in release_names:
 p=DATA/name; before=p.stat(); sha=filehash(p); after=p.stat()
 report['files'][name]={'size_bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':sha,'unchanged_during_hash':before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns}
 if not report['files'][name]['unchanged_during_hash']:report['issues'].append(f'{name}: changed during fingerprinting')
 print(f'fingerprinted {name}',flush=True)
report['official_checksum_verification']='not completed: no official checksum list found among release-level files; local SHA256 values are provenance fingerprints only'
report['elapsed_seconds']=time.time()-t0
report['finished_utc']=datetime.now(timezone.utc).isoformat()
report['script_sha256']=filehash(Path(__file__))
(OUT/'raw_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'done':str(OUT/'raw_audit.json'),'issues':report['issues'],'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False),flush=True)
