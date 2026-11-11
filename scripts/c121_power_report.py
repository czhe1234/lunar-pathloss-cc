"""Evaluate input-only physics probes after feature generation. Diagnostic, no fitting."""
import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.c121_pairs import Reader
A=R/'audit/c121_pairs';rd=Reader();lookup=np.load(R/'src/c70_centroid_lookup0.npz');ids=lookup['ids'];w=lookup['weights']
protocol=json.load(open(A/'power_probe_protocol.json'));pairs=json.load(open(A/'monitor/pairs.json'))
P=R/'audit/c101_residual/monitor';info=json.load(open(P/'done.json'));mi={i:j for j,i in enumerate(info['ids'])};pp=info['points_per_map'];pix=np.load(P/'pixel.npy',mmap_mode='r');target=np.load(P/'target.npy',mmap_mode='r');base=np.load(P/'base.npy',mmap_mode='r')
available=[n for n in [128,1024,8192,32768] if (A/f'power{n}/done.json').exists()]
arrays={};replay=[]
for i in protocol['maps']:
 arrays[i]={}
 for n in available:
  raw=np.load(A/f'power{n}/{i}.npy').astype('float64');arrays[i][n]=raw
 if i in rd.native:
  px=np.array(pix[mi[i]*pp:(mi[i]+1)*pp]);v,meta=rd.donors(i,px)
  raw=arrays[i][128];recreated=np.log10(raw[ids[px]].clip(1e-12,1e8))/4
  mean=(recreated*w[px,:,None]).sum(1);cached=(v[:,:,27:37]*meta[:,:3,None]).sum(1)
  replay.append(float(np.max(abs(mean-cached))))
out={'scope':protocol,'available_directions':available,'clipped_replay_max_abs':max(replay),'bands':[]}
for b in range(2):
 measurements={};pair_measure={};target_contrast=[];residual_contrast=[];is_clipped=[]
 for n in available:
  metrics=[]
  for i in protocol['maps']:
   px=np.array(pix[mi[i]*pp:(mi[i]+1)*pp]);sl=slice(mi[i]*pp,(mi[i]+1)*pp);y=np.array(target[sl,b]);valid=np.array(target[sl,2+b])>0
   raw=arrays[i][n];q=(raw[ids[px],6+b]*w[px]).sum(1);log=-10*np.log10(q.clip(1e-38))
   old=(np.clip(arrays[i][128][ids[px],6+b],1e-12,None)*w[px]).sum(1);oldlog=-10*np.log10(old)
   metrics.append(np.c_[log[valid],oldlog[valid],(q[valid]>1e-12),np.full(valid.sum(),i)])
  measurements[n]=np.concatenate(metrics)
  vals=[]
  for r in pairs:
   i=r['mapid']
   if r['band']!=b or i not in arrays:continue
   px=np.array([r['anchor_pixel'],r['opposite_pixel']]);raw=arrays[i][n];q=(raw[ids[px],6+b]*w[px]).sum(1);loss=-10*np.log10(q.clip(1e-38));delta=loss[0]-loss[1]
   vals.append(delta)
   if n==128:
    xx,_=rd.old(i,px);yy=np.array(r['target'])[:2,b];bb=np.array(r['base'])[:2,b]
    target_contrast.append(float((yy-xx[:,96+b])[0]-(yy-xx[:,96+b])[1]));residual_contrast.append(float(r['anchor_residual']-r['opposite_residual']))
    is_clipped.append(bool(np.all((np.clip(raw[ids[px],6+b],1e-12,None)*w[px]).sum(1)<=1.01e-12)))
  pair_measure[n]=np.array(vals)
 target_contrast=np.array(target_contrast);clip=np.array(is_clipped)
 bc={'band':b,'pairs':len(target_contrast),'clipped_pairs':int(clip.sum()),'probe_metrics':{},'convergence':{}}
 for n in available:
  d=pair_measure[n];x=measurements[n]
  bc['probe_metrics'][str(n)]={'valid_sampled_points':len(x),'unclipped_power_fraction':float(x[:,2].mean()),
  'pairs_same_sign_as_target_contrast':float(np.mean(d*target_contrast>0)),
  'paired_contrast_rmse_db':float(np.sqrt(np.mean((d-target_contrast)**2))),
  'clipped_pairs_correct_contrast':float(np.mean(d[clip]*target_contrast[clip]>0)) if clip.any() else None,
  'clipped_pairs_median_abs_contrast_db':float(np.median(abs(d[clip]))) if clip.any() else None,
  'contrast_correlation':float(np.corrcoef(d,target_contrast)[0,1])}
 for n,m in zip(available,available[1:]):
  d=abs(measurements[n][:,0]-measurements[m][:,0]);bc['convergence'][f'{n}_to_{m}']={'median_abs_change_db':float(np.median(d)),'p90_abs_change_db':float(np.quantile(d,.9)),'gt1db_fraction':float((d>1).mean()),'rms_change_db':float(np.sqrt(np.mean(d*d)))}
 out['bands'].append(bc)
(A/'power_report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='scope'},indent=2))
