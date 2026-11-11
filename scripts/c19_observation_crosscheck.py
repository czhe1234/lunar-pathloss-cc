"""Descriptive observation checks. Masks are used only for audit/metrics, never prediction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv
import numpy as np
from scipy.ndimage import distance_transform_edt
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c19_fine_physics import transport
from src.c12_material_features import slab
O=R/'audit/c19_fine_physics_20261101';D=R.parent/'data'
sel=json.loads((O/'selection.json').read_text());conf=np.array(sel['parts'])=='confirmation'
s=np.load(O/'diagnostic_samples.npz');p=np.load(O/'pixels.npy');m=np.load(O/'evaluation_masks.npz');x=np.load(O/'physical_features.npy',mmap_mode='r')
zone=s['zone'];QY=s['QY'];QP=s['QP'];Y=s['Y'];P=s['P'];five=m['five_valid'];query=m['query_valid'];strict=m['strict_state']
dist=np.empty((*p.shape,2));rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r')
for b,band in enumerate(['415','58']):
 masks=np.load(D/f'train_mask{band}.npy',mmap_mode='r')
 for j,i in enumerate(sel['index_rows']):
  mask=np.unpackbits(masks[i],bitorder='big').reshape(256,256).astype(bool)
  d=distance_transform_edt(mask) if not mask.all() else np.full((256,256),np.inf)
  dist[j,:,b]=d.ravel()[p[j]]
np.save(O/'distance_to_invalid_diagnostic_only.npy',dist)
zones={'occluded_proxy':zone>0,'height_sensitive':zone==1,'nominal_blocked':zone==2,'strict_nominal_blocked':(zone==2)&strict}
cross={}
for zn,z in zones.items():
 mm=conf[:,None]&five.all(2)&z
 out={'n':int(mm.sum())}
 for n,v in [('QY',QY),('QP',QP),('Qresidual',QY-QP),('fullresidual',Y-P)]:
  vv=v[mm];c=np.corrcoef(vv.T)[0,1];out[n]={'cross_band_correlation':float(c),'squared_correlation_percent':float(c*c*100),'band_rms':np.sqrt(np.mean(vv*vv,0)).tolist()}
 cross[zn]=out
hole={}
edges=[0,2,4,8,16,np.inf]
for b,band in enumerate(['415','58']):
 hole[band]={}
 for zn,z in zones.items():
  rec=[]
  for lo,hi in zip(edges[:-1],edges[1:]):
   mq=conf[:,None]&query[:,:,b]&z&(dist[:,:,b]>lo)&(dist[:,:,b]<=hi)
   mf=mq&five[:,:,b]
   rec.append({'distance_lower_exclusive_m':lo,'distance_upper_inclusive_m':None if hi==np.inf else hi,'n_query':int(mq.sum()),'n_five_valid':int(mf.sum()),
    'RMSE':float(np.sqrt(np.mean((Y[:,:,b]-P[:,:,b])[mq]**2))) if mq.any() else None,
    'QY_RMS':float(np.sqrt(np.mean(QY[:,:,b][mf]**2))) if mf.any() else None,
    'Qresidual_RMS':float(np.sqrt(np.mean((QY[:,:,b]-QP[:,:,b])[mf]**2))) if mf.any() else None,
    'query_sse':float(np.sum((Y[:,:,b]-P[:,:,b])[mq]**2))})
  hole[band][zn]=rec
physics={}
for b,band in enumerate(['415','58']):
 te,tm,_,_=slab(np.array([1.,.5,.1]),b)
 physics[band]={'single_slab_cosines':[1.,.5,.1],'TE_loss_dB':(-10*np.log10(te)).tolist(),'TM_loss_dB':(-10*np.log10(tm)).tolist()}
 off=6+4*b;loss=np.asarray(x[:,:,0,[off,off+16]],float);T=np.power(10.,-np.clip(loss,0,300)/10).mean(-1);Dproxy=np.asarray(x[:,:,0,49+b],float)
 frac=T/(T+Dproxy+1e-30);disagree=np.asarray(x[:,:,0,0])!=np.asarray(x[:,:,0,16])
 physics[band]['proxy_transmission_share']={}
 for zn,z in zones.items():
  mm=conf[:,None]&query[:,:,b]&z
  physics[band]['proxy_transmission_share'][zn]={'n':int(mm.sum()),'fraction_T_over_TplusD_gt_half':float(np.mean(frac[mm]>.5)),
  'median_share':float(np.median(frac[mm])),'triangle_crossing_count_disagreement_fraction':float(np.mean(disagree[mm]))}
# General rotation invariance, using non-axis-aligned rays and nonsymmetric published terrain.
r=rows[sel['index_rows'][17]];h=np.asarray(H[int(r['hm_row'])],float);tr=int(r['tx_row']);tc=int(r['tx_col']);pix=p[17,:64]
a=transport(h,tr,tc,pix).reshape(-1,2,16);rr,cc=np.divmod(pix,256)
b=transport(np.rot90(h),255-tc,tr,(255-cc)*256+rr).reshape(-1,2,16)[:,::-1]
delta=float(np.max(abs(a-b)));assert delta<1e-7,delta
out={'state':'completed','purpose':'Descriptive follow-up after primary confirmation results; not a new promotion test.',
 'cross_band':cross,'distance_to_invalid':hole,'physics_parameter_sensitivity':physics,
 'general_rotation_max_abs_difference':delta,'limitations':['Missing-pixel distance uses privileged masks only for diagnosis. It is not a permissible model input.',
 'Hole association does not separate Monte Carlo sampling, multi-bounce propagation, and interpolation.',
 'Cross-frequency correlation is on common-five-valid pixels. Its square is descriptive, not an achievable prediction gain.',
 'T/(T+D) is a reconstructed one-bounce proxy, not actual ray-power attribution.']}
(O/'observation_crosscheck.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
