"""Locked low-dimensional diagnostics; no propagation model training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,csv,time,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'audit/c19_fine_physics_20261101';D=R.parent/'data'
start=time.time()
sel=json.loads((O/'selection.json').read_text());parts=np.array(sel['parts']);cal=(parts=='calibration');conf=~cal
x=np.load(O/'physical_features.npy',mmap_mode='r');names=json.loads((O/'physical_features.json').read_text())['names'];ni={n:i for i,n in enumerate(names)}
pixels=np.load(O/'pixels.npy')
# All transformation, probe and effect-magnitude choices are fixed before outcome evaluation.
plan={'targets':['signed full residual Y-P on query-valid pixels','local curvature residual Q(Y)-Q(P) on all-five-valid pixels','log1p(QY**2) roughness magnitude'],
'Q':'centre minus average of four diagonal one-metre neighbours',
'physical_proxy':'power sum of candidate direct transmission and one-bounce diffuse surface power; not recovered actual propagation',
'models':['context','curvature_control','V_transport','H_transport_counterfactual','reset_transport_counterfactual','fixed_TE_TM_control','V_triangle0','V_triangle1','projection','arrival','registration','all_V'],
'linear_fit':'Within each input-only propagation zone; z-score using calibration only; clip standardized features to +-8; ridge n*0.01 on slopes, unpenalized intercept; no tuning, no early stopping.',
'bootstrap':'1000 terrain-paired resamples of306 confirmation terrains,seed20261112',
'gate':'>=5% occluded SSE reduction and positive95%CI; gain beyond curvature_control; corroborating physical counterfactual before training',
'limitations':'DEV previously used for selection. This is diagnostic confirmation, not official scoring or pristine holdout.'}
(O/'confirmation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
s=np.load(O/'diagnostic_samples.npz');Y=s['Y'];P=s['P'];QY=s['QY'];QP=s['QP'];ctx=s['context'];zone=s['zone'];qm=s['mask']
qmask=np.empty_like(qm);strict=np.ones_like(zone,dtype=bool)
G=np.load(R/'features/c3/train.npy',mmap_mode='r')
for b,band in enumerate(['415','58']):
 mm=np.load(D/f'train_mask{band}.npy',mmap_mode='r')
 for j,i in enumerate(sel['index_rows']):qmask[j,:,b]=np.unpackbits(mm[i],bitorder='big')[pixels[j]]
for j,i in enumerate(sel['index_rows']):
 g=np.asarray(G[i],float);e3=(2*np.sinh(3*g[:8])).max(0);e2=2*np.sinh(3*g[8]);zz=np.where(e3>=0,2,np.where(e2>=0,1,0))
 r,c=np.divmod(pixels[j],256)
 strict[j]=(zz[r-1,c-1]==zone[j])&(zz[r-1,c+1]==zone[j])&(zz[r+1,c-1]==zone[j])&(zz[r+1,c+1]==zone[j])
def q(a):return a[:,:,0]-a[:,:,1:].mean(2)
def feat(name):return np.asarray(x[...,ni[name]],float)
projection_names=['triangle_area_log_mean','triangle_area_log_std','normal_spread','direct_log_J_mean','direct_log_J_std','direct_log_J_max','direct_grazing_fraction','direct_signed_cos']
arrival_names=['surface_log_weight','surface_log_inverse_cos_mean','surface_grazing01','surface_grazing03','surface_grazing10','arrival_entropy','arrival_cos_mean','arrival_strongest_cos']
pro=np.stack([feat(n) for n in projection_names],-1)
arr=np.stack([feat(n) for n in arrival_names],-1)
a=2*np.pi*np.arange(16)/16;az=np.asarray(x[...,51:67],float)
arr=np.concatenate((arr,np.stack([(az*np.cos(a)).sum(-1),(az*np.sin(a)).sum(-1),(az*np.cos(2*a)).sum(-1),(az*np.sin(2*a)).sum(-1)],-1)),-1)
proj=np.concatenate((pro[:,:,0],q(pro)),2)
arrival=np.concatenate((arr[:,:,0],q(arr)),2)
rr,cc=np.divmod(pixels,256);dy=ctx[:,:,2];dx=ctx[:,:,3]
phase=[]
for v in [rr,cc,dy,dx]:
 phase.extend([np.cos(np.pi*v),np.sin(np.pi*v/2),np.cos(np.pi*v/2)])
phase.extend([np.cos(np.pi*(rr+cc)),np.cos(np.pi*(dy+dx))])
registration=np.stack(phase,2)
# Exact intersection crossing count and rotation have both centre and fine differences.
geo=np.stack([np.log1p((feat('triangle0_crossings')+feat('triangle1_crossings'))/2),
 (feat('triangle0_incidence_plane_turn')+feat('triangle1_incidence_plane_turn'))/2],-1)
geof=np.concatenate((geo[:,:,0],q(geo)),2)
surrogates={}
for band in ['415','58']:
 diffuse=feat('diffuse'+band)
 for pol in ['V','H','reset','fixed']:
  loss=np.stack([feat(f'triangle{d}_{pol}{band}') for d in [0,1]],-1)
  rel=np.power(10.,-np.clip(loss,0,300)/10)+diffuse[...,None]
  L=-10*np.log10(np.maximum(rel,1e-16));avg=-10*np.log10(np.maximum(rel.mean(-1),1e-16))
  surrogates[(band,pol)]=(avg,L)
weights=np.random.default_rng(20261112).multinomial(int(conf.sum()),np.full(conf.sum(),1/conf.sum()),size=1000)
zones={'all':np.ones(zone.shape,bool),'occluded_proxy':zone>0,'height_sensitive':zone==1,'nominal_blocked':zone==2,'strict_nominal_blocked':(zone==2)&strict}
results={};savepred={};fitrecord={}
def linfit(F,t,mask):
 est=np.zeros(t.shape);rec={}
 for z in [0,1,2]:
  use=cal[:,None]&mask&(zone==z);valid=mask&(zone==z)
  A=F[use];y=t[use];mu=A.mean(0);sd=A.std(0);sd=np.where(sd>1e-6,sd,1)
  X=np.c_[np.ones(len(A)),np.clip((A-mu)/sd,-8,8)]
  pen=np.eye(X.shape[1])*len(A)*.01;pen[0,0]=0
  beta=np.linalg.solve(X.T@X+pen,X.T@y)
  XX=np.c_[np.ones(valid.sum()),np.clip((F[valid]-mu)/sd,-8,8)]
  est[valid]=XX@beta
  rec[str(z)]={'n':int(use.sum()),'mean':mu.tolist(),'sd':sd.tolist(),'beta':beta.tolist()}
 return est,rec
def stats(err,base,control,mask):
 ans={}
 for zn,z in zones.items():
  m=mask&z;n=m.sum(1);ss=(err*err*m).sum(1);bs=(base*base*m).sum(1);cs=(control*control*m).sum(1)
  cf=conf;wc=weights@ss[cf];wb=weights@bs[cf];wcc=weights@cs[cf]
  ans[zn]={'calibration_RMSE':float(np.sqrt(ss[cal].sum()/n[cal].sum())),'confirmation_RMSE':float(np.sqrt(ss[cf].sum()/n[cf].sum())),
   'confirmation_n':int(n[cf].sum()),'SSE_reduction_percent':float(100*(1-ss[cf].sum()/bs[cf].sum())),
   'terrain_bootstrap95':np.percentile(100*(1-wc/wb),[2.5,97.5]).tolist(),
   'beyond_curvature_control_SSE_reduction_percent':float(100*(1-ss[cf].sum()/cs[cf].sum())),
   'beyond_control_bootstrap95':np.percentile(100*(1-wc/wcc),[2.5,97.5]).tolist(),
   'terrain_sse':ss[cf].tolist(),'terrain_n':n[cf].tolist()}
 return ans
# Fixed feature definitions and source hashes allow later review of every calibration.
design_names={}
for b,band in enumerate(['415','58']):
 distance=ctx[:,:,4];fs=20*np.log10(4*np.pi*distance*float({'415':415e6,'58':5.775e9}[band])/299792458.)
 context=np.stack([np.log(distance),np.arcsinh(ctx[:,:,5]),np.arcsinh(ctx[:,:,6]),(P[:,:,b]-fs)/10],2)
 control=np.concatenate((context,QP[:,:,b,None]),2)
 designs={'context':context,'curvature_control':control}
 for pol,name in [('V','V_transport'),('H','H_transport_counterfactual'),('reset','reset_transport_counterfactual'),('fixed','fixed_TE_TM_control')]:
  L,sep=surrogates[(band,pol)]
  f=np.stack([L[:,:,0]/10,q(L),abs(sep[:,:,0,0]-sep[:,:,0,1])/10],2)
  designs[name]=np.concatenate((control,geof,f),2)
 for d in [0,1]:
  L=surrogates[(band,'V')][1][...,d]
  designs[f'V_triangle{d}']=np.concatenate((control,geof,np.stack((L[:,:,0]/10,q(L)),2)),2)
 designs.update(projection=np.concatenate((control,proj),2),arrival=np.concatenate((control,arrival),2),registration=np.concatenate((control,registration),2))
 designs['all_V']=np.concatenate((designs['V_transport'],proj,arrival),2)
 design_names[band]={n:int(f.shape[-1]) for n,f in designs.items()}
 for target in ['signed_residual','curvature_residual','roughness_magnitude']:
  if target=='signed_residual':t=Y[:,:,b]-P[:,:,b];mask=qmask[:,:,b]
  elif target=='curvature_residual':t=QY[:,:,b]-QP[:,:,b];mask=qm[:,:,b]
  else:t=np.log1p(QY[:,:,b]**2);mask=qm[:,:,b]
  fits={}
  for name,F in designs.items():
   fit,rec=linfit(F,t,mask);fits[name]=fit;fitrecord[f'{band}_{target}_{name}']=rec
   savepred[f'{band}_{target}_{name}']=fit.astype('float32')
  # No-correction reference for residuals; zone mean for magnitude to make R2 informative.
  if target=='roughness_magnitude':
   base=np.zeros_like(t)
   for z in [0,1,2]:base[zone==z]=t[cal[:,None]&mask&(zone==z)].mean()
   base=t-base
  else:base=t
  results[f'{band}_{target}']={'baseline':stats(base,base,t-fits['curvature_control'],mask),
   **{n:stats(t-f,base,t-fits['curvature_control'],mask) for n,f in fits.items()}}
  print(json.dumps({'band':band,'target':target,'confirmation_occluded':{n:{k:v['occluded_proxy'][k] for k in ['confirmation_RMSE','SSE_reduction_percent','beyond_curvature_control_SSE_reduction_percent']} for n,v in results[f'{band}_{target}'].items()}}),flush=True)
# Curvature/error decomposition and associations without claiming causal explained variance.
decomp={}
for b,band in enumerate(['415','58']):
 decomp[band]={}
 for zn,z in zones.items():
  m=conf[:,None]&qm[:,:,b]&z
  yy=QY[:,:,b][m];pp=QP[:,:,b][m];ee=yy-pp
  decomp[band][zn]={'n':int(m.sum()),'QY_RMS':float(np.sqrt(np.mean(yy**2))),'QP_RMS':float(np.sqrt(np.mean(pp**2))),
   'Qresidual_RMS':float(np.sqrt(np.mean(ee**2))),'correlation_QY_QP':float(np.corrcoef(yy,pp)[0,1]),
   'QY_QP_cross_moment':float(np.mean(yy*pp))}
# Conditional raw physics associations and directions of target variation.
assoc={}
factors={'receiver_normal_spread':feat('normal_spread')[:,:,0],
 'receiver_direct_grazing':feat('direct_grazing_fraction')[:,:,0],
 'receiver_surface_grazing':feat('surface_grazing03')[:,:,0],
 'arrival_entropy':feat('arrival_entropy')[:,:,0],
 'facet_turn':geo[:,:,0,1]}
for b,band in enumerate(['415','58']):
 lv,sep=surrogates[(band,'V')]
 ff={**factors,'triangulation_power_disagreement':abs(sep[:,:,0,0]-sep[:,:,0,1]),'V_physics_local_curvature':abs(q(lv))}
 for z in [1,2]:
  for name,f in ff.items():
   m=cal[:,None]&qm[:,:,b]&(zone==z);edges=np.unique(np.quantile(f[m],[.25,.5,.75]));out=[]
   for j in range(len(edges)+1):
    mm=conf[:,None]&qm[:,:,b]&(zone==z)&(np.searchsorted(edges,f,side='right')==j)
    out.append({'n':int(mm.sum()),'feature_mean':float(np.mean(f[mm])) if mm.any() else None,
      'QY_RMS':float(np.sqrt(np.mean(QY[:,:,b][mm]**2))) if mm.any() else None,
      'Qresidual_RMS':float(np.sqrt(np.mean((QY[:,:,b]-QP[:,:,b])[mm]**2))) if mm.any() else None,
      'residual_RMSE':float(np.sqrt(np.mean((Y[:,:,b]-P[:,:,b])[mm]**2))) if mm.any() else None})
   assoc[f'{band}_zone{z}_{name}']={'calibration_quantile_edges':edges.tolist(),'confirmation_bins':out}
np.savez(O/'probe_predictions.npz',**savepred)
np.savez(O/'evaluation_masks.npz',query_valid=qmask,five_valid=qm,strict_state=strict)
(O/'fit_coefficients.json').write_text(json.dumps(fitrecord,indent=2)+'\n')
report={'state':'completed','seconds':time.time()-start,'plan':plan,'feature_dimensions':design_names,'statistics':results,'curvature_decomposition':decomp,'conditional_associations':assoc,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'official_VAL_not_used':True}
(O/'confirmation_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'state':'completed','seconds':time.time()-start}),flush=True)
