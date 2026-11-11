"""Frozen point/cell operator contrast, small calibration only, terrain confirmation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,time
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'audit/c20_cell_observation_20261101';S=R/'audit/c19_fine_physics_20261101'
meta=json.loads((O/'observations.json').read_text());assert meta['state']=='completed'
sel=json.loads((O/'selection.json').read_text());cal=np.array(sel['parts'])=='calibration';conf=~cal;N=sel['pixels_per_map']
plan={'primary':'triangle0_area16, linear power area average then linear power centroid interpolation, coefficient[0,1] per published-input zone',
 'controls':['centroid-only interpolation','area-average increment only','triangle1 sensitivity','dB interpolation export sensitivity','constant received diffuse background','exact fivepoint kernels on frozenC10 as postprocessing only','C10 local-curvature/context calibration'],
 'delta_clip_dB':30,'no_outcome_based_model_selection':True,'secondary':'Unconstrained signed slope and per-pixel privileged best correction are diagnostics only; never promote them as an area-average mechanism.',
 'numerical_limit':'One-bounce high-frequency angular quadrature is not converged at rare valid points. Full-surrogate high-frequency changes cannot by themselves support physical attribution.',
 'official_VAL_not_used':True}
(O/'evaluation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
s=np.load(S/'diagnostic_samples.npz');mask=np.load(S/'evaluation_masks.npz');Y=s['Y'][:,:N];P=s['P'][:,:N];zone=s['zone'][:,:N];ctx=s['context'][:,:N];QP=s['QP'][:,:N]
qm=mask['query_valid'][:,:N];strict=mask['strict_state'][:,:N]
L=np.asarray(np.load(O/'operator_loss.npy'),float);CP=np.load(O/'component_powers.npy',mmap_mode='r');post=np.asarray(np.load(O/'C10_fivepoint_controls.npy'),float);ops=meta['operator_names'];oi={n:i for i,n in enumerate(ops)}
vis=np.asarray(CP[:,:,:,3,4]);partial=((vis>1e-6)&(vis<1-1e-6)).any(2);dark=(vis<1e-6).all(2)
zones={'all':np.ones(zone.shape,bool),'occluded_proxy':zone>0,'height_sensitive':zone==1,'nominal_blocked':zone==2,'strict_nominal_blocked':(zone==2)&strict,
 'candidate_partial_LOS':partial,'candidate_no_LOS':dark&(zone>0)}
weights=np.random.default_rng(20261117).multinomial(conf.sum(),np.full(conf.sum(),1/conf.sum()),1000)
def stats(error,base,control,valid):
 ans={}
 for zn,z in zones.items():
  m=z&valid;nn=m.sum(1);ss=(error*error*m).sum(1);bs=(base*base*m).sum(1);cs=(control*control*m).sum(1);n=nn[conf].sum()
  if n==0:continue
  wb=weights@bs[conf];ws=weights@ss[conf];wc=weights@cs[conf]
  ans[zn]={'cal_RMSE':float(np.sqrt(ss[cal].sum()/max(nn[cal].sum(),1))),'confirm_RMSE':float(np.sqrt(ss[conf].sum()/n)),'confirm_n':int(n),
   'SSE_gain_percent':float(100*(1-ss[conf].sum()/bs[conf].sum())),'SSE_gain95':np.percentile(100*(1-ws/wb),[2.5,97.5]).tolist(),
   'extra_control_gain_percent':float(100*(1-ss[conf].sum()/cs[conf].sum())),'extra_control95':np.percentile(100*(1-ws/wc),[2.5,97.5]).tolist(),
   'terrain_sse':ss[conf].tolist(),'terrain_n':nn[conf].tolist()}
 return ans
def context_control(b):
 dist=ctx[:,:,4];f=[415e6,5.775e9][b];fs=20*np.log10(4*np.pi*dist*f/299792458.)
 X=np.stack((np.log(dist),np.arcsinh(ctx[:,:,5]),np.arcsinh(ctx[:,:,6]),(P[:,:,b]-fs)/10,QP[:,:,b]),2);error=Y[:,:,b]-P[:,:,b];fit=np.zeros(zone.shape);pars={}
 for z in [0,1,2]:
  m=cal[:,None]&qm[:,:,b]&(zone==z);mu=X[m].mean(0);sd=X[m].std(0);sd=np.maximum(sd,1e-6)
  xx=np.c_[np.ones(m.sum()),np.clip((X[m]-mu)/sd,-8,8)];pen=np.eye(6)*m.sum()*.01;pen[0,0]=0
  beta=np.linalg.solve(xx.T@xx+pen,xx.T@error[m]);allz=zone==z;xx=np.c_[np.ones(allz.sum()),np.clip((X[allz]-mu)/sd,-8,8)]
  fit[allz]=xx@beta;pars[str(z)]={'n':int(m.sum()),'mean':mu.tolist(),'sd':sd.tolist(),'beta':beta.tolist()}
 return fit,pars
def scalar(delta,target,b,positive=True):
 delta=np.clip(delta,-30,30);fit=np.zeros(zone.shape);rec={}
 for z in [0,1,2]:
  m=cal[:,None]&qm[:,:,b]&(zone==z);d=delta[m];e=target[m]
  free=float(np.dot(d,e)/(np.dot(d,d)+1e-12));beta=float(np.clip(free,0,1) if positive else np.clip(free,-1,1))
  fit[zone==z]=beta*delta[zone==z];rec[str(z)]={'n':int(m.sum()),'unconstrained_beta':free,'used_beta':beta}
 return fit,rec
result={};coefs={};saved={};distribution={};rawscores={};oracle={};start=time.time()
for b,band in enumerate(['415','58']):
 E=Y[:,:,b]-P[:,:,b];ctrl,coefs[band+'_context']=context_control(b);models={'C10':np.zeros_like(E),'curvature_control':ctrl,
 'C10_fivepoint_dB':post[:,:,0,b]-P[:,:,b],'C10_fivepoint_power':post[:,:,1,b]-P[:,:,b]}
 delta={}
 for d in [0,1]:
  point=L[:,:,d,oi['point'],b];cent=L[:,:,d,oi['centroid'],b]
  for name in ['centroid','area4','area16','area16_dbinterp','area16_fixed_diffuse']:
   delta[f'triangle{d}_{name}']=L[:,:,d,oi[name],b]-point
  delta[f'triangle{d}_area_increment']=L[:,:,d,oi['area16'],b]-cent
  for name in ['point','centroid','area4','area16','area16_dbinterp']:
   rawscores[f'{band}_triangle{d}_{name}']=stats(Y[:,:,b]-L[:,:,d,oi[name],b],E,E-ctrl,qm[:,:,b])
 for name,d in delta.items():
  correction,rec=scalar(d,E,b);models[name]=correction;coefs[band+'_'+name]=rec
  extra,rec=scalar(d,E-ctrl,b);models[name+'_after_control']=ctrl+extra;coefs[band+'_'+name+'_after_control']=rec
  correction,rec=scalar(d,E,b,positive=False);models[name+'_signed_sensitivity']=correction;coefs[band+'_'+name+'_signed_sensitivity']=rec
  distribution[band+'_'+name]={}
  for zn,z in zones.items():
   mm=conf[:,None]&qm[:,:,b]&z
   distribution[band+'_'+name][zn]={'n':int(mm.sum()),'median_delta_dB':float(np.median(d[mm])),'abs_p95_dB':float(np.percentile(abs(d[mm]),95)),
    'abs_delta_gt1_fraction':float(np.mean(abs(d[mm])>1)),'abs_delta_gt30_fraction':float(np.mean(abs(d[mm])>30)),
    'SSE_share_at_abs_delta_gt1':float(np.sum(E[mm&(abs(d)>1)]**2)/np.sum(E[mm]**2))}
 result[band]={n:stats(E-c,E,E-ctrl,qm[:,:,b]) for n,c in models.items()}
 for name,c in models.items():saved[band+'_'+name]=c.astype('float32')
 # Truth-using upper bound within the stated fixed-diffuse,positive[0,1],two-mesh family only.
 for family in ['area16','area16_fixed_diffuse']:
  best=E*E
  for d in [0,1]:
   de=np.clip(delta[f'triangle{d}_{family}'],-30,30)
   beta=np.clip(E/np.where(abs(de)>1e-12,de,1),0,1);beta[abs(de)<=1e-12]=0
   best=np.minimum(best,(E-beta*de)**2)
  oracle[band+'_'+family]=stats(np.sqrt(best),E,E-ctrl,qm[:,:,b])
 print(json.dumps({'band':band,'occluded':{n:{k:result[band][n]['occluded_proxy'][k] for k in ['confirm_RMSE','SSE_gain_percent','extra_control_gain_percent']} for n in ['C10','curvature_control','C10_fivepoint_dB','C10_fivepoint_power','triangle0_centroid','triangle0_area16','triangle0_area_increment','triangle0_area16_fixed_diffuse','triangle0_area16_after_control']}}),flush=True)
# Comparison of full cell operator and centroid-only, paired on the same query points.
paired={}
for band in ['415','58']:
 paired[band]={}
 for z in zones:
  base=np.array(result[band]['triangle0_centroid'][z]['terrain_sse']);new=np.array(result[band]['triangle0_area16'][z]['terrain_sse'])
  paired[band][z]={'SSE_gain_beyond_centroid_percent':float(100*(1-new.sum()/base.sum())),'CI95':np.percentile(100*(1-(weights@new)/(weights@base)),[2.5,97.5]).tolist()}
np.savez(O/'calibrated_corrections.npz',**saved)
(O/'calibration_coefficients.json').write_text(json.dumps(coefs,indent=2)+'\n')
report={'state':'completed','seconds':time.time()-start,'plan':plan,'statistics':result,'raw_surrogate_scores':rawscores,'operator_delta_distribution':distribution,
 'privileged_upper_bound_within_surrogate_family':oracle,'area_vs_centroid_paired':paired,'candidate_partial_LOS_fraction_all_queries':float(partial.mean()),
 'no_new_competition_model_training':True}
(O/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':'completed','seconds':time.time()-start}),flush=True)
