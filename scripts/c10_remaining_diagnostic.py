"""Read-only residual accounting for the complete, frozen C10 prediction arrays."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
a=argparse.ArgumentParser();a.add_argument('--predictions',required=True);a.add_argument('--output',required=True);args=a.parse_args()
P=R/args.predictions;out=(R/args.output).resolve();assert out.is_relative_to(R)
mapping=json.loads((P/'indices.json').read_text());score=json.loads((P/'independent_score.json').read_text());assert score['complete_requested_split']
split=mapping['source_split'];ids=mapping['index_rows'];rows=list(csv.DictReader((D/f'{split}_index.csv').open()));hm=np.load(D/f'{split}_hm.npy',mmap_mode='r')
bs=['415','58'];pred=[np.load(P/f'pl{b}.npy',mmap_mode='r') for b in bs];target=[np.load(D/f'{split}_pl{b}.npy',mmap_mode='r') for b in bs];masks=[np.load(D/f'{split}_mask{b}.npy',mmap_mode='r') for b in bs]
N=np.zeros(2,np.int64);S=np.zeros(2);E=np.zeros(2);tailN=np.zeros((2,2),np.int64);tailS=np.zeros((2,2));blockS=np.zeros(2);stateN=np.zeros((2,3,3),np.int64);stateS=np.zeros((2,3,3));jointN=0;jointS=np.zeros(2);commonS=0.;diffS=0.
yy,xx=np.mgrid[:256,:256];freq=np.array([415e6,5.775e9]);t0=time.time()
for j,i in enumerate(ids):
 row=rows[i];h=hm[int(row['hm_row'])].astype(float);tr=int(row['tx_row']);tc=int(row['tx_col']);dist=np.sqrt((xx-tc)**2+(yy-tr)**2+(h+1-h[tr,tc]-3)**2).clip(.25);fs=20*np.log10(4*np.pi*dist[None]*freq[:,None,None]/299792458.)
 errors=[];valid=[]
 for k in range(2):
  m=np.unpackbits(masks[k][i],bitorder='big').reshape(256,256).astype(bool);y=target[k][i].astype(float);p=pred[k][j].astype(float);e=p-y;errors.append(e);valid.append(m);s=e*e
  N[k]+=m.sum();S[k]+=s[m].sum();E[k]+=e[m].sum()
  for q,th in enumerate([5,10]):
   keep=m&(abs(e)>th);tailN[k,q]+=keep.sum();tailS[k,q]+=s[keep].sum()
  bn=m.reshape(32,8,32,8).sum((1,3));be=np.where(m,e,0).reshape(32,8,32,8).sum((1,3));blockS[k]+=np.divide(be*be,bn,out=np.zeros_like(be),where=bn>0).sum()
  yt=np.digitize(y-fs[k],[3,10]);pt=np.digitize(p-fs[k],[3,10]);code=(yt*3+pt)[m];stateN[k]+=np.bincount(code,minlength=9).reshape(3,3);stateS[k]+=np.bincount(code,weights=s[m],minlength=9).reshape(3,3)
 joint=valid[0]&valid[1];e0=errors[0][joint];e1=errors[1][joint];jointN+=joint.sum();jointS+=np.array([e0@e0,e1@e1]);common=(e0+e1)/2;diff=(e1-e0)/2;commonS+=common@common;diffS+=diff@diff
 if j%512==0:print(json.dumps({'maps_done':j+1,'total':len(ids),'seconds':time.time()-t0}),flush=True)
result={'prediction_directory':str(P),'scope':'All requested maps and valid pixels; accounting of frozen errors, not causal attribution or model updates','maps_per_band':len(ids),'bands':{},'joint_valid_pixels':int(jointN),'common_residual_energy_fraction':float(commonS/(commonS+diffS)),'dual_band_difference_error_RMSE_db':float(np.sqrt(4*diffS/jointN)),'state_definition':'extra loss relative to nominal TX3/RX1 FSPL: <3, 3-10, >=10 dB; labels and prediction binned separately. Not actual propagation state.','seconds':time.time()-t0,'no_training_or_GPU':True}
for k,b in enumerate(bs):
 assert N[k]==score['bands'][b]['valid_pixels'] and abs(np.sqrt(S[k]/N[k])-score['bands'][b]['rmse_db'])<1e-8
 assert stateN[k].sum()==N[k] and abs(stateS[k].sum()/S[k]-1)<1e-10
 result['bands'][b]={'valid_pixels':int(N[k]),'RMSE_db':float(np.sqrt(S[k]/N[k])),'mean_error_db':float(E[k]/N[k]),'RMSE_if_all_errors_over_10db_were_perfectly_removed':float(np.sqrt((S[k]-tailS[k,1])/N[k])),'8x8_residual_mean_projection_SSE_fraction':float(blockS[k]/S[k]),'tails':{str(th):{'pixel_fraction':float(tailN[k,q]/N[k]),'SSE_fraction':float(tailS[k,q]/S[k])} for q,th in enumerate([5,10])},'state_pair_counts':stateN[k].tolist(),'state_pair_SSE_fraction':(stateS[k]/S[k]).tolist(),'class_mismatch_SSE_fraction':float((S[k]-np.trace(stateS[k]))/S[k]),'both_high_loss_SSE_fraction':float(stateS[k,2,2]/S[k]),'both_high_loss_RMSE_db':float(np.sqrt(stateS[k,2,2]/stateN[k,2,2]))}
assert abs((2*commonS+2*diffS)/jointS.sum()-1)<1e-10
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
