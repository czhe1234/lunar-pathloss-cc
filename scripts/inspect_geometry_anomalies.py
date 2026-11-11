"""Visual inspection of previously selected discovery anomalies; geometry only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
s=importlib.util.spec_from_file_location('alignment',Path(__file__).with_name('diagnose_geometry_alignment.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
examples=json.loads((a.OUT/'label_model_consistency.json').read_text())['examples'];records=[]
for num,ex in enumerate(examples):
 i=ex['original_row'];j=ex['prediction_row'];r=a.rows[i];h=a.H[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);rr=ex['receiver_row'];cc=ex['receiver_col'];t=np.linspace(0,1,2049);profile=a.interpolate(h,tr+(rr-tr)*t,tc+(cc-tc)*t);dz=h[rr,cc]+1.-h[tr,tc]-3.;line=h[tr,tc]+3+dz*t;line2=h[tr,tc]+2+(dz+1)*t;distance=np.hypot(rr-tr,cc-tc)*t
 gr,gc=np.meshgrid(np.arange(2,256,4),np.arange(2,256,4),indexing='ij');e3,d=a.profile(h,tr,tc,gr.ravel(),gc.ravel(),n=384);e2,_=a.profile(h,tr,tc,gr.ravel(),gc.ravel(),tx_height=2.,n=384);e3=e3.reshape(64,64);e2=e2.reshape(64,64)
 yr,yc=np.meshgrid(np.arange(256),np.arange(256),indexing='ij');dd=np.sqrt((yr-tr)**2+(yc-tc)**2+(h+1.-h[tr,tc]-3.)**2);ref=20*np.log10(4*np.pi*dd[None]/a.wavelength[:,None,None]);valid=np.stack([np.unpackbits(b[i],bitorder='big').reshape(256,256).astype(bool) for b in a.M]);ys=np.stack([b[i].astype(np.float64) for b in a.Y]);ps=np.stack([b[j].astype(np.float64) for b in a.P])
 fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True);fig.suptitle(f'Original row {i}; TX ({tr},{tc}); RX ({rr},{cc})',fontsize=14)
 im=ax[0,0].imshow(h,cmap='terrain');ax[0,0].plot([tc,cc],[tr,rr],'k-');ax[0,0].scatter([tc,cc],[tr,rr],c=['red','cyan']);ax[0,0].set_title('Terrain and inspected segment');fig.colorbar(im,ax=ax[0,0],shrink=.8,label='m')
 ax[0,1].plot(distance,profile,label='Terrain');ax[0,1].plot(distance,line,label='TX 3m / RX 1m');ax[0,1].plot(distance,line2,label='TX 2m / RX 1m',ls='--');ax[0,1].legend(fontsize=8);ax[0,1].set_title('Height profile');ax[0,1].set_xlabel('Horizontal distance (m)')
 ax[0,2].plot(distance,line-profile,label='3m clearance');ax[0,2].plot(distance,line2-profile,label='2m clearance');ax[0,2].axhline(0,c='k',lw=.8);ax[0,2].legend(fontsize=8);ax[0,2].set_title('Clearance along inspected segment');ax[0,2].set_ylim(-1,5)
 for k in (0,1):
  z=np.where(valid[k],ys[k]-ref[k],np.nan);im=ax[1,k].imshow(z,cmap='magma',vmin=-3,vmax=35);ax[1,k].contour(gc,gr,e3,levels=[0],colors=['cyan'],linewidths=.7);ax[1,k].contour(gc,gr,e2,levels=[0],colors=['lime'],linewidths=.7);ax[1,k].scatter([tc,cc],[tr,rr],c=['red','white'],s=16);ax[1,k].set_title(('415 MHz' if k==0 else '5.8 GHz')+' label excess; cyan:3m, green:2m');fig.colorbar(im,ax=ax[1,k],shrink=.8,label='dB')
 im=ax[1,2].imshow(np.where(valid[1],ps[1]-ys[1],np.nan),cmap='RdBu_r',vmin=-25,vmax=25);ax[1,2].scatter([cc],[rr],c='k',s=16);ax[1,2].set_title('B0 prediction - label, 5.8 GHz');fig.colorbar(im,ax=ax[1,2],shrink=.8,label='dB')
 name=f'anomaly_{num+1}';fig.savefig(a.OUT/(name+'.png'),dpi=150);plt.close(fig);Image.open(a.OUT/(name+'.png')).convert('RGB').resize((1400,800)).save(a.OUT/(name+'_preview.jpg'),quality=75)
 records.append({**ex,'tx_row':tr,'tx_col':tc,'min_clearance3':float((line-profile)[1:-1].min()),'min_clearance2':float((line2-profile)[1:-1].min()),'profile_distance_m':float(distance[-1]),'figure':name+'.png'})
a.write('anomaly_inspection.json',records);print(json.dumps(records),flush=True)
