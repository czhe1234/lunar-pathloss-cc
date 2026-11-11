"""Paired-band error accounting on identical valid RX pixels, no fitting."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';O=R/'audit/c4_remaining_error_20261031';F=O/'official_val';P=R/'runs/c4_full_refit_s20261031/predictions/val_full_bf16';state=np.load(F/'state_fields.npy',mmap_mode='r');Y=[np.load(D/f'val_pl{b}.npy',mmap_mode='r') for b in ['415','58']];M=[np.load(D/f'val_mask{b}.npy',mmap_mode='r') for b in ['415','58']];P=[np.load(P/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']];names=['all','height_sensitive','clear_near_boundary','clear_grazing','clear_robust','shallow_blocked','deep_blocked'];stats={n:np.zeros(12) for n in names};frequency_offset=20*np.log10(5.775e9/415e6)
for lo in range(0,2330,8):
 n=min(8,2330-lo);y=np.stack([a[lo:lo+n] for a in Y],1).astype(float);p=np.stack([a[lo:lo+n] for a in P],1).astype(float);valid=np.stack([np.unpackbits(a[lo:lo+n],axis=1,bitorder='big').reshape(n,256,256) for a in M],1).all(1);st=state[lo:lo+n].astype(float);e3,e2,c=st[:,0],st[:,1],st[:,2];sels=[np.ones_like(valid),(e3<0)&(e2>=0),(e3<0)&(e2<0)&(e2>=-.5),(e2<-.5)&(np.abs(c)<=.03),(e2<-.5)&(np.abs(c)>.03),(e3>=0)&(e3<=2),e3>2];err=p-y;dy=y[:,1]-y[:,0]-frequency_offset;dp=p[:,1]-p[:,0]-frequency_offset
 for name,sel in zip(names,sels):
  v=valid&sel;a=err[:,0][v];b=err[:,1][v];t=dy[v];q=dp[v];stats[name]+=np.array([len(a),np.sum(a*a),np.sum(b*b),np.sum(a*b),np.sum(a),np.sum(b),np.sum(t),np.sum(q),np.sum(t*t),np.sum(q*q),np.sum(t*q),np.sum((q-t)**2)])
results={}
for name,s in stats.items():
 n,sl,sh,cross,al,ah,ty,py,ty2,py2,tpy,diff=s;vt=ty2/n-(ty/n)**2;vp=py2/n-(py/n)**2
 results[name]={'joint_valid_pixels':int(n),'rmse_by_band':np.sqrt([sl/n,sh/n]).tolist(),'high_minus_low_sse':sh-sl,'share_of_total_high_minus_low_sse':float((sh-sl)/(stats['all'][2]-stats['all'][1])),'band_difference_error_rmse':float(np.sqrt(diff/n)),'common_error_energy_fraction':float((sl+sh+2*cross)/(2*(sl+sh))),'target_excess_band_difference_mean':ty/n,'prediction_excess_band_difference_mean':py/n,'target_excess_band_difference_std':float(np.sqrt(vt)),'prediction_excess_band_difference_std':float(np.sqrt(vp)),'band_difference_correlation':float((tpy/n-ty*py/n**2)/np.sqrt(vt*vp))}
(O/'joint_band_audit.json').write_text(json.dumps({'frequency_reference_db':frequency_offset,'results':results,'note':'Same valid pixels in both bands. Common/difference error energies are algebraic descriptors, not causal shares or noisefloors.'},indent=2)+'\n');print(json.dumps(results),flush=True)
