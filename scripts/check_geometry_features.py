"""Validate geometry and inference parity on synthetic HEIGHT inputs, never new PL labels."""
import sys,fcntl,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import *
from input_features import make_inputs,terrain_profiles,PROFILE_SAMPLES
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=gpu_guard();torch.set_num_threads(8);checks={}
yy,xx=np.indices((256,256));heights=np.stack([np.zeros_like(xx),.08*xx+.04*yy,15*np.exp(-((xx-130)**2+(yy-100)**2)/200),3*np.sin(xx/11)+2*np.cos(yy/17)]).astype(np.float32)
r=np.array([0,128,255,63]);c=np.array([255,127,0,190]);d=ResidentData.__new__(ResidentData)
d.h=torch.from_numpy(heights).cuda();d.hmrow=torch.arange(4,device='cuda');d.txr=torch.from_numpy(r).cuda();d.txc=torch.from_numpy(c).cuda();d.yy,d.xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij');d.y=torch.zeros((4,2,256,256),device='cuda',dtype=torch.float16);d.mask=torch.ones_like(d.y,dtype=torch.uint8);d.profile=terrain_profiles(d.h,d.txr,d.txc).half()
with torch.inference_mode():
 for mode in ('raw','geometry','profile'):
  for k in range(8):
   x,_,_=d.batch(np.arange(4),mode,k);hr=d4(d.h,k);rr,cc=d4_coords(d.txr,d.txc,k);pr=d4(d.profile.float(),k) if mode=='profile' else None
   inference=make_inputs(hr,rr,cc,mode,profile=pr)
   error=float((x-inference).abs().max());assert error==0,(mode,k,error);checks[f'inference_parity_{mode}_d4_{k}']=error
 for k in range(8):
  hr=d4(d.h,k);rr,cc=d4_coords(d.txr,d.txc,k)
  actual=terrain_profiles(hr,rr,cc);expected=d4(d.profile.float(),k)
  error=float((actual-expected).abs().max());assert error<=2e-4,(k,error);checks[f'profile_geometric_d4_{k}_max_abs']=error
 flat_expected=(-3+2*PROFILE_SAMPLES/(PROFILE_SAMPLES+1))/100
 flat_actual=float(d.profile[0,0,100,100]);assert abs(flat_actual-flat_expected)<1e-5
 assert torch.count_nonzero(d.profile[0,1])==0
 checks['flat_plane_expected_max_excess_over_100']=flat_expected;checks['flat_plane_actual']=flat_actual
 # Separate NumPy bilinear interpolation for fixed points.
 rng=np.random.default_rng(731);points=rng.integers(0,256,size=(40,2));maxerr=0.
 for i in range(4):
  for yr,xr in points:
   t=np.arange(1,PROFILE_SAMPLES+1)/(PROFILE_SAMPLES+1);x=c[i]+(xr-c[i])*t;y=r[i]+(yr-r[i])*t
   x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,255);y1=np.minimum(y0+1,255);wx=x-x0;wy=y-y0;a=heights[i]
   profile=(1-wx)*(1-wy)*a[y0,x0]+wx*(1-wy)*a[y0,x1]+(1-wx)*wy*a[y1,x0]+wx*wy*a[y1,x1]
   line=(a[r[i],c[i]]+3)*(1-t)+(a[yr,xr]+1)*t;z=profile-line
   expected=np.array([z.max(),np.maximum(z,0).mean()])/100
   actual=d.profile[i,:,yr,xr].float().cpu().numpy();maxerr=max(maxerr,float(np.abs(actual-expected).max()))
 assert maxerr<1e-4,maxerr;checks['independent_numpy_bilinear_max_abs']=maxerr
atomic_json(ROOT/'audit/geometry_sanity.json',{'all_passed':True,'checks':checks,'device':info,'only_artificial_height_inputs_no_propagation_or_new_target_data':True});print(json.dumps({'geometry_checks':len(checks),'all_passed':True}),flush=True)
