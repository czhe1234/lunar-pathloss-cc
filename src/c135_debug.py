import sys,csv,ctypes,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c135_paths import *
gpu_guard();e=Paths();rows=list(csv.DictReader((DATA/'train_index.csv').open()));row=rows[4];h=np.load(DATA/'train_hm.npy',mmap_mode='r')[int(row['hm_row'])];tr=int(row['tx_row']);tc=int(row['tx_col'])
s=e.forward(h,tr,tc);d,sp,*_=e.rt.raw(h,tr,tc)
f=e.lib.c135_refl_audit;P=ctypes.c_void_p;I=ctypes.c_int;F=ctypes.c_float;f.argtypes=[P,P,P,P,P,I,F,F,F,F,P,P]
out=torch.empty(len(s['g'])*len(e.rt.spec),4,device='cuda')
f(s['g'].data_ptr(),s['n'].data_ptr(),e.rt.material.data_ptr(),e.refl.data_ptr(),e.rt.spec.data_ptr(),len(e.rt.spec),tc*255/256,(tr+1)*255/256,e.rt.source_height(h,tr,tc),s['top'],out.data_ptr(),torch.cuda.current_stream().cuda_stream)
a=out[:,:2];b=out[:,2:];err=(a-b).abs();dif=s['spec'][:,0]-sp[:,:2];print(dict(energy_relative_L1=(err.sum(0)/a.sum(0)).tolist(),energy_max_fraction=(err/a.clamp_min(1e-25)).amax(0).tolist(),rx_net_fraction=(dif.sum(0)/sp[:,:2].sum(0)).tolist(),rx_mismatch_faces=(dif.abs()>1e-6*sp[:,:2]).sum(0).tolist(),total_faces=len(sp)))
