"""Exploratory receiver-sheet controls motivated by common-band direct-power deficits."""
import argparse,csv,ctypes,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,DATA,atomic_json
from src.c78_geometry import planes,A
from src.c70_geometry import qpoints
import torch
import numpy as np
C=['first_hit_fraction','front_face_fraction','prior_receiver_hits','grazing_fraction','mean_log1p_expected','Poisson_zero_4B','Poisson_zero_40M','Poisson_zero_4M','min_log1p_expected_weight_ge_05','sign_mixing']
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='train');p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=3);a=p.parse_args()
    gpu=gpu_guard();panel=np.load(A/f'{a.split}_panel.npz');pos=np.arange(len(panel['ids']))[a.shard::a.of];ids,pix=panel['ids'],panel['pixels']
    rows=list(csv.DictReader((DATA/f'{a.split}_index.csv').open()));hh=np.load(DATA/f'{a.split}_hm.npy',mmap_mode='r');geo=np.load(R/'src/c70_centroid_lookup0.npz')
    lookup=torch.tensor(geo['ids'],device='cuda',dtype=torch.int32);ws=torch.tensor(geo['weights'],device='cuda',dtype=torch.float32);quad=qpoints(4)
    lib=ctypes.CDLL(str(R/'src/libc78_sheet.so'));fn=lib.sheet_geometry78;P=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_float
    fn.argtypes=[P,P,I,P,I,S,S,S,S,P,P];fn.restype=I
    out=A/f'{a.split}_sheet_{a.shard}of{a.of}';out.mkdir(exist_ok=False);np.save(out/'positions.npy',pos);arr=np.lib.format.open_memmap(out/'sheet.npy','w+',dtype=np.float32,shape=(len(pos),pix.shape[1],len(C)))
    for k,j in enumerate(pos):
        row=rows[ids[j]];r,c=int(row['tx_row']),int(row['tx_col']);h=hh[int(row['hm_row'])];ht=torch.from_numpy(np.array(h,copy=True)).cuda();g=planes(ht,False)
        px=torch.from_numpy(pix[j].astype('int64')).cuda();target=lookup[px].flatten().contiguous();w=ws[px]
        vals=torch.empty((len(target),8),device='cuda')
        code=fn(g.data_ptr(),target.data_ptr(),len(target),quad.data_ptr(),len(quad),c*255/256,(r+1)*255/256,float(np.floor(h[min(255,r+1),c])+3),float(h.max()),vals.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(code)
        v=vals.reshape(-1,3,8);avg=(v*w[:,:,None]).sum(1);minimum=torch.where(w>=.05,v[:,:,4],100.).amin(1)
        z=torch.cat((avg,minimum[:,None],(4*avg[:,1:2]*(1-avg[:,1:2])).clamp(0,1)),1)
        arr[k]=z.cpu().numpy()
    arr.flush();atomic_json(out/'complete.json',dict(columns=C,gpu=gpu,maps=len(pos)))
if __name__=='__main__':main()
