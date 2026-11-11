"""Published-input45-channel data; labels and masks are returned separately."""
from pathlib import Path
import ctypes,csv,json
import numpy as np
from scipy.ndimage import uniform_filter
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
_lib=ctypes.CDLL(str(R/'src/c21_phase.so'))
_lib.phase_map.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
_lib.phase_map.restype=None
def phase(h,tr,tc):
 h=np.ascontiguousarray(h,np.float32);out=np.empty((14,256,256),np.float32)
 _lib.phase_map(h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),tr,tc,out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
 return out
class Data:
 def __init__(self,targets=True):
  self.rows=list(csv.DictReader((D/'train_index.csv').open()));self.H=np.load(D/'train_hm.npy',mmap_mode='r')
  self.G=np.load(R/'features/c3/train.npy',mmap_mode='r');self.E=np.load(R/'features/c4_extra/train.npy',mmap_mode='r')
  self.phasepath=R/'features/c21_phase/train.npy';self.Ph=None
  if self.phasepath.exists() and (self.phasepath.parent/'complete.json').exists():self.Ph=np.load(self.phasepath,mmap_mode='r')
  self.yy,self.xx=np.mgrid[:256,:256].astype(np.float32)
  dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
  self.ids={s:np.array([i for i,r in enumerate(self.rows) if (r['terrain_id'] in dev)==(s=='dev')]) for s in ['fit','dev']}
  if targets:
   self.Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
   self.M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
 def inputs(self,i):
  row=self.rows[int(i)];h=np.asarray(self.H[int(row['hm_row'])],np.float32);tr=int(row['tx_row']);tc=int(row['tx_col']);ht=h[tr,tc]
  dx=self.xx-tc;dy=self.yy-tr;dz=h+1-ht-3;dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25);sy,sx=np.gradient(h)
  basic=np.stack((h/100,(h-ht)/50,(h-uniform_filter(h,size=5,mode='nearest'))/5,sx/5,sy/5,dx/128,dy/128,dz/50,np.log1p(dist)/6,((dx==0)&(dy==0)).astype(np.float32),np.exp(-(dx*dx+dy*dy)/8)))
  ph=np.asarray(self.Ph[i],np.float32) if self.Ph is not None else phase(h,tr,tc)
  x=np.concatenate((basic,np.asarray(self.G[i],np.float32),np.asarray(self.E[i],np.float32),ph),0)
  fs=(20*np.log10(4*np.pi*dist[None]*np.array([415e6,5.775e9])[:,None,None]/299792458.)).astype(np.float32)
  return x,np.array([tr,tc,ht+3],np.float32),fs
 def sample(self,i):
  x,tx,fs=self.inputs(i);y=np.stack([a[i] for a in self.Y]).astype(np.float32)
  m=np.stack([np.unpackbits(a[i],bitorder='big').reshape(256,256) for a in self.M]).astype(np.float32)
  return x,tx,fs,y,m
