"""Independent lunar radio-map baseline. No imports from the parent project."""
import os
# User authorization 2026-11-01: physical cards 1, 3, 4 only.
# Legacy invocations keep GPU 3 unless the new launcher explicitly selects a UUID.
APPROVED_GPU_UUIDS = frozenset({
    'GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009',
    'GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0',
    'GPU-70b6fceb-c7c4-9467-3c81-097c9284c160',
})
GPU_UUID = os.environ.get('CC_GPU_UUID', 'GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0')
if GPU_UUID not in APPROVED_GPU_UUIDS or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
    raise RuntimeError('Use CC/run_gpu.sh {1|3|4} or legacy CC/run_gpu3.sh; one approved full UUID is required before importing torch.')
from pathlib import Path
import csv, ctypes, hashlib, json, time, uuid
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'data'
OFFSET=80.0
SCALE=40.0

def gpu_guard():
    assert torch.cuda.device_count()==1, 'Exactly the approved GPU must be visible'
    driver=ctypes.CDLL('libcuda.so.1')
    assert driver.cuInit(0)==0
    dev=ctypes.c_int();assert driver.cuDeviceGet(ctypes.byref(dev),0)==0
    raw=(ctypes.c_ubyte*16)();assert driver.cuDeviceGetUuid(ctypes.byref(raw),dev)==0
    actual='GPU-'+str(uuid.UUID(bytes=bytes(raw)))
    assert actual==GPU_UUID,(actual,GPU_UUID)
    props=torch.cuda.get_device_properties(0)
    return {'uuid':actual,'logical_device':0,'name':props.name,'memory_bytes':props.total_memory,'torch':torch.__version__,'cuda_runtime':torch.version.cuda}

def check_source_unchanged():
    audit=json.loads((ROOT/'audit/raw_audit.json').read_text())
    for name,ref in audit['files'].items():
        st=(DATA/name).stat()
        if st.st_size!=ref['size_bytes'] or st.st_mtime_ns!=ref['mtime_ns']:
            raise RuntimeError(f'Data source changed since independent audit: {name}')
    return audit

def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def d4(x,k):
    x=torch.rot90(x,k%4,(-2,-1))
    return torch.flip(x,[-1]) if k>=4 else x

def d4_coords(r,c,k):
    if k%4==1:r,c=255-c,r
    elif k%4==2:r,c=255-r,255-c
    elif k%4==3:r,c=c,255-r
    if k>=4:c=255-c
    return r,c

class ResidentData:
    """Keeps raw release arrays on the single allowed GPU; masks never enter model input."""
    def __init__(self,split='train'):
        self.split=split;t0=time.time()
        self.rows=list(csv.DictReader((DATA/f'{split}_index.csv').open()))
        n=len(self.rows)
        hm=np.load(DATA/f'{split}_hm.npy',mmap_mode='r',allow_pickle=False)
        self.h=torch.empty(hm.shape,dtype=torch.float32,device='cuda')
        for lo in range(0,len(hm),256):
            self.h[lo:lo+256].copy_(torch.from_numpy(np.array(hm[lo:lo+256],copy=True)))
        self.y=torch.empty((n,2,256,256),dtype=torch.float16,device='cuda')
        self.mask=torch.empty((n,2,256,256),dtype=torch.uint8,device='cuda')
        for channel,band in enumerate(['415','58']):
            arr=np.load(DATA/f'{split}_pl{band}.npy',mmap_mode='r',allow_pickle=False)
            packed=np.load(DATA/f'{split}_mask{band}.npy',mmap_mode='r',allow_pickle=False)
            assert arr.shape==(n,256,256) and arr.dtype==np.float16
            assert packed.shape==(n,8192) and packed.dtype==np.uint8
            for lo in range(0,n,512):
                hi=min(lo+512,n)
                self.y[lo:hi,channel].copy_(torch.from_numpy(np.array(arr[lo:hi],copy=True)))
                mask=np.unpackbits(packed[lo:hi],axis=1,bitorder='big').reshape(hi-lo,256,256)
                self.mask[lo:hi,channel].copy_(torch.from_numpy(mask))
        self.hmrow=torch.tensor([int(r['hm_row']) for r in self.rows],dtype=torch.long,device='cuda')
        self.txr=torch.tensor([int(r['tx_row']) for r in self.rows],dtype=torch.long,device='cuda')
        self.txc=torch.tensor([int(r['tx_col']) for r in self.rows],dtype=torch.long,device='cuda')
        self.yy,self.xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij')
        self.group=np.array([r['group']=='multi' for r in self.rows],dtype=np.bool_)
        self.profile=None
        torch.cuda.synchronize()
        print(json.dumps({'event':'data_loaded','split':split,'samples':n,'seconds':time.time()-t0,'allocated_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
    def indices(self,part):
        if self.split=='val' or part=='full':return np.arange(len(self.rows),dtype=np.int64)
        dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids'])
        assert part in ['fit','dev']
        return np.array([i for i,r in enumerate(self.rows) if (r['terrain_id'] in dev)==(part=='dev')],dtype=np.int64)
    def batch(self,idx,mode='raw',augment=0):
        idx=torch.as_tensor(idx,dtype=torch.long,device='cuda')
        h=self.h[self.hmrow[idx]]
        r,c=self.txr[idx],self.txc[idx]
        tx=torch.zeros_like(h)
        tx.flatten(1).scatter_(1,(r*256+c)[:,None],1.)
        y=self.y[idx].float();m=self.mask[idx].float()
        if augment:
            h=d4(h,augment);tx=d4(tx,augment);y=d4(y,augment);m=d4(m,augment)
            r,c=d4_coords(r,c,augment)
        features=[h/100.,tx]
        if mode in ['geometry','profile']:
            dx=self.xx[None]-c[:,None,None];dy=self.yy[None]-r[:,None,None]
            txh=h[torch.arange(len(idx),device='cuda'),r,c]
            dz=h+1.-(txh[:,None,None]+3.)
            distance=torch.sqrt(dx.square()+dy.square()+dz.square())
            features += [dx/256.,dy/256.,torch.log1p(distance)/6.,dz/100.]
        if mode=='profile':
            assert self.profile is not None,'Input-only profile features must be prepared first'
            p=self.profile[idx].float()
            if augment:p=d4(p,augment)
            features.extend([p[:,0],p[:,1]])
        x=torch.stack(features,dim=1).contiguous(memory_format=torch.channels_last)
        return x,y,m

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(inplace=True),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(inplace=True))
    def forward(self,x):return self.net(x)

class UNet(nn.Module):
    def __init__(self,in_channels=2,width=32,out_channels=2):
        super().__init__()
        cs=[width*2**i for i in range(5)]
        self.enc=nn.ModuleList([Block(in_channels,cs[0])]+[Block(cs[i-1],cs[i]) for i in range(1,5)])
        self.dec=nn.ModuleList([Block(cs[i]+cs[i-1],cs[i-1]) for i in range(4,0,-1)])
        self.head=nn.Conv2d(cs[0],out_channels,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.enc):
            if i:x=F.max_pool2d(x,2)
            x=block(x);skips.append(x)
        for block,skip in zip(self.dec,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=block(torch.cat([x,skip],dim=1))
        # FP32 final projection avoids quantizing the predicted dB output to BF16.
        with torch.autocast('cuda',enabled=False):return self.head(x.float())

def masked_loss(pred_normalized,target_db,mask):
    error=pred_normalized.float()-(target_db-OFFSET)/SCALE
    num=(error.square()*mask).sum(dim=(0,2,3))
    den=mask.sum(dim=(0,2,3)).clamp_min(1)
    return (num/den).mean()

def model_for(mode='raw',width=32):
    model=UNet({'raw':2,'geometry':6,'profile':8}[mode],width)
    return model.to('cuda',memory_format=torch.channels_last)

@torch.inference_mode()
def evaluate(model,data,indices,mode='raw',batch_size=64,amp=True,save_per_map=None,prediction_path=None):
    was_training=model.training;model.eval();t0=time.time()
    totals=torch.zeros(2,device='cuda',dtype=torch.float64);counts=torch.zeros_like(totals);alltot=torch.zeros_like(totals)
    sse_rows=[];count_rows=[]
    predfile=np.lib.format.open_memmap(prediction_path,mode='w+',dtype=np.float32,shape=(len(indices),2,256,256)) if prediction_path else None
    for lo in range(0,len(indices),batch_size):
        ii=indices[lo:lo+batch_size];x,y,m=data.batch(ii,mode)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):pred=model(x)*SCALE+OFFSET
        err=(pred.float()-y).square()
        ss=(err*m).sum(dim=(2,3),dtype=torch.float64);cc=m.sum(dim=(2,3),dtype=torch.float64)
        totals+=ss.sum(0);counts+=cc.sum(0);alltot+=err.sum(dim=(0,2,3),dtype=torch.float64)
        sse_rows.append(ss.cpu().numpy());count_rows.append(cc.cpu().numpy())
        if predfile is not None:predfile[lo:lo+len(ii)]=pred.float().cpu().numpy()
    ss=np.concatenate(sse_rows);cc=np.concatenate(count_rows)
    rmse=torch.sqrt(totals/counts).cpu().tolist();allrmse=torch.sqrt(alltot/(len(indices)*65536)).cpu().tolist()
    result={'rmse415':rmse[0],'rmse58':rmse[1],'mse_mean':sum(x*x for x in rmse)/2,'all_pixel_rmse415':allrmse[0],'all_pixel_rmse58':allrmse[1],'maps_per_band':len(indices),'valid_pixels':counts.cpu().tolist(),'seconds':time.time()-t0,'groups':{}}
    for flag,label in [(False,'single'),(True,'multi')]:
        select=data.group[indices]==flag
        if select.any():result['groups'][label]={'maps':int(select.sum()),'rmse':np.sqrt(ss[select].sum(0)/cc[select].sum(0)).tolist()}
    if save_per_map:
        with Path(save_per_map).open('w') as f:
            writer=csv.writer(f);writer.writerow(['index_row','sample_id','terrain_id','group','sse415','count415','sse58','count58'])
            for j,i in enumerate(indices):
                r=data.rows[int(i)];writer.writerow([i,r['sample_id'],r['terrain_id'],r['group'],ss[j,0],int(cc[j,0]),ss[j,1],int(cc[j,1])])
    if predfile is not None:predfile.flush()
    if was_training:model.train()
    return result
