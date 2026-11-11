"""Nominal path-sampling resolution audit on frozen DEV64 monitor pixels; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,time
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[1];A=R/'audit/c43_architecture_20261102'
p=json.loads((A/'protocol.json').read_text());ids=p['dev64_ids'];pix=np.array(p['monitor_pixels']);rr,cc=np.divmod(pix,256)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');G=np.load(R/'features/c3/train.npy',mmap_mode='r')
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
stats={str(n):{'valid_pixels':np.zeros(2),'missed_blocked_pixels':np.zeros(2),'underestimated_by_025m_pixels':np.zeros(2),'blocked_pixels':np.zeros(2)} for n in [8,32,128]}
errs={};start=time.time()
for k,i in enumerate(ids):
    r=rows[i];h=np.asarray(H[int(r['hm_row'])]);tr=int(r['tx_row']);tc=int(r['tx_col'])
    ht=float(h[tr,tc])+3;zr=h[rr,cc].astype(float)+1
    dense=2*np.sinh(np.asarray(G[i,:8],np.float32)[:,rr,cc]*3).max(0)
    ms=[np.unpackbits(M[b][i],bitorder='big')[pix].astype(bool) for b in [0,1]]
    for n in [8,32,128]:
        t=np.linspace(.05,.95,n) if n==8 else (np.arange(n)+.5)/n
        qx=tc+(cc-tc)[None]*t[:,None];qy=tr+(rr-tr)[None]*t[:,None]
        sampled=map_coordinates(h,np.stack((qy,qx)),order=1,mode='nearest',prefilter=False)
        clearance=(sampled-(ht+(zr-ht)[None]*t[:,None])).max(0)
        blocked=dense>.25;miss=blocked&(clearance<=0);under=dense-clearance>.25
        s=stats[str(n)]
        for b,m in enumerate(ms):
            s['valid_pixels'][b]+=m.sum();s['blocked_pixels'][b]+=(blocked&m).sum()
            s['missed_blocked_pixels'][b]+=(miss&m).sum();s['underestimated_by_025m_pixels'][b]+=(under&m).sum()
    if (k+1)%64==0:print(json.dumps({'maps':k+1,'seconds':time.time()-start}),flush=True)
for s in stats.values():
    s['missed_blocked_fraction']=np.divide(s['missed_blocked_pixels'],s['blocked_pixels']).tolist()
    s['underestimated_fraction_of_valid']=np.divide(s['underestimated_by_025m_pixels'],s['valid_pixels']).tolist()
    for key in ['valid_pixels','blocked_pixels','missed_blocked_pixels','underestimated_by_025m_pixels']:s[key]=s[key].astype(int).tolist()
out={'scope':'264 frozen internal DEV maps,8192 frozen uniform pixels/map; nominal TX3m/RX1m raster geometry',
     'reference':'256 midpoint samples from C3 input-only geometric cache; fp16 cache quantization',
     'statistics':stats,'seconds':time.time()-start,
     'limits':'Nominal unshifted raw-height references only. CNN feature receptive fields and learned offsets may recover information; this does NOT prove the trained attention misses the same obstacles. It motivates testing denser ordered profiles, not a claimed causal attribution.'}
(A/'sampling_resolution.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
