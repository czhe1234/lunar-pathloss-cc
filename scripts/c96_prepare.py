"""H/TX-only propagation-strip information test. No C95 inputs or numerical-offset models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,json,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[1];A=R/'audit/c96_information';D=R.parent/'data'
def strip(h,tr,tc,pixels):
    rr,cc=np.divmod(pixels,256);dx=cc.astype(np.float32)-tc;dy=rr.astype(np.float32)-tr
    horizontal=np.hypot(dx,dy).clip(1.);nx=-dy/horizontal;ny=dx/horizontal
    # Official nominal antenna/probe heights. No frame/epsilon fitting.
    txz=float(h[tr,tc])+3.;rxz=h[rr,cc]+1.;dz=rxz-txz
    t=np.broadcast_to((np.arange(64,dtype=np.float32)+.5)/64,(len(pixels),64))
    near=np.arange(1,25,dtype=np.float32)[None,:]/horizontal[:,None]
    t=np.concatenate([t,near.clip(0,1),(1-near).clip(0,1)],1)
    cols=tc+dx[:,None]*t;rows=tr+dy[:,None]*t
    side=np.array([-12.,-4.,-1.,1.,4.,12.],np.float32)
    sx=cols[:,None,:]+side[None,:,None]*nx[:,None,None]
    sy=rows[:,None,:]+side[None,:,None]*ny[:,None,None]
    domain=(sx>=0)&(sx<=255)&(sy>=0)&(sy<=255)
    heights=map_coordinates(h,np.stack([sy,sx]),order=1,mode='nearest',prefilter=False)
    e=heights-(txz+dz[:,None,None]*t[:,None,:])
    raw=np.arcsinh(e/2)/3
    stats=np.stack([e.max(2),e.mean(2),e.std(2),(e>0).mean(2),domain.mean(2)],2)
    stats=np.concatenate([np.arcsinh(stats[:,:,:3]/2)/3,stats[:,:,3:]],2)
    return np.concatenate([raw.reshape(len(pixels),-1),stats.reshape(len(pixels),-1)],1).astype('float32')
def main():
    rows=list(csv.DictReader(open(D/'train_index.csv')));H=np.load(D/'train_hm.npy',mmap_mode='r')
    fitmaps=np.load(R/'features/c38_dense/fit_map_indices.npy')
    allfit={int(v):j for j,v in enumerate(fitmaps)}
    chosen=json.loads((R/'audit/c72_continuation/wide_fit.json').read_text())
    devmaps=np.load(R/'features/c6_points/dev_map_indices.npy')
    bi=json.loads((R/'audit/c88_mesh/base_index.json').read_text());db={int(i):j for j,i in enumerate(bi['ids'])}
    groups={int(i):g for i,g in zip(bi['ids'],bi['groups'])}
    fit_terrains={rows[i]['terrain_id'] for i in chosen};dev_terrains={rows[int(i)]['terrain_id'] for i in devmaps}
    assert len(fit_terrains)==5508 and len(dev_terrains)==612 and not fit_terrains&dev_terrains
    assert set(map(int,devmaps))==set(db)
    counts=Counter(rows[int(i)]['terrain_id'] for i in fitmaps);selected=Counter(rows[i]['terrain_id'] for i in chosen)
    protocol=dict(question='Does dense lateral terrain along the TX-RX corridor improve held-out masked RMSE beyond old294 features?',
      fit_maps=len(chosen),fit_terrains=5508,points_per_fit_map=32,fit_points=len(chosen)*32,
      development='same128terrain monitor /484terrain confirmation asC82; previously used in research; no officialVAL targets.',
      features_A='old294H/TX descriptors plus frozenC38 two-band predicted excess; excludes target/mask.',
      features_B='A plus6lateral profiles at offsets[-12,-4,-1,1,4,12]m,64normalized distance samples and24one-meter samples near each endpoint;672raw+30summary.',
      target='Residual to frozenC38, trained onFIT only. Compare the resulting predictor toC82 on identical held-out pixels.',
      criterion='Expand only if frozen candidate beatsC82 by>=0.30dB in each band on confirmation with positive terrain-paired gain. Also reportA/Bchange.',
      raw_sampling='32uniform draws from each original1024input-onlyFIT pixel cache; same rows/order forA/B; multiTX inclusion weights preserve sample objective.',
      sample_selection_reads_targets=False,feature_extractor_reads_targets=False,official_val_used=False,
      GPU='Only physical1/3/4 byUUID; all three currentcuInit999. Pilot usesCPU. No driver reset.',
      scope='Small information screen, not a final model or official validation score.')
    p=A/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2)+'\n')
    for part,maps,pp in [('fit',chosen,32),('dev',devmaps.tolist(),256)]:
        out=A/part;out.mkdir(exist_ok=True)
        if (out/'done.json').exists():continue
        old=R/'features'/('c38_dense' if part=='fit' else 'c10_points')
        xx=np.load(old/(part+'_x.npy'),mmap_mode='r')
        labdir=R/'features'/('c38_dense' if part=='fit' else 'c6_points')
        yy=np.load(labdir/(part+'_y.npy'),mmap_mode='r');mm=np.load(labdir/(part+'_mask.npy'),mmap_mode='r')
        fs=np.load(labdir/(part+'_fspl.npy'),mmap_mode='r');px=np.load(labdir/(part+'_pixel.npy'),mmap_mode='r')
        teacher=np.load(R/f'features/c47_teacher/{part}_base.npy',mmap_mode='r')
        ref=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r') if part=='dev' else None
        devpos={int(i):j for j,i in enumerate(devmaps)}
        n=len(maps)*pp;shapes={'x':(n,296),'strip':(n,702),'y':(n,2),'mask':(n,2),'base':(n,2),'row_weight':(n,),'mapid':(n,),'pixel':(n,)}
        if part=='dev':shapes['c82']=(n,2)
        arrays={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype=(np.uint8 if k=='mask' else np.int32 if k in ['mapid','pixel'] else np.float32),shape=v) for k,v in shapes.items()}
        def get(j):
            i=int(maps[j]);row=rows[i]
            if part=='fit':
                position=allfit[i];sel=np.random.default_rng([960431,i]).choice(1024,32,replace=False);ix=position*1024+sel
            else:ix=np.arange(devpos[i]*256,(devpos[i]+1)*256)
            pixels=np.array(px[ix],np.int32);x=np.array(xx[ix],np.float32);bt=np.array(teacher[ix],np.float32)
            base=np.array(fs[ix],np.float32)+20*bt
            new=strip(H[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']),pixels)
            assert new.shape==(pp,702) and np.isfinite(new).all()
            weight=counts[row['terrain_id']]/selected[row['terrain_id']] if part=='fit' else 1.
            r=dict(x=np.c_[x,bt],strip=new,y=np.array(yy[ix]),mask=np.array(mm[ix]),base=base,
               row_weight=np.full(pp,weight),mapid=np.full(pp,i),pixel=pixels)
            if part=='dev':r['c82']=np.array(ref[db[i]].reshape(2,-1)[:,pixels].T)
            return j,r
        started=time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for j,r in pool.map(get,range(len(maps))):
                sl=slice(j*pp,(j+1)*pp)
                for k,v in r.items():arrays[k][sl]=v
                if (j+1)%400==0:
                    progress=dict(part=part,maps=j+1,total=len(maps),seconds=round(time.monotonic()-started,1))
                    (out/'progress.json').write_text(json.dumps(progress)+'\n');print(progress,flush=True)
        for v in arrays.values():v.flush()
        meta=dict(maps=maps,points=n,seconds=time.monotonic()-started,features_A=296,features_B=998)
        if part=='dev':meta['groups']=[groups[int(i)] for i in maps]
        (out/'done.json').write_text(json.dumps(meta)+'\n');print('done',part,n,round(meta['seconds'],1),flush=True)
if __name__=='__main__':main()
