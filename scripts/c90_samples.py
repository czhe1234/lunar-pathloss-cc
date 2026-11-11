"""Build the angle-response training experiment from released TRAIN and frozen input-only fields."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c90_facet';D=R.parent/'data'
def mm(p):return np.load(p,mmap_mode='r')
def source_map(dirs,filename):
    store={}
    for s in dirs:
        if not (s/'complete.json').is_file():continue
        arr=mm(s/filename)
        store.update((i,(arr,j)) for j,i in enumerate(json.load(open(s/'indices.json'))))
    return store
def batch_input(h,r,c,px,p,co):
    y,x=np.divmod(px,256);rr=(y[:,None]+np.repeat(np.arange(-3,4),7)).clip(0,255)
    cc=(x[:,None]+np.tile(np.arange(-3,4),7)).clip(0,255)
    patch=(h[rr,cc]-h[y,x,None]).clip(-10,10)
    dx=x-c;dy=y-r;dz=h[y,x]+1-(np.floor(h[min(r+1,255),c])+3);length=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
    fs=20*np.log10(length[:,None]*np.array([415e6,5.775e9])[None]*4*np.pi/299792458)
    u=dx/length;v=dy/length
    # Preserve elevation while permuting incidence azimuth at each pixel. The control uses the same terrain and features.
    hashed=(px.astype(np.int64)*1664525+1013904223)&0xffffffff
    angle=(hashed%8191)/8191*(2*np.pi);rotx=u*np.cos(angle)-v*np.sin(angle);roty=u*np.sin(angle)+v*np.cos(angle)
    cen=p[:,y,x].T
    edges=[(p[:,(y+j).clip(0,255),(x+i).clip(0,255)].T-cen)/6 for j,i in ((1,0),(-1,0),(0,1),(0,-1))]
    feat=np.column_stack([patch,co,cen,fs,u,v,dz/length,np.log1p(length),rotx,roty,*edges])
    assert feat.shape==(len(px),112) and np.isfinite(feat).all()
    return feat.astype('float16')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--of',type=int,default=3);a=ap.parse_args()
    rows=list(csv.DictReader(open(D/'train_index.csv')));hh=mm(D/'train_hm.npy')
    yy=[mm(D/f'train_pl{s}.npy') for s in ('415','58')];ww=[mm(D/f'train_mask{s}.npy') for s in ('415','58')]
    compact=source_map([*(R/'features/c73_compact').glob('part*'),*(R/'features/c73_dev_extra_compact').glob('part*')],'features.npy')
    fitbase=source_map([*(R/'features/c66_frozen_maps').glob('fit*of2'),*(R/'features/c75_fit_extension').glob('fit*')],'prediction.npy')
    data=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1';bi=json.load(open(R/'audit/c88_mesh/base_index.json'))
    assert bi['ids']==json.load(open(data/'indices.json'))['index_rows']
    dev={i:j for j,i in enumerate(bi['ids'])};devterr={rows[i]['terrain_id'] for i in dev}
    devbase=[mm(data/f'pl{b}.npy') for b in ('415','58')];c82=mm(R/'audit/c88_mesh/base.npy')
    exmap=mm(R/'features/c38_dense/fit_map_indices.npy');ex=mm(R/'features/c38_dense/fit_pixel.npy').reshape(len(exmap),-1);exidx={i:j for j,i in enumerate(exmap)}
    panel=np.load(R/'audit/c78_clear/train_panel.npz');pixels=panel['pixels']
    fit=sorted(set(compact)&set(fitbase));assert all(rows[i]['terrain_id'] not in devterr for i in fit)
    for group in ('fit','cal','select','confirm'):
        if group=='fit':ids=fit;m=1536
        else:
            ids=[i for i,j in dev.items() if bi['groups'][j]==('fit' if group=='cal' else group)];m=2048
        ids=ids[a.shard::a.of];out=A/(group+str(a.shard));out.mkdir(exist_ok=True);ts=time.monotonic()
        x=np.lib.format.open_memmap(out/'x.npy',mode='w+',shape=(len(ids),m,112),dtype='float16')
        targets=np.lib.format.open_memmap(out/'targets.npy',mode='w+',shape=(len(ids),m,6),dtype='float32')
        for k,i in enumerate(ids):
            row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);h=hh[int(row['hm_row'])]
            if group=='fit':
                rng=np.random.default_rng([90681,i]);allowed=np.ones(65536,bool);allowed[ex[exidx[i]].astype(np.int64)]=False
                px=rng.choice(np.flatnonzero(allowed),m,replace=False);aa,j=fitbase[i];p=aa[j].astype(np.float32)
            else:
                j=dev[i];px=pixels[j].astype(np.int64);p=np.stack([a[j] for a in devbase])
            aa,ci=compact[i];co=aa[ci].reshape(45,-1)[:,px].T
            x[k]=batch_input(h,tr,tc,px,p,co)
            y=np.stack([v[i].ravel()[px] for v in yy],1);w=np.stack([np.unpackbits(v[i])[px] for v in ww],1)
            ref=p.reshape(2,-1)[:,px].T if group=='fit' else c82[j].reshape(2,-1)[:,px].T
            targets[k]=np.column_stack([y,w,ref])
            if (k+1)%350==0:print(group,k+1,len(ids),round(time.monotonic()-ts),flush=True)
        x.flush();targets.flush();(out/'done.json').write_text(json.dumps(dict(ids=ids,terrains=[rows[i]['terrain_id'] for i in ids],pixels=m,seconds=time.monotonic()-ts))+'\n')
if __name__=='__main__':main()
