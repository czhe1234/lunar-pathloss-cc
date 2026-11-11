"""Candidate TX altitude convention found on training terrains; full-map feature extraction."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard,DATA,atomic_json
from src.c67_physics import Transport
import numpy as np
class RegisteredTransport(Transport):
    def source_height(self,h,row,col):
        return float(np.floor(h[min(row+1,255),col])+3)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dir',default='features/c69_transport')
    p.add_argument('--panel',default='');p.add_argument('--split',default='train')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=1)
    a=p.parse_args();gpu=gpu_guard();engine=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=128)
    if a.panel: ids=json.loads((ROOT/a.panel).read_text())
    else:
        panel=json.loads((ROOT/'audit/c66_transport/panels.json').read_text());ids=panel['dev_maps']+panel['fit_maps']
    ids=ids[a.shard::a.of]
    out=ROOT/a.dir/f'part{a.shard}of{a.of}';out.mkdir(parents=True,exist_ok=False);atomic_json(out/'indices.json',ids)
    arr=np.lib.format.open_memmap(out/'features.npy','w+',dtype=np.float16,shape=(len(ids),16,256,256))
    rows=list(csv.DictReader((DATA/f'{a.split}_index.csv').open()));heights=np.load(DATA/f'{a.split}_hm.npy',mmap_mode='r')
    t=time.monotonic()
    for j,i in enumerate(ids):
        r=rows[i];h=heights[int(r['hm_row'])]
        x=engine.features(h,int(r['tx_row']),int(r['tx_col'])).half().cpu().numpy()
        assert np.isfinite(x).all()
        arr[j]=x
        if j%64==0 or j+1==len(ids):
            v=dict(maps=j+1,total=len(ids),seconds=round(time.monotonic()-t,2));atomic_json(out/'progress.json',v);print(v,flush=True)
    arr.flush();atomic_json(out/'complete.json',dict(vars(a),gpu=gpu,features=16))
if __name__=='__main__':main()
