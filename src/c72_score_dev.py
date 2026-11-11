"""Full internal TRAIN-only evaluation. Report the 484 held-out terrains separately."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c67_dense import Panel,ResidualMap
from src.c70_field import FieldMap
import torch
import numpy as np
def load_net(run):
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    if ck.get('network','local')=='field':net=FieldMap(ck['channels'],ck['width'])
    elif ck.get('network') in ('native','observed'):
        from src.c72_native import NativeMap
        net=NativeMap(ck['channels'],ck['width'],ck['network']=='native')
    else:net=ResidualMap(ck['channels'],ck['width'],ck['mode']=='spatial')
    net.load_state_dict(ck['state']);return net.to('cuda',memory_format=torch.channels_last).eval()
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--features',required=True)
    a=p.parse_args();gpu=gpu_guard();torch.set_num_threads(5);torch.set_float32_matmul_precision('high')
    run=ROOT/'runs'/a.run;assert (run/'result.json').is_file()
    net=load_net(run);allidx=json.load(open(ROOT/'runs/c54_combo_dev/bf16_compiled/all_shard0of1/indices.json'))['index_rows']
    rows=list(csv.DictReader((DATA/'train_index.csv').open()))
    terrains=[rows[i]['terrain_id'] for i in allidx]
    panel=json.load(open(ROOT/'audit/c66_transport/panels.json'));outside=~np.isin(terrains,panel['dev_terrains'])
    dest=run/'full_internal';dest.mkdir(exist_ok=False)
    out=np.lib.format.open_memmap(dest/'correction.npy','w+',dtype='float16',shape=(len(allidx),2,256,256))
    stats=[];st=time.monotonic()
    for start in range(0,len(allidx),64):
        ix=allidx[start:start+64];data=Panel(ix,'dev',a.features)
        chunk=[]
        for j in range(0,len(ix),8):
            sl=slice(j,j+8)
            with torch.autocast('cuda',torch.bfloat16):corr=net(data.x[sl].float()).float()
            corr=corr.half().float() # storage and deployment precision for the correction
            e=data.y[sl];w=data.w[sl];r=e-corr
            chunk.append(torch.cat(((e.square()*w).sum((2,3),dtype=torch.float64),(r.square()*w).sum((2,3),dtype=torch.float64),w.sum((2,3))),1).cpu().numpy())
            out[start+j:start+j+len(corr)]=corr.cpu().numpy()
        stats.extend(np.concatenate(chunk));del data
        if start%320==0 or start+len(ix)==len(allidx):
            print('maps',start+len(ix),'of',len(allidx),'seconds',round(time.monotonic()-st,1),flush=True)
    out.flush();d=np.asarray(stats);atomic_json(dest/'indices.json',allidx)
    np.save(dest/'sse_counts.npy',d)
    report={'run':a.run,'gpu':gpu,'official_val_used':False,'seconds':time.monotonic()-st}
    def scores(v):
        return np.sqrt(v[...,:2]/v[...,4:6]),np.sqrt(v[...,2:4]/v[...,4:6])
    for name,sel in [('all612',np.ones(len(d),bool)),('outside484',outside),('panel128_allTX',~outside)]:
        t=np.array(terrains)[sel];unique,inv=np.unique(t,return_inverse=True);groups=np.zeros((len(unique),6));np.add.at(groups,inv,d[sel])
        agg=groups.sum(0);b,m=scores(agg);rng=np.random.default_rng(72031)
        draws=groups[rng.integers(len(groups),size=(2000,len(groups)))].sum(1);bb,mm=scores(draws)
        report[name]=dict(terrains=len(unique),maps=int(sel.sum()),valid=agg[4:].astype(int).tolist(),baseline=b.tolist(),new=m.tolist(),gain=(b-m).tolist(),paired_terrain_gain95=np.quantile(bb-mm,[.025,.975],axis=0).T.tolist())
    atomic_json(dest/'result.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
