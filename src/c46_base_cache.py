"""Input-only C38 internal-FIT teacher cache; no labels or masks are opened."""
from pathlib import Path
import sys,json,csv,argparse,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard,atomic_json
from c34_inference import Predictor
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
proto=json.loads((R/'audit/c44_dense_profile_20261102/protocol.json').read_text())
ids=proto['fit64_ids'][a.shard::2];rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
o=R/'features/c46_base';o.mkdir(exist_ok=True)
dest=o/f'fit_part{a.shard}.npy';assert not dest.exists()
out=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(ids),2,256,256))
net=Predictor(R/'runs/c38_1024_continue')
for j,i in enumerate(ids):
 r=rows[i];out[j]=net.predict(np.asarray(H[int(r['hm_row'])]),int(r['tx_row']),int(r['tx_col']))
 if (j+1)%32==0:print(json.dumps({'shard':a.shard,'maps':j+1}),flush=True)
out.flush()
atomic_json(o/f'fit_part{a.shard}.json',{'indices':ids,'input_only':True,'labels_masks_read':False,'GPU':net.device,
 'teacher':'C38 internal-FIT model, never trained on DEV terrains','checkpoint_sha256':hashlib.sha256((R/'runs/c38_1024_continue/best.pt').read_bytes()).hexdigest()})
print('cache complete',a.shard,flush=True)
