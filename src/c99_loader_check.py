"""Check file-order loading preserves H/TX/label/mask/reference correspondence."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c99_data import Reader,Panel
g=gpu_guard();torch.set_num_threads(4)
p=json.loads((R/'audit/c99_axial/protocol.json').read_text());reader=Reader();result={}
for kind,ids in [('fit',[p['fit'][j] for j in [0,37,113]]),('monitor',p['monitor'][:3])]:
 panel=Panel(reader,ids,reference=kind=='monitor');permutation=torch.tensor([2,0,1],device='cuda');order=[ids[j] for j in [2,0,1]]
 aa=panel.batch(permutation);bb=reader.tensors(order)
 diffs=[float((a.float()-b.float()).abs().max()) for a,b in zip(aa,bb)]
 assert max(diffs)<1e-6,diffs
 if kind=='monitor':
  expected=torch.from_numpy(np.stack([reader.c82[reader.c82index[i]] for i in order])).cuda()
  assert torch.equal(panel.reference[permutation],expected)
 result[kind]=dict(ids=ids,max_differences=diffs)
 del panel,aa,bb;torch.cuda.empty_cache()
atomic_json(R/'audit/c99_axial/loader_check.json',dict(passed=True,checks=result,GPU=g));print('LOADER CHECK PASSED',result,flush=True)
