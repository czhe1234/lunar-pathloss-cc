import json,numpy as np,torch
from c21_train import setup,model,evaluate_loaded,A
from cc_core import ROOT,atomic_json
setup(20261119);net=model(False);ck=torch.load(ROOT/'runs/c21_small_a_s20261119/best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model'])
dat=tuple(torch.from_numpy(np.load(A/f'small_{n}.npy')).cuda() for n in ['x','tx','fs','y','m']);r={};pred=[]
for amp in [True,False]:
 met,p=evaluate_loaded(net,dat,1024,True,amp);r['BF16' if amp else 'FP32']=met;pred.append(np.array(p))
r['prediction_difference_rms_db']=np.sqrt(np.mean((pred[0]-pred[1])**2,axis=(0,2,3))).tolist();atomic_json(A/'precision_check.json',r);print(json.dumps(r),flush=True)
