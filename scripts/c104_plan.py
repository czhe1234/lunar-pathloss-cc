import json,hashlib,torch,numpy as np
from pathlib import Path
from src.c104_train import create,A,NORM,digest
p=json.loads((A/'priors.json').read_text());nm=np.load(NORM);initial={}
for mode in ['mse','state']:
 n=create(mode,torch.tensor(nm['mean']),torch.tensor(nm['std']),torch.tensor(p['prior'],dtype=torch.float32),torch.tensor(p['proposal'],dtype=torch.float32),torch.tensor(p['class_mean'],dtype=torch.float32))
 initial[mode]=digest(n)
protocol=dict(question='Do C82 signed rare residual states become predictable with direct class and conditional-amplitude supervision?',
 arms=['mse','state'],features=235,steps=2000,batch=16384,parameters=sum(x.numel() for x in n.parameters()),initial_state_sha256=initial,
 inputs='C101coarse169 + C103terrain66. Same features/backbone, all head weights zero; bias gives functional C82 at initialization.',
 sampler='Half each band; valid signed3class proposal q=sqrt(pi)/sum; withinclass weighted CDF. Same seed104073. True maskedMSE importance pi/q. No mask/label input.',
 state='prior-corrected softmax(logits+log(pi/q)); signed conditional means constrained inside class; expectedmean prediction. MSE/25 +.25proposalCE +.25conditionalMSE/25',
 control='Same sampling,backbone,12outputlayout; direct residualMSE/25, unused output slots. Seed104041.',
 priors=p,fit_terrains=1984,fit_points=4323840,monitor_terrains=128,confirmation_terrains=484,confirmation_points_permap=2048,
 selection='Min jointmean monitorRMSE incl unchangedC82; freeze checkpoint/model before confirmation',
 promotion='Both bands gain>=.30dB versus same C82 pixels and pairedterrain95percentlower>0; fullnative check before fullTRAIN.',
 official_target='bothofficialfixedVALRMSE<=1dB',official_val_used=False,GPU={'mse':1,'state':3,'check_and_confirmation':4},
 source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in ['src/c104_train.py','audit/c104_tail/priors.json']})
assert not (A/'protocol.json').exists()
(A/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps({'parameters':protocol['parameters'],'initial':initial}))
