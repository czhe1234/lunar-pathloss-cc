import json,hashlib
from pathlib import Path
from src.c105_context import create,digest
R=Path.cwd();A=R/'audit/c106_fit'
old=json.loads((R/'audit/c105_context/protocol.json').read_text())
small=json.loads((R/'audit/c105_context/small.json').read_text())
net=create()
p=dict(question='Are16map fitting failures due to optimization duration,weightdecay,orLRschedule with unchanged architecture?',
 tier='auxiliary/dev;followedbyfixedinternalholdout',arms=['original','no_decay','lr_decay'],
 fit16=old['fit'][:16],monitor=old['monitor'],confirmation='same484terrain1934map2048pointpanel',
 baseline16_rmse=small['initial']['rmse'],max_steps=4000,batch=8,initial_state_sha256=digest(net),
 parameters=sum(x.numel() for x in net.parameters()),sampling_seed=105073,
 primary_metrics=['fit.rmse','fit.per_map_band_rmse','fit.residual_group_rmse'],
 fit_acceptance='all32permapbandRMSE<.98andbothaggregateRMSE<.95;firstpassingcheckpoint;noheldoutselection',
 heldout_primary='maskedRMSEbothbands;positiveC82residual>=5locationPRandpowerRMSE;fixed5predictionthreshold',
 promotion='bothbandsimprove>=.30dBvssameC82pixelsandpairedterrain95lowerpositive;fullnativebeforefullTRAIN',
 GPU={'original':1,'no_decay':3,'lr_decay':4},official_val_used=False,
 source_sha256={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in ['src/c106_fit.py','src/c105_context.py','src/c99_axial.py']})
assert not (A/'protocol.json').exists();(A/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
print({'fit16':p['fit16'],'initial_hash':p['initial_state_sha256'],'parameters':p['parameters']})
