import json,hashlib
from pathlib import Path
from src.c105_context import create,digest,A
R=Path.cwd();A.mkdir(exist_ok=True);assert not (A/'protocol.json').exists()
m=create()
fit=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text())
monitor=json.loads((R/'audit/c101_residual/monitor/done.json').read_text())['ids']
p=dict(question='Can end-to-end native C82 residual fields recover rare structure with spatial context, beyond frozen-query readouts?',
 arms=['local','context'],fit=fit,monitor=monitor,epochs=6,batch=16,
 initial_state_sha256=digest(m),parameters=sum(x.numel() for x in m.parameters()),seed=105041,data_order_seed=105073,
 parent_checkpoint='runs/c99_global/best.pt epoch11; body warmstarted identically; headzero/newinputcolumns0; functional initialC82',
 parent_sha256=hashlib.sha256((R/'runs/c99_global/best.pt').read_bytes()).hexdigest(),
 hypothesis='C101-C104point/frozenreadoutfailed; wholeC82excessimage offers surrounding radiomap structure and trainable nativefilters.',
 small_check='16FITmaps,200updates,batch8;no official/devlabels;initialC82. Largepilot allowed only if bothbandfitSSEreduce>=25percent. Capacitychecknotgeneralizationproof.',
 budget='6epochsmax,patience2;fullnativeTRAINtargets on2815maps/1984terrains;monitor628maps/128terrains. Both arms same data/init/budget.',
 selection='minfullmonitor jointmean inclC82; freeze before same484terrain/1934map/2048point confirm',
 promotion='bothsamepanelconfirmation gains>=.30dB and positiveterraingroup95percentlower, thenfullnative checkbeforefullTRAIN',
 official_val_used=False,target='bothofficialbandRMSE<=1dB',GPU={'small':4,'local':1,'context':3})
(A/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
print({'fit':len(fit),'monitor':len(monitor),'parameters':p['parameters'],'hash':p['initial_state_sha256']})
