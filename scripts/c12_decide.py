"""Apply the predeclared C12 promotion gate on internal DEV only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
c10=json.loads((R/'runs/c10_points_phase_geometry_s20261101/result.json').read_text())
p=R/'runs/c12_points_consistent_height_s20261101'
c12=json.loads((p/'result.json').read_text())
assert c12['state']=='completed'
bands={}
for b in ['415','58']:
 a=c10['bands'][b];d=c12['bands'][b]
 assert a['fit_valid_points']==d['fit_valid_points'] and a['dev_valid_points']==d['dev_valid_points']
 gain=100*(1-d['rmse_db']/a['rmse_db'])
 bands[b]={'C10_RMSE_db':a['rmse_db'],'C12_RMSE_db':d['rmse_db'],'relative_reduction_percent':gain,'C12_fit_RMSE_db':d['fit_rmse_at_selected_iteration'],'best_iteration':d['best_iteration'],'meets_3_percent_gate':gain>=3}
decision={'criterion':'At least 3 percent relative RMSE reduction in each band on unchanged sampled internal DEV, declared before training.','bands':bands,'promote_to_full_validation':all(d['meets_3_percent_gate'] for d in bands.values()),'official_validation_used_to_fit_or_select_C12':False,'source':'audit/c12_protocol.json'}
(R/'audit/c12_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
print(json.dumps(decision),flush=True)
