import json,hashlib,torch
from pathlib import Path
from c21_train import setup,model
from c21_pair import full_dev
from cc_core import ROOT,atomic_json,check_source_unchanged
r=ROOT/'runs/c21_pair_s20261123';cfg=json.loads((r/'config.json').read_text());assert (r/'complete.json').exists()
setup(cfg['seed']);check_source_unchanged();result={}
for label,geo in [('A',False),('B',True)]:
 net=model(geo);ck=torch.load(r/f'best_{label}.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);step=ck['step'];del ck
 met=full_dev(net,cfg['fit_panel'],r/f'fit_panel_{label}')
 met['scope']='fixed128 FIT terrain maps, complete valid pixels; selected main model, not small4-map sanity'
 atomic_json(r/f'fit_panel_{label}/metrics.json',met);result[label]={'selected_step':step,**met};del net;torch.cuda.empty_cache()
atomic_json(r/'fit_panel_complete.json',result)
