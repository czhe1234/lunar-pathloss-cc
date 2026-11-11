"""Freeze the TRAIN-selected C82 readout and verify every inherited C76 dependency."""
from pathlib import Path
import hashlib,json,os,shutil
R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
decision=json.loads((A/'blend_decision.json').read_text())
assert all(row[0]>0 for row in decision['gain95'][0])
old=R/'runs/c76_selected';new=R/'runs/c82_selected';new.mkdir(exist_ok=True)
parent=json.loads((old/'freeze.json').read_text())
for rel,h in parent['files'].items():
    assert sha(old/rel)==h
    if rel=='model.json':continue
    dst=new/rel;dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.link(old/rel,dst)
    assert sha(dst)==h
config=json.loads((old/'model.json').read_text());config['model']='C82'
(new/'model.json').write_text(json.dumps(config,indent=2)+'\n')
shutil.copyfile(A/('blend_'+decision['mode'])/'coeff.json',new/'spectrum.json')
impl=dict(parent['implementation'])
for name,h in impl.items():assert sha(R/'src'/name)==h
for name in ('c82_blend.py','c82_inference.py','c82_predict.py','c76_predict.py'):impl[name]=sha(R/'src'/name)
f=dict(model='C82',parent_freeze=sha(old/'freeze.json'),official_VAL_used_for_selection=False,
       selection='Fixed C76 plus TRAIN-only multiscale local/graph readout; selected full on 48 terrains. No official validation used in this fitting/selection.',
       major_internal=False,confirmation=decision,files={k:sha(new/k) for k in [*parent['files'],'spectrum.json']},implementation=impl)
(new/'freeze.json').write_text(json.dumps(f,indent=2)+'\n')
print(sha(new/'freeze.json'),decision['gain'])
