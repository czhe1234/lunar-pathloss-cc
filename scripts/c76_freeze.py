"""Freeze the measured TRAIN-only selection and a matching C54 internal replay configuration."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parents[1]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    original=R/'runs/c73_wide_real/full_internal/result.json';local_name='c72_observed';chosen=original
    orig=json.load(open(original));reason='C72 observed was the faster monitor-selected order'
    wider=R/'runs/c75_wide_observed'
    decision=json.load(open(wider/'promotion.json'))
    if decision['advance_to_complete_DEV']:
        assert 'joint_complete_DEV_improvement' in decision
        if decision['joint_complete_DEV_improvement']:
            alternative=wider/'full_internal/fusion/result.json';alternative_score=json.load(open(alternative))
            if all(x<y for x,y in zip(alternative_score['outside484']['chosen'],orig['outside484']['chosen'])):
                local_name=wider.name;chosen=alternative;reason='C75 passed the predeclared confirmation gate and improved both full-DEV and outside484 composite scores'
    result=json.load(open(chosen));full=result['all612'];outside=result['outside484']
    assert min(full['gain'])>.1 and min(e[0] for e in full['paired_gain95'])>0
    assert min(outside['gain'])>.1 and min(e[0] for e in outside['paired_gain95'])>0
    weight=result['common_alpha'];gate=result['gate_alpha'] if result['use_gate'] else None
    values=[v for row in gate for v in row] if gate else [weight]
    need_base=min(values)<1;need_graph=max(values)>0
    destination=R/'runs/c76_selected';out=R/'runs/.c76_build';out.mkdir(exist_ok=True);(out/'weights').mkdir(exist_ok=True)
    cfg=dict(model='C76',graph_weight=weight,gate_weights=gate,fast_descriptors=True,internal_local_run=local_name)
    if need_base:
        src=R/'runs/c65_combo_full';dst=out/'weights/base';dst.mkdir(exist_ok=True)
        for f in ['manifest.json','teacher.pt','low.pt','high.pt']:shutil.copy2(src/f,dst/f)
        cfg.update(baseline='weights/base',correction='weights/post.pt');shutil.copy2(R/'runs'/local_name/'best.pt',out/cfg['correction'])
    if need_graph:
        cfg['graph']='weights/graph.pt';shutil.copy2(R/'runs/c73_wide_real/best.pt',out/cfg['graph'])
    (out/'model.json').write_text(json.dumps(cfg,indent=2)+'\n')
    code={p.name for p in (R/'releases/C65_20261102/src').glob('*.py') if (R/'src'/p.name).is_file()}
    code.update('c57_transmission.py c66_lattice.py libc66_lattice.so c66_lattice.cpp c67_dense.py c67_physics.py c67_transport.cu libc67_transport.so c69_registered.py c70_geometry.py c70_geometry.cu libc70_geometry.so c70_centroid_lookup0.npz c70_field.py c72_native_features.py c72_native.py c73_compact.py c73_graph.py c73_visibility.cu libc73_visibility.so c74_surface.cu libc74_surface.so c74_features.py c74_inference.py'.split())
    srcsha={f:sha(R/'src'/f) for f in sorted(code)}
    manifest=dict(model='C76',selection_reason=reason,mix_source=str(chosen),internal=full,outside484=outside,
        official_VAL_used_for_selection=False,final_scoring_planned=True,website_submission=False,
        official_BASE='C65 using full official TRAIN',local_correction_training=local_name,graph_training='7979 maps / 5508 FIT terrains' if need_graph else None,
        files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()},implementation=srcsha)
    (out/'freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
    out.rename(destination);out=destination
    qa=R/'audit/c72_continuation/final_internal_interface';qa.mkdir()
    qa_cfg=dict(cfg)
    if need_base:
        qa_cfg['baseline']=str(R/'runs/c54_combo_dev');qa_cfg['correction']=str(out/cfg['correction'])
    if need_graph:qa_cfg['graph']=str(out/cfg['graph'])
    (qa/'model.json').write_text(json.dumps(qa_cfg,indent=2)+'\n')
    print(json.dumps({'cfg':cfg,'internal':full,'source':str(chosen)}),flush=True)
if __name__=='__main__':main()
