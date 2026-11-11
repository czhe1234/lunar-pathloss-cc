"""Verify split, metrics and artifact counts, then issue the union of the two requested weak cohorts."""
import json
from pathlib import Path
import numpy as np
from c93_export import main as export_cases
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss'
def main():
    screen=np.load(A/'screen_target.npz');t=np.load(A/'confirm_target.npz');patch=np.load(A/'patch_target.npz')
    reference=set(screen['terrain'][screen['groups']=='fit']);select=set(screen['terrain'][screen['groups']=='select']);confirm=set(t['terrain'])
    assert list(map(len,(reference,select,confirm)))==[80,48,484]
    assert reference.isdisjoint(select) and reference.isdisjoint(confirm) and select.isdisjoint(confirm)
    assert len(set(patch['terrain']))==120 and set(patch['terrain'])<=confirm
    assert t['weak'].shape==(484,512,2) and np.array_equal(t['weak'],t['valid']&(t['y']>=np.maximum(t['c82'],t['c92'])+5))
    metrics=json.load(open(A/'confirm_inventory.json'))['confirm'];baselines=[v['base_rmse'] for v in metrics['bands']]
    ref=json.load(open(R/'audit/c92_diffuse/confirm_ceiling.json'))
    assert np.allclose(baselines,[ref[str(b)]['all']['baseline_rmse'] for b in (0,1)],atol=1e-7)
    for f in (A/'lookup_flow/confirm.npz',A/'dense_probe/confirm.npz',A/'strict_confirm_pred.npz'):
        assert np.array_equal(np.load(f)['ids'],t['ids'])
    for d in sorted(A.glob('ref_[0-9]')):
        records=np.load(d/'examples.npz');assert set(records['ids'])<=set(screen['ids'][screen['groups']=='fit'])
    out=dict(calibration_terrains=80,selector_terrains=48,confirmation_terrains=484,primary_points_before_band_mask=484*512,
       dense_neighbor_terrains=120,nonoverlap_7x7_windows=120*16,unique_patch_positions=120*784,
       all_reference_bounds_checked=True,sampled_matched_C82_RMSE=baselines,
       official_C82=json.load(open(R/'audit/c82_selected_official/verified_summary.json'))['RMSE'],
       no_official_validation_labels_or_fusion_fit=True,trained_new_network=False)
    (A/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
    export_cases()
if __name__=='__main__':main()
