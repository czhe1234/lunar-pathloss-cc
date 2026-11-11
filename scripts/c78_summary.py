"""Cross-check the reported denominators and collect the frozen C78 results."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/c78_clear'
BANDS = (415e6, 5.775e9)
NAMES = ('415 MHz', '5.775 GHz')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cross_tab(split, panel, selection, n):
    # Stream the geometry shards. Categorise the exact same masked C76 pixels in both tests.
    ref = np.load(AUDIT / (split + '_reference.npy'), mmap_mode='r')
    counts = np.zeros((2, 3, 2), np.int64)
    sse = np.zeros_like(counts, dtype=np.float64)
    seen = np.zeros(len(selection), bool)
    for shard in sorted(AUDIT.glob(split + '_geom_*of*')):
        pos = np.load(shard / 'positions.npy')
        geom = np.load(shard / 'geometry.npy', mmap_mode='r')
        assert not seen[pos].any()
        seen[pos] = True
        for step in range(0, len(pos), 24):
            ix = pos[step:step + 24]
            keep = selection[ix]
            if not keep.any():
                continue
            geo = geom[step:step + len(ix)][keep]
            rr = ref[ix[keep]]
            frac = geo[..., :11].min(axis=2)
            group = np.where(frac < .90, 0, np.where(frac >= .99, 2, 1))
            for b, freq in enumerate(BANDS):
                clear = (rr[..., 6] > .99) & (rr[..., 4 + b] > .5)
                limit = 20 * np.log10((4 * np.pi * freq / 299792458) * geo[..., 17].clip(.001))
                weak = rr[..., b] - limit > 3
                cat = (group * 2 + weak)[clear]
                error = (rr[..., b] - rr[..., 2 + b])[clear].astype(np.float64)
                counts[b] += np.bincount(cat, minlength=6).reshape(3, 2)
                sse[b] += np.bincount(cat, weights=error ** 2, minlength=6).reshape(3, 2)
    assert seen.all() and seen.sum() == len(panel['ids'])
    np.testing.assert_array_equal(counts.sum(axis=(1, 2)), n)
    return counts, sse


def summarize(split, official):
    panel = np.load(AUDIT / (split + '_panel.npz'))
    internal = split == 'train'
    selection = ~panel['calibration'] if internal else np.ones(len(panel['ids']), dtype=bool)
    geom = read(AUDIT / (split + '_geometry_attribution.json'))
    grp = geom['confirm484' if internal else 'all_maps']
    counter = read(AUDIT / (split + '_counterfactual_results.json'))['confirm484' if internal else 'actual680']
    marker = read(AUDIT / (split + '_matched_and_marker.json'))
    checks = read(AUDIT / (split + '_uncertainty_checks.json'))
    cert = checks['geometry_check']
    assert cert['blocked_rays'][0] == 0 and cert['blocked_rays'][1] > 0
    assert tuple(counter[k] for k in ('terrains', 'maps')) == ((484, 1934) if internal else (680, 2330))
    assert len(np.unique(panel['terrain'][selection])) == counter['terrains']
    if internal:
        assert not set(panel['terrain'][selection]) & set(panel['terrain'][~selection])
    counts, sse = cross_tab(split, panel, selection, [row[1] for row in counter['counts']])
    report = dict(terrains=counter['terrains'], maps=counter['maps'], locations_per_map=panel['pixels'].shape[1],
                  nominal_bin_disagreement_all_maps=geom['nominal_bin_disagreement'],
                  matched_diagnostics_fit='128 official TRAIN terrains; frozen coefficients',
                  geometry_check_64_terrains=cert, quadrature_sensitivity_48_maps=checks['numeric_sensitivity'],
                  paired_band_residuals_wide_certificate=grp['paired']['wide_cert'], bands={})
    categories = ('switch90', 'mid90to99', 'stable99')
    for b, band in enumerate(NAMES):
        bg = grp['bands'][str(b)]
        nd = counts[b]
        ss = sse[b]
        np.testing.assert_array_equal(nd.sum(axis=1), [bg[k]['n'] for k in categories])
        np.testing.assert_allclose(ss.sum(axis=1) / ss.sum(), [bg[k]['sse_share'] for k in categories],
                                   rtol=0, atol=2e-7)
        assert nd[:, 1].sum() == bg['all_deficit3']['n']
        assert nd[2, 1] == bg['stable_deficit3']['n']
        assert abs(bg['original_clear']['rmse'] - counter['results']['C76']['rmse'][b][1]) < 1e-6
        if not internal:
            assert abs(bg['all_valid_RMSE'] - official['RMSE'][b]) < .003
        pieces = {}
        for j, key in enumerate(categories):
            pieces[key] = dict(pixels_by_weak_flag=nd[j].tolist(), SSE_by_weak_flag=ss[j].tolist(),
                               share_of_all_clear_SSE_by_weak_flag=(ss[j] / ss.sum()).tolist())
        rows = {k: {f: bg[k][f] for f in ('n', 'pixel_share', 'sse_share', 'rmse', 'huge_sse_share')}
                for k in ('original_clear', *categories, 'nominal_certificate', 'two_mesh_altitude_certificate',
                          'wide_certificate', 'all_deficit3', 'wide_deficit3', 'stable_deficit3')}
        controls = {}
        for model in ('geometry', 'power', 'geometry_power', 'geometry_sheet_power'):
            result = counter['results'][model]
            controls[model] = dict(full_clear_gains=result['gain'][b][:2], terrain_95=result['full_clear_gain95'][b])
        for comp in ('geometry_power__vs__geometry', 'geometry_power__vs__geometry_shuffled_power',
                     'geom_spec__vs__geometry', 'geom_diff__vs__geometry', 'geometry_power__vs__geometry_old_power'):
            result = counter['contrasts'][comp]
            controls[comp] = dict(full_clear_gains=result['delta'][b][:2], terrain_95=result['full_clear95'][b])
        report['bands'][band] = dict(sampled_valid_pixels=counter['counts'][b][0], sampled_all_RMSE=bg['all_valid_RMSE'],
            original_clear_share_of_all_SSE=bg['original_clear_vs_all_SSE'], original_clear_subgroups=rows,
            terrain_cluster_intervals=checks['cluster_intervals'][str(b)], geometry_by_weak_flag=pieces,
            raw_physics_rmse_wide=dict(zip(counter['physics_RMSE_order'], counter['raw_physics_RMSE'][b][5])),
            intact_direct_nonnegative_extra_capped_floor_share_wide=counter['capped_direct_floor_overlap_share'][b][5],
            RMSE_reductions_vs_frozen_controls=controls, weak_marker_wide=marker[str(b)]['wide'],
            matched_wide=marker[str(b)]['matched_wide_certificate'])
    return report


def main():
    official = read(ROOT / 'audit/c76_final_delivery.json')['official']
    protocol = read(AUDIT / 'protocol.json')
    decision = read(AUDIT / 'internal_decision.json')
    replay = read(AUDIT / 'official_replay_complete.json')
    assert digest(ROOT / 'runs/c76_selected/freeze.json') == protocol['freeze_sha256']
    assert digest(AUDIT / 'internal_coefficients.json') == decision['coefficients_sha256'] == replay['coefficient_hash_unchanged']
    assert not decision['train_new_network'] and not replay['official_USED_for_new_model_selection']
    prov = read(ROOT / 'audit/c10_official_scorer_20261101/provenance.json')
    scorer = {}
    for name in ('metrics.py', 'score_submission.py'):
        sha = digest(ROOT / 'audit/c10_official_scorer_20261101/official' / name)
        assert sha == prov['sources'][name]['sha256']
        scorer[name] = sha
    gpus = {i: u for i, u in [(1, 'GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009'),
                               (3, 'GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'),
                               (4, 'GPU-70b6fceb-c7c4-9467-3c81-097c9284c160')]}
    for split in ('train', 'val'):
        for kind in ('geom', 'power', 'sheet'):
            groups = [read(p) for p in AUDIT.glob(split + '_' + kind + '_*of*/complete.json')]
            assert len(groups) == 3 and {entry['gpu']['uuid'] for entry in groups} == set(gpus.values())
            assert sum(entry['maps'] for entry in groups) == {'train':2562, 'val':2330}[split]
    out = dict(title='C78 attribution of original C76 clear-region errors', official_C76=official,
               protocol=protocol, decision=decision, frozen_official_replay=replay, authorized_gpu_uuids=gpus,
               source_sha256=dict(c76_freeze=protocol['freeze_sha256'], diagnostic_coefficients=decision['coefficients_sha256'],
                                  C78_protocol=digest(AUDIT / 'protocol.json'), official_scorer=scorer),
               cross_tab_note='Geometry is partitioned by minimum tested visibility. Weak flag column order [not weaker by >3 dB, weaker by >3 dB] versus the weakest allowed intact direct power. SSE denominator is all original clear valid pixels in that band.',
               flat_plane_checks=[read(p) for p in sorted(AUDIT.glob('power_check_GPU-*.json'))],
               internal_confirm484=summarize('train', official), actual_official_val680=summarize('val', official))
    dest = AUDIT / 'summary.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(dest, digest(dest))
    for band, r in out['actual_official_val680']['bands'].items():
        print(band, {key: [round(x, 5) for x in values['share_of_all_clear_SSE_by_weak_flag']]
                     for key, values in r['geometry_by_weak_flag'].items()})


if __name__ == '__main__':
    main()
