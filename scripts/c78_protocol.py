"""Freeze analysis rows, pixels, interventions and thresholds before reading outcome data."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c78_clear';A.mkdir(exist_ok=False)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));internal=json.load(open(R/'runs/c73_wide_real/full_internal/indices.json'))
selected=set(json.load(open(R/'audit/c66_transport/panels.json'))['dev_terrains'])
terrain=[rows[i]['terrain_id'] for i in internal];cal=np.isin(terrain,list(selected))
assert len(set(terrain))==612 and len(set(np.array(terrain)[cal]))==128 and len(set(np.array(terrain)[~cal]))==484
valrows=list(csv.DictReader((R.parent/'data/val_index.csv').open()))
def pixels(ids,seed):
    return np.stack([np.sort(np.random.default_rng(np.random.SeedSequence([seed,i])).choice(65536,2048,replace=False)).astype('int32') for i in ids])
np.savez(A/'train_panel.npz',ids=internal,pixels=pixels(internal,780621),calibration=cal,terrain=np.array(terrain))
vals=list(range(len(valrows)));np.savez(A/'val_panel.npz',ids=vals,pixels=pixels(vals,780622),terrain=np.array([r['terrain_id'] for r in valrows]))
protocol=dict(parent='C76 frozen 50% C72 local + 50% C73 real graph',freeze_sha256=hashlib.sha256((R/'runs/c76_selected/freeze.json').read_bytes()).hexdigest(),
    partitions='Internal official-TRAIN experiment: original 128 terrains (all TXs) calibrate diagnostic coefficients, other 484 terrains confirm once. Uniform sample of 2048 pixels without replacement per map, fixed independently of targets. Full published VAL is a separate frozen-deployment descriptive check only, after the internal training decision; no VAL calibration or model selection.',
    outcomes='Per-band masked global RMSE, share of all originally-clear SSE and pixels, rates and shares for absolute errors >=3 and >=5 dB, signed bias (true minus prediction). Complementary bands use their own masks; paired-band summaries use joint mask. Group uncertainty by terrain.',
    geometry='Original clear = C76 compact channel 16 > .99 (weighted 9 area samples per receiver triangle). On the identical pixels vary only area sampling, triangle diagonal, TX z convention, TX/RX height and half-pixel TX offsets. Primary switch = any physically bracketed alternative drops below .90. Report arbitrary stricter >=.99 separately. Coarsened DTM is stress only. Prove a conservative subset clear for every point of every supported triangle via swept ray boxes and terrain-height upper bounds. Distinguish failure to certify from an observed visibility flip.',
    radiometry='Total pathloss is observed, per-path truth is unavailable. Compute geometric direct reference and the weakest free-space direct power allowed by the bracket; partition observed power above/below direct bounds with 1/3/6 dB margins. A deficit is evidence against an intact continuum direct component with only additive nonnegative multipath, not proof of geometric occlusion. Split visibility uncertainty from additive specular/diffuse power. Compare no extra power, specular only, diffuse only and both; compare high-resolution one-bounce estimates and source-weighted diffuse integration. Cross-frequency common/difference errors and physically matched shuffles are controls.',
    diagnostic_fit='Fit only small fixed linear diagnostic corrections on the calibration terrain set. Always apply the frozen coefficients to distinct confirmation terrains. Geometry-only and power-only incremental controls share the same original point set; true-vs-shuffled label-free features test conditional spatial pairing. A label-derived subgroup is diagnostic, never an inference input.',
    training_gate='Run a further neural training candidate only if a deployable diagnostic improves original-clear RMSE by at least 0.15 dB and full-map RMSE by 0.08 dB in BOTH bands on the 484 confirmation terrains, with positive paired 95% lower bounds on full-map gains and at least 0.05 dB clear incremental gain over the matched control from a specific physical family. Claimed major version still requires 0.30 dB in both bands vs C76; target remains each official VAL RMSE <=1 dB.',
    approved='Compute on physical 1/3/4 only under run_gpu.sh full UUID guard. No website submission. Raw data only from top-level release. No official VAL updates.')
(A/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print('maps/terrains',len(internal),int(cal.sum()),int((~cal).sum()),len(valrows))
