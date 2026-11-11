"""Freeze the registration and original-solver tests before looking at new scores."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
A = R / 'audit/c79_receiver'
A.mkdir(exist_ok=True)
panel = np.load(R / 'audit/c78_clear/train_panel.npz')
rng = np.random.default_rng(790431)
terr = panel['terrain']
cal_terr = np.unique(terr[panel['calibration']])
rng.shuffle(cal_terr)
parts = {'fit':cal_terr[:80], 'select':cal_terr[80:], 'confirm':np.unique(terr[~panel['calibration']])}
records = []
for group, names in parts.items():
    for name in sorted(names):
        positions = np.flatnonzero(terr == name)
        j = int(rng.choice(positions))
        records.append(dict(group=group, terrain=str(name), pos=j, index=int(panel['ids'][j])))
assert len({r['terrain'] for r in records}) == len(records) == 612
assert [sum(r['group'] == g for r in records) for g in parts] == [80,48,484]
s = dict(base='Frozen C76; terrain-held-out C54 under C72 for internal measurements; deployed C65 under C72 for official scores.',
         panel=records, pixels='Use the existing independent 2048 uniform pixel positions for each chosen TRAIN map. Terrain split 80 fit, 48 screen, 484 confirm; all TXs for final promotion, official VAL after freeze only.',
         geometry='New observation tests: vertex quantisation in metres and relative to terrain dynamic range; smooth/sharpen or inverse raster/vertex registration; interpolation node support versus centroid bounds and subpixel origin; use alternative native-face projected solid angles and angular spans. C78 used smooth published heights with two fixed diagonal conventions.',
         second='Independently reconstruct the two-band finite-sample map with the original Sionna RT 1.1.0 as a TRAIN-side check. Use the registered scene, explicitly record angular coverage. Probe path-resolved powers or finite sampling only as input-derived candidates. No actual masks or PL as input.',
         promotion='For a full-map candidate require paired improvement >=0.08 dB on BOTH bands on the confirmation terrains and a meaningful matched-family increment; when targeting original clear use C78 clear >=0.15 and matched >=0.05. A major version requires at least 0.30 dB on BOTH bands in complete official validation vs C76. Target each <=1. A small change is not a major version.',
         physical_gpus=[1,3,4], original_freeze_sha=hashlib.sha256((R/'runs/c76_selected/freeze.json').read_bytes()).hexdigest())
dest=A/'protocol.json'
if dest.exists():assert json.loads(dest.read_text())==s
else:dest.write_text(json.dumps(s,indent=2)+'\n')
print('frozen', [(g,len(ns)) for g,ns in parts.items()])
