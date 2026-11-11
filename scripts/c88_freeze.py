"""Freeze the experiment before looking at its results. No official VAL in development."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1];A=R/'audit/c88_mesh'
A.mkdir(exist_ok=True)
protocol={
 'base':'C82, actual TRAIN-only internal counterpart, frozen spectrum.json; same terrain aggregation and masks in all arms.',
 'data':'Official TRAIN only. C79 fixed 80 terrain calibration, 48 selection, 484 confirmation; choose on selection and only evaluate the frozen winner on confirmation. FIT excludes all 612. No validation Labels as training features.',
 'registered_operator':'Test for a spatially fixed subpixel receiver response from TRAIN. Learn local slope/area/interpolation differences per absolute cell, across independent terrains and TX; compare tied, spatially matched and permuted registration with identical fitting procedure. Forbid terrain ID, label and validity mask at inference.',
 'true_mesh':'Test physically different resolutions 128,192,256,384 and fixed, checkerboard, Delaunay, and shorter 3D diagonal. Trace true surface to native receiving faces, pool subfaces and project to raster. Nominal vs changed resolution/topology; no per-scene labels. Use both bands and common versus differential terms.',
 'gate':'Advance to full official VAL if both complete-map confirmation gains >=0.08 dB over C82; call major only if complete official improvement >=0.30 dB in each band. A failed gate ends that branch, not the 1 dB objective.',
 'sources':{k:hashlib.sha256((R/k).read_bytes()).hexdigest() for k in ['runs/c82_selected/freeze.json','audit/c79_receiver/protocol.json','audit/internal_split.json']}}
dest=A/'protocol.json'
if dest.exists():assert json.loads(dest.read_text())==protocol
else:dest.write_text(json.dumps(protocol,indent=2)+'\n')
