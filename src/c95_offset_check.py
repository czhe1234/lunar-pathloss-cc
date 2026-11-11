"""Measure the pinned Mitsuba receiver-intersection offset, independent of dataset labels."""
from pathlib import Path
import json,sys
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from c79_sionna import mi,dr,rt
si=mi.SurfaceInteraction3f();si.n=mi.Vector3f(0,0,1)
d=dr.normalize(mi.Vector3f(1,0,.001));out=[]
for p in ([0,0,0],[128,0,0],[256,0,0],[0,0,1024]):
    si.p=mi.Point3f(*p);rr=si.spawn_ray(d)
    out.append(dict(p=p,ray_o=rr.o.numpy().ravel().tolist(),delta=(rr.o-si.p).numpy().ravel().tolist()))
s=dict(sionna=rt.__version__,mitsuba=mi.__version__,drjit=dr.__version__,backend=mi.variant(),records=out,
       empirical_RayEpsilon=out[0]['delta'][2],
       interpretation='Rigid translation leaves physical geometry unchanged but changes normal spawning offset; at grazing incidence this changes later receiver crossings.')
(R/'audit/c95_alignment/offset_definition.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s))
