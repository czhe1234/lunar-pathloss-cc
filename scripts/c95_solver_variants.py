"""Copy pinned source and change only receiver-ray advancement. Original source is never edited."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment/solver_variants';A.mkdir(exist_ok=True)
p=R/'.rt_packages/sionna/rt/radio_map_solvers/radio_map_solver.py';s=p.read_text();needle='ray_mp_int = si_mp.spawn_ray(d=ray.d)'
assert s.count(needle)==1
replacements={
'normal':'''ray_mp_int = mi.Ray3f(
                dr.fma(dr.detach(dr.mulsign(self.c95_epsilon *
                    dr.select(self.c95_position_scaled,
                        1.0 + dr.max(dr.abs(si_mp.p-self.c95_origin)), 1.0),
                    dr.dot(si_mp.n, ray.d))), dr.detach(si_mp.n), si_mp.p),
                ray.d)''',
'along':'''ray_mp_int = mi.Ray3f(dr.fma(self.c95_epsilon, ray.d, si_mp.p), ray.d)'''
}
manifest={}
for mode,line in replacements.items():
    new=s.replace(needle,line)
    path=A/f'radio_map_solver_{mode}.py';compile(new,str(path),'exec');path.write_text(new)
    manifest[mode]=dict(original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),changed_statement=needle,new_statement=line)
(A/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('one receiver advance statement changed per variant')
