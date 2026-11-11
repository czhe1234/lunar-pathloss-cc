"""Fixed nested 1x / 8x reference libraries on the same 80 calibration terrains."""
import json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss'
q=np.load(A/'screen_target.npz');fit=q['groups']=='fit';ids=q['ids'][fit];pp=q['pixels'][fit];out=[]
for i,old in zip(ids,pp):
    seen=np.ones(65536,bool);seen[old]=False
    fresh=np.random.default_rng([93784,int(i)]).choice(np.flatnonzero(seen),4096-len(old),replace=False)
    out.append(np.r_[old,fresh])
x=np.stack(out).astype(np.int32);assert len(ids)==80 and all(len(np.unique(v))==4096 for v in x)
np.savez_compressed(A/'ref_plan.npz',ids=ids,pixels=x)
