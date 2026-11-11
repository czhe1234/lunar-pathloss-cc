"""Choose diagnostic windows and disjoint confirmation terrains without opening PL or masks."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment';P=R/'audit/c93_joint_miss'
old=json.loads((P/'exact_windows.json').read_text())['maps']
f=np.load(P/'patch_features.npz');pub=json.loads((P/'patches.json').read_text());names=f['names'].tolist()
old_ids={e['index'] for e in old};extra=[]
for j,i in enumerate(f['ids']):
    v=f['x'][j].reshape(16,49,-1);d=np.expm1(v[:,:,names.index('range')]);clear=v[:,:,names.index('vis')]>.99
    ok=(d.min(1)>=65)&(clear.mean(1)>=.80);arms=np.array(pub['maps'][str(i)]['arm'])
    rng=np.random.default_rng([93899,int(i)]);sel=[]
    for m in ('uniform','grazing'):
        avail=np.flatnonzero(ok&(arms==m))
        if len(avail):sel.append(int(rng.choice(avail)))
    if sel and int(i) not in old_ids:
        extra.append(dict(index=int(i),patch_map=j,windows=sel,arms=arms[sel].tolist()))
rng=np.random.default_rng(950431);pick=set(rng.choice(len(old),16,replace=False).tolist())
screen=[e for j,e in enumerate(old) if j in pick];rest=[e for j,e in enumerate(old) if j not in pick]
configs={
 'base':{},
 'tx_xy_vertex':{'txxy':'vertex'},
 'tx_xy_half':{'txxy':'half'},
 'tx_z_original_row':{'txz':'row'},
 'tx_z_continuous':{'txz':'continuous'},
 'tx_z_surface_floor':{'txz':'surface'},
 'ground_antidiag':{'ground':'anti'},
 'rx_antidiag':{'rx':'anti'},
 'mesh_antidiag':{'ground':'anti','rx':'anti'},
 'mesh_delaunay':{'ground':'delaunay','rx':'delaunay'},
 'mesh_step1':{'grid':'integer'},
 'rx_z0p5':{'rxheight':.5},
 'rx_z1p5':{'rxheight':1.5},
 'rx_normal1':{'rxoffset':'normal'},
}
out=dict(screen=screen,extension=rest,confirm=extra,configs=configs,
         source='H/TX-only C93 window selection; screen16 + extension44 are the previously studied C94 terrains; confirm58 exclude all C94 terrains. Historical C93 labels existed but are not used for C95 candidate selection.',
         decision='Select on screen, verify on extension, freeze before confirm. Score weak location and power with false positives, not only reductions of bound violations.')
assert len(extra)==58 and len(screen)==16
assert not ({e['index'] for e in extra}&old_ids)
p=A/'plan.json'
if p.exists():
    saved=json.loads(p.read_text())
    assert all(saved[k]==out[k] for k in ('screen','extension','confirm'))
    assert all(saved['configs'][k]==v for k,v in configs.items())
else:p.write_text(json.dumps(out,indent=2)+'\n')
print('screen,extension,confirm',len(screen),len(rest),len(extra),'initial_variants',len(configs))
