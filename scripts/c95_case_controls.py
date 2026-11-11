"""Compare prespecified interventions on the three posthoc first-hit counterexample sites."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment'
p=json.loads((A/'plan.json').read_text());events=json.loads((A/'component_summary.json').read_text())['first_hit_examples']
configs=['base','ground_antidiag','rx_antidiag','mesh_antidiag','tx_xy_vertex','tx_xy_half','tx_z_original_row','tx_z_continuous','tx_z_surface_floor','mesh_step1','rx_z0p5','rx_z1p5','rx_normal1','phase_plus1e7','phase_minus1e7','tx_z_minus0p5','tx_z_plus0p5','mesh_delaunay','pixel_x_-0.5','pixel_x_-1','pixel_x_+1','pixel_y_-1','pixel_y_+1','topo_center0','topo_center2','topo_positive256_0','topo_positive256_1','topo_positive255_1']
constant=20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458)
rows={}
for cfg in configs:
    rr=[]
    for event in events:
        i,sample,b,r,c,y,*_=event;path=A/'cases'/f'{i}_{cfg}.npz';assert path.exists(),str(path)
        z=np.load(path);k=np.flatnonzero(z['pixel']==r*256+c)[0];n=z['native'][k];w=z['weights'][k];active=w>1e-7
        positive=bool(np.all(n[active]>0))
        upper=float((-10*np.log10(n[active])).max()+constant[b]) if positive else None
        direct=float(-10*np.log10(z['average'][k])+constant[b]) if z['average'][k]>0 else None
        rr.append(dict(index=i,sample=sample,band=b,row=r,col=c,label=y,all_positive=positive,
             weakest_bound=upper,linear_direct_PL=direct,hard=positive and y-upper>5))
    rows[cfg]=dict(remaining_hard=sum(x['hard'] for x in rr),events=rr)
out=dict(events=len(events),positions=3,models=rows,
     warning='Cases were selected after confirmation revealed first-hit contradictions. Removing a bound violation is not accurate power prediction. These case controls are not independent validation or model selection evidence.')
(A/'case_controls.json').write_text(json.dumps(out,indent=2)+'\n')
for c,z in rows.items():print(c,z['remaining_hard'],[None if v['weakest_bound'] is None else round(v['label']-v['weakest_bound'],2) for v in z['events']])
