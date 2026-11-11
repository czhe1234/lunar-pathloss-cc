"""Score the prespecified physical mixture on the frozen internal splits."""
import csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
DATA=R.parent/"data"
def atomic_json(path,obj):
    tmp=path.with_suffix(".json.tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");tmp.replace(path)
def summarize(ids,s,control,root,local_name="C72"):
    plan=json.load(open(R/'runs/c69_registered_spatial/protocol.json'))
    panel=json.load(open(R/'audit/c66_transport/panels.json'));rows=list(csv.DictReader((DATA/'train_index.csv').open()))
    terrain=np.array([rows[i]['terrain_id'] for i in ids]);monitor=np.isin(ids,plan['monitor']);confirm=np.isin(ids,plan['confirmation'])
    outside=~np.isin(terrain,panel['dev_terrains']);assert monitor.sum()==len(plan['monitor']) and confirm.sum()==len(plan['confirmation'])
    a=s[monitor].sum(0);den=a[...,4].sum(1)
    def mix(v,alpha):
        return ((1-alpha)*v[...,1]+alpha*v[...,2]-alpha*(1-alpha)*v[...,3]).clip(0)
    def rm(v,alpha):return np.sqrt(mix(v,alpha).sum(-1)/v[...,4].sum(-1))
    grid=np.linspace(0,1,9);shared=float(min(grid,key=lambda x:rm(a,x).mean()))
    prior=.01*a[...,3].sum(1,keepdims=True)
    gated=np.clip((.5*(a[...,1]-a[...,2]+a[...,3])+prior*shared)/(a[...,3]+prior+1e-15),0,1)
    cc=s[confirm].sum(0);improve=rm(cc,shared)-rm(cc,gated)
    useg=bool((improve>=.03).all());coef=gated if useg else shared
    report=dict(common_alpha=shared,gate_alpha=gated.tolist(),gate_confirm_gain_over_common=improve.tolist(),use_gate=useg,
        model=local_name+' correction + real C73 graph',alpha_means_graph_fraction=True,official_VAL_used=False)
    rng=np.random.default_rng(74011)
    for key,sel in [('monitor64',monitor),('confirmation64',confirm),('all612',np.ones(len(s),bool)),('outside484',outside)]:
        v=s[sel].sum(0);bd=v[...,4].sum(1)
        mixrm=rm(v,coef);base=np.sqrt(v[...,0].sum(1)/bd)
        unique,inv=np.unique(terrain[sel],return_inverse=True)
        vals=np.stack((s[sel,...,0].sum(2),mix(s[sel],coef).sum(2),s[sel,...,4].sum(2)),axis=1)
        pergroup=np.zeros((len(unique),3,2));np.add.at(pergroup,inv,vals)
        boot=pergroup[rng.integers(len(unique),size=(2000,len(unique)))].sum(1)
        delta=np.sqrt(boot[:,0]/boot[:,2])-np.sqrt(boot[:,1]/boot[:,2])
        report[key]=dict(terrains=len(unique),maps=int(sel.sum()),valid=bd.astype(int).tolist(),C54=base.tolist(),C72=rm(v,0).tolist(),
            real_graph=rm(v,1).tolist(),shuffled_graph=np.sqrt(control[sel].sum(0)/bd).tolist(),fixed_common=rm(v,shared).tolist(),
            gated=rm(v,gated).tolist(),chosen=mixrm.tolist(),gain=(base-mixrm).tolist(),paired_gain95=np.quantile(delta,[.025,.975],axis=0).T.tolist())
    if local_name!='C72':
        for key in ['monitor64','confirmation64','all612','outside484']:report[key][local_name]=report[key].pop('C72')
    atomic_json(root/'result.json',report);print(json.dumps(report),flush=True)
