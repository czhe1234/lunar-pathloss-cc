"""Independent CPU score reimplementation from saved H/TX-only direct maps."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c131_transport';D=R.parent/'data'
plan=json.loads((A/'protocol.json').read_text());ids=plan['monitor'];rows=list(csv.DictReader((D/'train_index.csv').open()))
yarr=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')]
marr=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r');bi={int(i):j for j,i in enumerate(json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids'])}
x=np.load(A/'direct_n16_monitor/power.npy',mmap_mode='r')
scale=(299792458/(4*np.pi*np.array([415e6,5.775e9])))**2
num=np.zeros(2,np.int64);ss=np.zeros(2);bss=np.zeros(2);bys={g:dict(n=np.zeros(2,np.int64),s=np.zeros(2),base_s=np.zeros(2)) for g in ['full_visible','partial','blocked']}
tail_num=np.zeros(2,np.int64);tail_s=np.zeros(2);tail_blocked=np.zeros(2,np.int64);viol_num=np.zeros(2,np.int64)
example=[];ratios=[[],[]];top=[]
for k,i in enumerate(ids):
 y=np.stack([z[i] for z in yarr]).astype('float64');v=np.stack([np.unpackbits(z[i],bitorder='big').reshape(256,256) for z in marr]).astype(bool)
 p=np.array(base[bi[i]],dtype='float64');d=x[k,2].astype('float64')[None]*scale[:,None,None]
 cap=np.full_like(d,3000.);nz=d>0;cap[nz]=-10*np.log10(d[nz])
 residual=np.where(v,np.clip(y-cap,0,None),0);err=np.where(v,(p-y)**2,0)
 num+=v.sum((1,2));ss+=(residual**2).sum((1,2));bss+=err.sum((1,2))
 categories={'full_visible':x[k,3]>.999,'blocked':x[k,3]<1e-6}
 categories['partial']=~(categories['full_visible']|categories['blocked'])
 for g,c in categories.items():
  w=v&c[None];bys[g]['n']+=w.sum((1,2));bys[g]['s']+=np.where(w,residual**2,0).sum((1,2));bys[g]['base_s']+=np.where(w,err,0).sum((1,2))
 tail=v&(abs(p-y)>=5);tail_num+=tail.sum((1,2));tail_s+=np.where(tail,err,0).sum((1,2))
 wrong=v&(residual>1);viol_num+=wrong.sum((1,2));tail_blocked+=(wrong&tail).sum((1,2))
 for b in range(2):
  ratios[b].append(residual[b,wrong[b]].astype('float32'))
  # One illustrative candidate per map/band, selected by floor error, not a frequency estimate.
  j=int(np.argmax(residual[b]));rr,cc=divmod(j,256)
  top.append(dict(map=i,terrain=rows[i]['terrain_id'],band=['415','58'][b],row=rr,col=cc,tx_row=int(rows[i]['tx_row']),tx_col=int(rows[i]['tx_col']),
   label_db=float(y[b,rr,cc]),C82_db=float(p[b,rr,cc]),direct_ceiling_db=float(cap[b,rr,cc]),floor_error_db=float(residual[b,rr,cc]),visible_fraction=float(x[k,3,rr,cc])))
r=json.loads((A/'direct_n16_monitor/result.json').read_text())
floor=np.sqrt(ss/num);assert np.allclose(floor,r['floor_rmse_los_then_through'][0],rtol=0,atol=1e-10)
out=dict(status='verified',maps=len(ids),terrains=len(set(rows[i]['terrain_id'] for i in ids)),count=num.tolist(),
 floor_rmse=floor.tolist(),C82=np.sqrt(bss/num).tolist(),floor_sse=ss.tolist(),
 regions={g:dict(count=z['n'].tolist(),floor_rmse_within_region=np.sqrt(z['s']/np.maximum(z['n'],1)).tolist(),
  contribution_to_overall_floor=np.sqrt(z['s']/num).tolist(),fraction_floor_sse=(z['s']/ss).tolist(),C82_rmse=np.sqrt(z['base_s']/np.maximum(z['n'],1)).tolist()) for g,z in bys.items()},
 contradiction_fraction=(viol_num/num).tolist(),tail_contradiction_fraction=(tail_blocked/np.maximum(tail_num,1)).tolist(),
 required_direct_attenuation_db_quantiles=[np.quantile(np.concatenate(a),[.5,.9,.99]).tolist() for a in ratios],
 examples=sorted(top,key=lambda q:q['floor_error_db'],reverse=True)[:20],
 official_VAL_used=False,trained=False)
(A/'floor_verified.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='examples'},ensure_ascii=False,indent=2))
