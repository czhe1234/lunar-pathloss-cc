"""Eight preselected TRAIN-monitor terrains: input-only measurement changes vs saved C82 residual.
No fitted coefficient, no monitor training. Fixed unit-gain correction is a diagnostic probe only.
"""
from pathlib import Path
import json,csv
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c131_transport';O=A/'official_direct';D=R.parent/'data'
ids=json.loads((O/'ids_8000000.json').read_text());allids=json.loads((A/'protocol.json').read_text())['monitor'];mi={i:j for j,i in enumerate(allids)}
power=np.load(A/'direct_n16_monitor/power.npy',mmap_mode='r')
base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r');bi={int(i):j for j,i in enumerate(json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids'])}
target=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];mask=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
variants=['official8M_vs_native','official32M_vs_native','official128M_vs_native','official32M_vs_line32M']
stats={k:dict(dot=np.zeros(2),d2=np.zeros(2),r2=np.zeros(2),n=np.zeros(2,np.int64),base_sse=np.zeros(2),probe_sse=np.zeros(2),count=np.zeros(2,np.int64),tail_agree=np.zeros(2),tail_n=np.zeros(2),supported_tail_n=np.zeros(2),all_tail_n=np.zeros(2)) for k in variants}
per=[]
for i in ids:
 y=np.stack([z[i] for z in target]).astype('float64');v=np.stack([np.unpackbits(z[i],bitorder='big').reshape(256,256) for z in mask]).astype(bool)
 p=np.array(base[bi[i]],dtype='float64');r=y-p;full=power[mi[i],3]>.999
 vals={tag:np.load(O/f'power_{i}_{tag}.npz') for tag in ['8000000','32000000','128000000','32000000_line']}
 n=vals['8000000']['native415']
 ds=[(vals[str(size)]['sionna415'],n) for size in [8000000,32000000,128000000]]
 ds.append((vals['32000000']['sionna415'],vals['32000000_line']['sionna415']))
 for name,(new,old) in zip(variants,ds):
  s=stats[name];ok=full&(new>0)&(old>0)
  delta=np.zeros_like(n);delta[ok]=-10*np.log10(new[ok]/old[ok])
  # Frozen safeguard for an explanatory unit-gain probe, not selected on labels.
  delta=np.clip(delta,-20,20)
  w=v&ok[None];tail=v&(abs(r)>=5);validtail=tail&ok[None]
  s['dot']+=np.where(w,r*delta,0).sum((1,2));s['d2']+=np.where(w,delta**2,0).sum((1,2));s['r2']+=np.where(w,r**2,0).sum((1,2));s['n']+=w.sum((1,2))
  s['base_sse']+=np.where(v,r**2,0).sum((1,2));s['probe_sse']+=np.where(v,(r-delta)**2,0).sum((1,2));s['count']+=v.sum((1,2))
  s['tail_agree']+=(validtail&(r*delta>0)).sum((1,2));s['tail_n']+=validtail.sum((1,2))
  s['supported_tail_n']+=validtail.sum((1,2));s['all_tail_n']+=tail.sum((1,2))
  per.append(dict(map=i,variant=name,base_sse=np.where(v,r**2,0).sum((1,2)).tolist(),probe_sse=np.where(v,(r-delta)**2,0).sum((1,2)).tolist(),count=v.sum((1,2)).tolist()))
out={}
for name,s in stats.items():
 b=np.sqrt(s['base_sse']/s['count']);c=np.sqrt(s['probe_sse']/s['count'])
 out[name]=dict(residual_delta_cosine=(s['dot']/np.sqrt(s['r2']*s['d2'])).tolist(),supported_count=s['n'].tolist(),
  C82=b.tolist(),fixed_unit_probe=c.tolist(),gain=(b-c).tolist(),tail_sign_accuracy=(s['tail_agree']/s['tail_n']).tolist(),
  supported_tail_fraction=(s['supported_tail_n']/s['all_tail_n']).tolist(),count=s['count'].tolist())
result=dict(status='complete',maps=8,terrains=8,ids=ids,results=out,per_map=per,trained=False,official_VAL_used=False,
 scope='Small original TRAIN-monitor mechanism probe only; neither 128-terrain promotion nor final deployment score. Correction applied only at H/TX-computed full-visible pixels with nonzero simulated direct power; delta clipped to [-20,20] before examining scores.')
(A/'measurement_alignment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
