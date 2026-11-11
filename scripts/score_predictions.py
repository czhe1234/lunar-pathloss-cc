"""Independent NumPy scoring of saved predictions. Never imports torch/training code."""
import argparse,csv,json,time,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data'
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);args=p.parse_args();out=ROOT/args.predictions
mapping=json.loads((out/'indices.json').read_text());split=mapping['source_split'];ii=np.array(mapping['index_rows'],dtype=np.int64)
rows=list(csv.DictReader((DATA/f'{split}_index.csv').open()))
assert len(ii)==len(set(ii.tolist())) and all(rows[i]['sample_id']==sid for i,sid in zip(ii,mapping['sample_ids']))
meta=json.loads((out/'inference.json').read_text());assert len(ii)==meta['maps_per_band']
hm=np.load(DATA/f'{split}_hm.npy',mmap_mode='r',allow_pickle=False)
relief=np.ptp(hm,axis=(1,2));edges=[0,32,64,128,256,float('inf')];re=[0,10,30,60,120,float('inf')]
yy,xx=np.indices((256,256));results={};map_rows={};start=time.time()
for band in ('415','58'):
    pred=np.load(out/f'pl{band}.npy',mmap_mode='r',allow_pickle=False)
    target=np.load(DATA/f'{split}_pl{band}.npy',mmap_mode='r',allow_pickle=False);packed=np.load(DATA/f'{split}_mask{band}.npy',mmap_mode='r',allow_pickle=False)
    assert pred.shape==(len(ii),256,256) and pred.dtype==np.float32
    total=0.;count=0;all_total=0.;ds=np.zeros(5);dc=np.zeros(5,dtype=np.int64);rs=np.zeros(5);rc=np.zeros(5,dtype=np.int64);group={g:[0.,0,0] for g in ('single','multi')}
    for lo in range(0,len(ii),8):
        ids=ii[lo:lo+8];pr=pred[lo:lo+len(ids)].astype(np.float64);truth=target[ids].astype(np.float64)
        assert np.isfinite(pr).all() and np.isfinite(truth).all()
        mask=np.unpackbits(packed[ids],axis=1,bitorder='big').reshape(len(ids),256,256).astype(bool)
        err=np.square(pr-truth);ss=np.where(mask,err,0).sum(axis=(1,2));cc=mask.sum(axis=(1,2));assert (cc>0).all()
        total+=float(ss.sum());count+=int(cc.sum());all_total+=float(err.sum())
        for j,i in enumerate(ids):
            row=rows[i];g=group[row['group']];g[0]+=float(ss[j]);g[1]+=int(cc[j]);g[2]+=1
            mr=map_rows.setdefault(int(i),{'index_row':int(i),'sample_id':row['sample_id'],'terrain_id':row['terrain_id'],'group':row['group'],'relief_m':float(relief[int(row['hm_row'])])})
            mr[f'sse{band}']=float(ss[j]);mr[f'count{band}']=int(cc[j]);mr[f'rmse{band}']=float(np.sqrt(ss[j]/cc[j]))
            d=np.hypot(xx-int(row['tx_col']),yy-int(row['tx_row']))
            for k in range(5):
                select=mask[j] & (d>=edges[k]) & (d<edges[k+1]);ds[k]+=err[j][select].sum();dc[k]+=select.sum()
            k=min(4,int(np.searchsorted(re,mr['relief_m'],side='right')-1));rs[k]+=ss[j];rc[k]+=cc[j]
    def bins(s,c,e):return [{'low':e[k],'high':e[k+1] if np.isfinite(e[k+1]) else None,'valid_pixels':int(c[k]),'rmse_db':float(np.sqrt(s[k]/c[k])) if c[k] else None} for k in range(5)]
    results[band]={'sse':total,'valid_pixels':count,'rmse_db':float(np.sqrt(total/count)),'all_pixel_rmse_diagnostic':float(np.sqrt(all_total/(len(ii)*65536))),'groups':{g:{'maps':a[2],'valid_pixels':a[1],'sse':a[0],'rmse_db':float(np.sqrt(a[0]/a[1])) if a[1] else None} for g,a in group.items()},'distance_m_bins':bins(ds,dc,edges),'terrain_relief_m_bins':bins(rs,rc,re)}
result={'prediction_directory':str(out.relative_to(ROOT)),'source_split':split,'evaluation_split':'official_validation' if split=='val' else 'development' if meta['part']=='dev' else 'training','maps_per_band':len(ii),'calculation':'NumPy float64 error squared and summed over unpacked mask==1, pooled before square root','bands':results,'goal_met_on_official_validation':split=='val' and len(ii)==len(rows) and all(results[b]['rmse_db']<=1 for b in ('415','58')),'seconds':time.time()-start}
result_path=out/'independent_score.json';result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
with (out/'error_by_map.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(next(iter(map_rows.values())).keys()));w.writeheader();w.writerows(map_rows.values())
torch_path=out/'same_prediction_torch_score.json'
if not torch_path.exists():
    with (out/'same_prediction_torch_score.log').open('w') as log:
        subprocess.run([str(ROOT/'run_gpu3.sh'),'scripts/score_saved_torch.py','--predictions',str(out.relative_to(ROOT))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
torch_score=json.loads(torch_path.read_text())
same_diff={b:abs(results[b]['rmse_db']-torch_score['bands'][b]['rmse_fp64_error_fp64_sum']) for b in ('415','58')}
assert all(results[b]['valid_pixels']==torch_score['bands'][b]['valid_pixels'] for b in ('415','58'))
result['same_prediction_numpy_torch_absolute_difference_db']=same_diff
result['same_prediction_scorers_agree_within_1e_minus8_db']=max(same_diff.values())<=1e-8
result['all_verification_checks_passed']=result['same_prediction_scorers_agree_within_1e_minus8_db']
train_result=ROOT/'runs'/meta['run']/'result.json'
if train_result.exists() and not meta['precision'].startswith('FP32'):
    ref=json.loads(train_result.read_text())
    if ref['evaluation_split']==result['evaluation_split'] and ref['maps_per_band']==len(ii):
        diff={b:abs(results[b]['rmse_db']-ref[f'rmse{b}']) for b in ('415','58')}
        result['inference_replay_absolute_difference_db']=diff
        result['inference_replay_agrees_within_1e_minus3_db']=max(diff.values())<=1e-3
        result['all_verification_checks_passed'] &= result['inference_replay_agrees_within_1e_minus3_db']
        result['verification_scope']='Scorers compare the SAME saved predictions at 1e-8 dB. Separately rerun GPU inference may differ numerically and is checked at 1e-3 dB. This is not a change to the 1 dB performance target.'
result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
assert result['all_verification_checks_passed'],result
print(json.dumps(result,allow_nan=False),flush=True)
