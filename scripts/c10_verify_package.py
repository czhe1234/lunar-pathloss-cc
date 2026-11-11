"""Run the shipped CLI and compare its output with frozen official predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,hashlib,subprocess,numpy as np
R=Path(__file__).resolve().parents[1];P=R/'releases/C10_20261101'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
delivery=read(R/'audit/c10_final_delivery.json');manifest=read(P/'manifest.json')
for name,spec in manifest.items():
 p=P/name
 assert p.is_file() and sha(p)==spec['sha256'] and p.stat().st_size==spec['bytes']
assert sha(Path(delivery['archive']))==delivery['archive_sha256']
row=next(csv.DictReader((R.parent/'data/val_index.csv').open()))
env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
cmd=[str(R/'run_gpu3.sh'),str(P/'scripts/c10_predict_one.py'),'--model-directory','models','--heightmap',str(R.parent/'data/val_hm.npy'),'--hm-row',row['hm_row'],'--tx-row',row['tx_row'],'--tx-col',row['tx_col'],'--output','qa_prediction.npy']
run=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True,check=True)
output=P/'qa_prediction.npy';pr=np.load(output,allow_pickle=False)
assert pr.dtype==np.float32 and pr.shape==(2,256,256) and np.isfinite(pr).all()
pred=R/'runs/c10_full_refit_s20261101/predictions/val_full_gpu_c10_fast_fresh';mapping=read(pred/'indices.json');assert mapping['index_rows'][0]==0
bands={}
for k,b in enumerate(['415','58']):
 ref=np.load(pred/f'pl{b}.npy',mmap_mode='r')[0]
 assert np.array_equal(ref,pr[k])
 bands[b]={'array_equal':True,'max_absolute_difference_db':float(abs(ref-pr[k]).max())}
saved=R/'audit/c10_package_prediction_sample0.npy';assert not saved.exists();output.rename(saved)
result={'passed':True,'manifest_files_verified':len(manifest),'archive_sha256':sha(Path(delivery['archive'])),'CLI_executed':cmd,'CLI_stdout':run.stdout,'CLI_stderr':run.stderr,'sample_id':row['sample_id'],'bands':bands,'output_retained':str(saved),'no_target_or_mask_input':True}
(R/'audit/c10_package_verification.json').write_text(json.dumps(result,indent=2)+'\n')
delivery['checks']['release_package_output_matches_official_prediction']=True;delivery['package_verification']=str(R/'audit/c10_package_verification.json')
(R/'audit/c10_final_delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
print(json.dumps(result),flush=True)
