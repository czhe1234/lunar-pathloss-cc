import sys,os,json,csv,hashlib,tarfile,fcntl,importlib.util,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,gpu_guard,check_source_unchanged,atomic_json
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);info=gpu_guard();check_source_unchanged()
out=ROOT/'deliveries'/args.run;sys.path.insert(0,str(out));spec=importlib.util.spec_from_file_location('delivered_inference',out/'inference.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);predictor=mod.Predictor(out)
rows=list(csv.DictReader((DATA/'val_index.csv').open()));hm=np.load(DATA/'val_hm.npy',mmap_mode='r');rr=rows[:8];h=np.array(hm[[int(a['hm_row']) for a in rr]],copy=True);rc=np.array([[int(a['tx_row']),int(a['tx_col'])] for a in rr]);pred=predictor.predict_many(h,rc);assert pred.shape==(8,2,256,256) and pred.dtype==np.float32 and np.isfinite(pred).all()
reference=np.stack([np.load(ROOT/'runs'/args.run/f'predictions/val_full_bf16/pl{b}.npy',mmap_mode='r')[:8] for b in ['415','58']],1);delta=pred.astype(float)-reference.astype(float);rms=float(np.sqrt(np.mean(delta**2)));maximum=float(np.max(np.abs(delta)));assert rms<.05,(rms,maximum)
original=json.loads((out/'manifest.json').read_text());assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==sha for name,sha in original['files'].items())
result={'passed':True,'device':info,'samples':8,'inputs':'Only original validation heightmaps and TX coordinates; no labels or masks supplied to predictor','saved_prediction_replay_rms_difference_db':rms,'saved_prediction_replay_max_abs_difference_db':maximum,'tolerance_rms_db':.05,'scope':'Checks exported input-only inference against saved validated predictions, allowing GPU numerical replay variation. Full official metric comes from independent full-set scoring.','original_data_size_and_mtime_unchanged':True,'all_original_package_file_hashes_verified':True}
atomic_json(out/'package_smoke_check.json',result)
shutil_path=ROOT/'requirements.lock.txt'
if shutil_path.exists():(out/'requirements.lock.txt').write_bytes(shutil_path.read_bytes())
manifest=original
for name in ['package_smoke_check.json','requirements.lock.txt']:
 if (out/name).exists():manifest['files'][name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
manifest['package_smoke_check_passed']=True;atomic_json(out/'manifest.json',manifest)
archive=out.with_suffix('.tar.gz');temp=archive.with_name(archive.name+'.tmp')
with tarfile.open(temp,'w:gz') as tar:tar.add(out,arcname=out.name)
temp.replace(archive)
with tarfile.open(archive,'r:gz') as tar:
 for name,sha in manifest['files'].items():assert hashlib.sha256(tar.extractfile(out.name+'/'+name).read()).hexdigest()==sha
result['archive_verified_against_manifest']=True;result['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();atomic_json(ROOT/'audit'/f'{args.run}_delivery_verification.json',result);print(json.dumps(result),flush=True)
