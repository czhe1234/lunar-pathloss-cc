from pathlib import Path
import csv,json,hashlib,datetime,subprocess,os
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';O=R/'audit/c4_remaining_error_20261031';checks={}
def check(name,condition):
 checks[name]=bool(condition)
 if not condition:raise AssertionError(name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads((R/'audit/raw_audit.json').read_text());selection=json.loads((O/'selection.json').read_text());S=json.loads((O/'statistics_summary.json').read_text());J=json.loads((O/'joint_band_audit.json').read_text());ST=json.loads((O/'structure_audit.json').read_text());package=R/'deliveries/c4_full_refit_s20261031';mf=json.loads((package/'manifest.json').read_text())
for name,a in source['files'].items():
 p=D/name;stat=p.stat();check('unchanged_raw_stat_'+name,stat.st_size==a['size_bytes'] and stat.st_mtime_ns==a['mtime_ns'])
 if p.suffix in ['.json','.md'] or name in ['train_index.csv','val_index.csv']:check('unchanged_raw_hash_'+name,sha(p)==a['sha256'])
check('checkpoint_sha_unchanged',sha(R/'runs/c4_full_refit_s20261031/best.pt')==selection['frozen_checkpoint_sha256'])
for name in ['weights.pt','inference.py','c3_geometry.py','c4_geometry.py','c4_model.py']:check('frozen_export_'+name,sha(package/name)==mf['files'][name])
rows={s:list(csv.DictReader((D/f'{s}_index.csv').open())) for s in ['train','val']};check('train_val_terrain_ids_disjoint',not ({r['terrain_id'] for r in rows['train']} & {r['terrain_id'] for r in rows['val']}));check('matched_train_excludes_old_development',not ({rows['train'][i]['terrain_id'] for i in selection['train_indices']} & set(selection['original_development_terrain_ids'])))
check('official_all_indices_order',selection['val_indices']==list(range(2330)))
for split,label in [('train','matched_train'),('val','official_val')]:
 a=S[label];b=ST[label];ids=selection[split+'_indices'];mapping=json.loads((O/label/'indices.json').read_text());check(label+'_ids',mapping['index_rows']==ids and mapping['sample_ids']==[rows[split][i]['sample_id'] for i in ids]);check(label+'_map_terrain_counts',len(ids)==2330 and len(set(ids))==2330 and len({rows[split][i]['terrain_id'] for i in ids})==680)
 ss=np.array(a['zones']['all']['sse']);nn=np.array(a['zones']['all']['n']);check(label+'_main_rmse',np.allclose(np.sqrt(ss/nn),a['rmse'],rtol=0,atol=1e-12));check(label+'_six_zone_sse',np.allclose(np.sum([x['sse'] for x in b['subzones'].values()],0),ss,atol=1e-5,rtol=1e-12));check(label+'_six_zone_counts',np.array_equal(np.sum([x['n'] for x in b['subzones'].values()],0),nn));check(label+'_class_sse',np.allclose(np.array(a['target_and_predicted_excess_classes']['sse']).sum((1,2)),ss,rtol=1e-12));check(label+'_class_counts',np.array_equal(np.array(a['target_and_predicted_excess_classes']['counts']).sum((1,2)),nn));check(label+'_paired_receivers',b['same_terrain_nearest_tx']['receiver_points_per_map']==1024)
 n_by_group={g:sum(rows[split][i]['group']==g for i in ids) for g in ['single','multi']};check(label+'_matched_groups',n_by_group=={'single':647,'multi':1683})
check('official_score_matches_delivery',np.allclose(S['official_val']['rmse'],[3.812677339620629,4.990174945470119],rtol=0,atol=1e-12));jr=J['results'];check('joint_band_counts',sum(v['joint_valid_pixels'] for k,v in jr.items() if k!='all')==jr['all']['joint_valid_pixels']);check('joint_extra_sse_closes',np.isclose(sum(v['high_minus_low_sse'] for k,v in jr.items() if k!='all'),jr['all']['high_minus_low_sse'],rtol=1e-12));check('joint_common_matches_other_calculation',abs(jr['all']['common_error_energy_fraction']-S['official_val']['joint_band_common_error_sse_fraction'])<1e-12)
repeat=json.loads((O/'identical_input_recheck.json').read_text());check('nine_duplicates',repeat['train']['duplicate_input_groups']==9);check('duplicate_masks_identical',all(v['masks_identical'] for x in repeat['train']['pairs'] for v in x['bands'].values()));check('no_val_replicate_score',repeat['val']['pooled_pair_rmse']==[None,None])
precision=json.loads((O/'precision_audit.json').read_text());check('strict_precision',precision['tf32_enabled']==False and precision['autocast_enabled']==False and precision['model_updated']==False);check('gpu_uuid_precision',precision['device']['uuid']=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0')
for ext in ['pdf','png','svg','eps','jpg']:
 name='c4_remaining_error_preview.jpg' if ext=='jpg' else f'c4_remaining_error.{ext}';check('figure_'+ext,(O/'figures'/name).stat().st_size>1000)
# Read status only; never stop unrelated processes.
ps=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True);active=[line.strip() for line in ps.splitlines() if ('scripts/train' in line or 'c4_remaining_frozen_probe.py' in line or 'c4_strict_precision_audit.py' in line) and ('python' in line.split(None,1)[-1].split()[0] if line.split(None,1) else False)]
gpu=subprocess.check_output(['nvidia-smi','-i','GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();gpu_apps=subprocess.check_output(['nvidia-smi','-i','GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
files={}
for p in O.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.md','.npz','.png','.pdf','.svg','.eps','.jpg'] and p.name!='verification.json':files[str(p.relative_to(R))]=sha(p)
for name in ['c4_remaining_frozen_probe.py','c4_strict_precision_audit.py','c4_remaining_statistics.py','c4_remaining_structure_audit.py','c4_joint_band_audit.py','c4_phase_diagnose.py','c4_remaining_figure.py','c4_remaining_report.py','c4_remaining_verify.py']:
 p=R/'scripts'/name;files[str(p.relative_to(R))]=sha(p)
p=R/'results/C4剩余大误差归因_20261101.md';files[str(p.relative_to(R))]=sha(p)
result={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks_passed':len(checks),'checks':checks,'checkpoint_sha256':selection['frozen_checkpoint_sha256'],'gpu_snapshot_uuid_memoryMiB_utilizationPercent':gpu,'gpu_compute_processes':gpu_apps,'matching_training_or_probe_processes':active,'no_parameter_updates':True,'official_validation_usage':'Post-hoc diagnostic only; no checkpoint selection or change. Record this exposure for future research.','operational_note':'CPU joint-band accumulator was restarted after vectorizing scalar sums; only the completed vectorized output is used. No neural training was run.','files_sha256':files}
(O/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checks','files_sha256']},ensure_ascii=False))
