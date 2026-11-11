from pathlib import Path
import json,time,hashlib,shutil,subprocess
R=Path.cwd();audit=R/'audit/c65_full_refit_20261102'
jobs=json.loads((audit/'train_launch.json').read_text())
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
while any(alive(j['pid']) for j in jobs):time.sleep(3)
for j in jobs:
 result=json.loads((R/'runs'/j['name']/'result.json').read_text());assert result['status']=='complete' and result['steps']==17770
bundle=R/'runs/c65_combo_full';bundle.mkdir(exist_ok=False);roles={}
for role,source in [('low','runs/c65_low_full/best.pt'),('high','runs/c65_high_full/best.pt'),('teacher','runs/c38_full_refit/best.pt')]:
 target=bundle/(role+'.pt');shutil.copy2(R/source,target);roles[role]={'source':source,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
manifest={'model':'C65','high_architecture':'c49','routing':'415MHz=lowhead0;5.775GHz=highhead1;fixed for all pixels','weights':roles,'allTRAIN_terrains':6120,'allTRAIN_maps':25720,'official_VAL_used_for_training_or_selection':False,'steps_each_band':17770,'precision':'compiledBF16; teacher4096; correction4096','parent_internalDEV_score':'audit/c54_band_experts_20261102/c54_combo_dev_compiled_all_score.json'}
(bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(bundle/'config.json').write_text(json.dumps(manifest,indent=2)+'\n')
qa=[]
for card,name in [(3,'H100'),(4,'A800')]:
 log=open('logs/c65_interface_'+name+'.log','w');p=subprocess.Popen(['./run_gpu.sh',str(card),'src/c65_interface_check.py'],stdout=log,stderr=subprocess.STDOUT);qa.append(p)
for p in qa:assert p.wait()==0
for name in ['H100','A800']:
 d=json.loads((audit/('interface_'+name+'.json')).read_text());assert d['complete'] and d['rawH_TX_interface_verified']
pred=[]
for card,shard in [(1,0),(3,1),(4,2)]:
 log=open('logs/c65_VAL_shard'+str(shard)+'.log','w');p=subprocess.Popen(['./run_gpu.sh',str(card),'src/c65_predict.py','--shard',str(shard),'--of','3'],stdout=log,stderr=subprocess.STDOUT);pred.append({'card':card,'shard':shard,'pid':p.pid})
(audit/'VAL_prediction_launch.json').write_text(json.dumps(pred,indent=2)+'\n')
while any(alive(p['pid']) for p in pred):time.sleep(3)
for shard in range(3):
 path=bundle/'predictions'/('val_full_c34_fresh_shard'+str(shard)+'of3')/'inference.json'
 assert json.loads(path.read_text())['complete']
for name,cmd in [('independent_score',['.venv/bin/python','scripts/c34_score.py','--run','c65_combo_full','--split','val','--shards','3']),('official_score',['.venv/bin/python','scripts/c34_official.py','--run','c65_combo_full','--shards','3'])]:
 with open('logs/c65_'+name+'.log','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
(audit/'evaluation_complete.json').write_text(json.dumps({'complete':True,'frozen_bundle_manifest':str(bundle/'manifest.json'),'official_result':'audit/c65_combo_full_official/verified_summary.json','website_submission':False})+'\n')
