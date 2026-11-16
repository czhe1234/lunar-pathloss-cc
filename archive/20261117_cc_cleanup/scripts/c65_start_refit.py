from pathlib import Path
import subprocess,time,json
R=Path.cwd();p=Path('/proc/3815906/stat')
while p.exists() and p.read_text().split()[2]!='Z':time.sleep(2)
meta=json.loads(Path('features/c65_full_teacher/meta.json').read_text());assert meta['complete']
jobs=[]
for card,band,name in [(3,0,'c65_low_full'),(1,1,'c65_high_full')]:
 log=open('logs/'+name+'.log','wb')
 child=subprocess.Popen(['./run_gpu.sh',str(card),'src/c65_full_refit.py','--band',str(band),'--name',name],stdout=log,stderr=subprocess.STDOUT)
 jobs.append({'pid':child.pid,'card':card,'name':name})
Path('audit/c65_full_refit_20261102/train_launch.json').write_text(json.dumps(jobs,indent=2))
