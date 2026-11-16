from pathlib import Path
import time,subprocess,json
p=Path('/proc/3828220/stat')
while p.exists() and p.read_text().split()[2]!='Z':time.sleep(2)
assert json.loads(Path('audit/c65_full_refit_20261102/evaluation_complete.json').read_text())['complete']
for name in ['c65_deliver.py','c65_report.py']:
 with open('logs/'+name.replace('.py','')+'.log','w') as f:subprocess.run(['.venv/bin/python','scripts/'+name],stdout=f,stderr=subprocess.STDOUT,check=True)
Path('audit/c65_full_refit_20261102/delivery_controller_complete.json').write_text(json.dumps({'complete':True,'website_submission':False}))
