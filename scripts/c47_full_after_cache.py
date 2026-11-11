import time,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
while not all((R/f'features/c47_teacher/{part}_meta.json').exists() for part in ['fit','dev']):time.sleep(3)
jobs={}
for card,kind in [(4,'flat'),(1,'conv'),(3,'attention')]:
 name='c47_'+kind+'_full';f=open(R/('logs/'+name+'.log'),'w')
 args=['./run_gpu.sh',str(card),'src/c47_train.py','--scope','full','--kind',kind,'--name',name,'--batch','4096','--steps','12000']
 p=subprocess.Popen(args,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 jobs[name]={'pid':p.pid,'card':card,'args':args,'log':str(f.name)}
(R/'audit/c47_full_information_20261102/full_launch.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(jobs,flush=True)
