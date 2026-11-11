from pathlib import Path
import argparse,sys,time,json
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from src.c60_inference import Predictor
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
data=np.load(a.input,allow_pickle=False);h=np.array(data['heightmap'],np.float32);r=int(data['tx_row']);c=int(data['tx_col'])
net=Predictor(R/'model');t=time.time();pred=net.predict(h,r,c);np.save(a.output,pred,allow_pickle=False)
print(json.dumps({'output':a.output,'shape':list(pred.shape),'dtype':str(pred.dtype),'seconds_including_first_compile':time.time()-t,'input':'heightmap and integerTX only'}))
