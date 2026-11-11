import argparse,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from src.c74_inference import Predictor
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
d=np.load(a.input,allow_pickle=False);net=Predictor(root)
np.save(a.output,net.predict(d['heightmap'],int(d['tx_row']),int(d['tx_col'])),allow_pickle=False)
