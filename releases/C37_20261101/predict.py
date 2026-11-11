"""Predict two pathloss maps from one heightmap and integer TX pixel."""
import argparse,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from c34_inference import Predictor
p=argparse.ArgumentParser();p.add_argument('--heightmap',required=True);p.add_argument('--tx-row',type=int,required=True);p.add_argument('--tx-col',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
pr=Predictor(R/'model').predict(np.load(a.heightmap,allow_pickle=False),a.tx_row,a.tx_col)
np.save(a.output,pr,allow_pickle=False)
