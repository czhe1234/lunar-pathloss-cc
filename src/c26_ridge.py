"""Closed-form global/per-map readout controls with disjoint pixel sets."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parents[1];A=R/"audit/c26_representation_20261101"
LAMBDAS=[.0001,.01,1.]
def sufficient(x,y):
    n=len(y);sx=x.sum(0);sy=y.sum()
    return n,sx,sy,x.T@x,x.T@y
def solve(stats,lam):
    n,sx,sy,xx,xy=stats
    mu=sx/n;my=sy/n
    cov=xx-n*np.outer(mu,mu)
    std=np.sqrt(np.maximum(np.diag(cov)/n,0)).clip(1e-4)
    gram=cov/std[:,None]/std[None,:]
    gram.flat[::len(std)+1]+=lam*n
    rhs=(xy-sx*my)/std
    w=cho_solve(cho_factor(gram,lower=True,check_finite=False),rhs,check_finite=False)/std
    return w,my-mu@w
def main(args):
    t0=time.time();base=R/"features"/("c26_"+args.kind)
    assert (base/"complete.json").exists()
    X=np.load(base/"features.npy",mmap_mode="r")
    Y=np.load(base/"targets.npy",mmap_mode="r");FS=np.load(base/"fspl.npy",mmap_mode="r")
    M=np.load(base/"mask.npy",mmap_mode="r");P=np.load(base/"baseline.npy",mmap_mode="r")
    K,_,_,D=X.shape;total=[];permap=np.zeros((3,2));base_ss=np.zeros(2);count=np.zeros(2,np.int64)
    per_map_rows=[]
    for b in [0,1]:
        accumulator=(0,np.zeros(D),0.,np.zeros((D,D)),np.zeros(D))
        for j in range(K):
            mt=M[j,b,:8192].astype(bool);mv=M[j,b,8192:].astype(bool)
            xt=np.asarray(X[j,b,:8192][mt],np.float64)
            yt=np.asarray(Y[j,b,:8192][mt],np.float64)-np.asarray(FS[j,b,:8192][mt],np.float64)
            xv=np.asarray(X[j,b,8192:][mv],np.float64)
            yv=np.asarray(Y[j,b,8192:][mv],np.float64)-np.asarray(FS[j,b,8192:][mv],np.float64)
            assert len(yt)>0 and len(yv)>0
            stats=sufficient(xt,yt);accumulator=tuple(a+c for a,c in zip(accumulator,stats))
            base_ss[b]+=np.square(np.asarray(P[j,b,8192:][mv],np.float64)-np.asarray(Y[j,b,8192:][mv],np.float64)).sum()
            count[b]+=len(yv);row={"band":b,"map_position":j,"train_count":len(yt),"test_count":len(yv),"sse":[]}
            for li,lam in enumerate(LAMBDAS):
                w,c=solve(stats,lam);err=float(np.square(xv@w+c-yv).sum())
                permap[li,b]+=err;row["sse"].append(err)
            per_map_rows.append(row)
            if (j+1)%16==0:print(json.dumps({"kind":args.kind,"band":b,"maps":j+1,"seconds":time.time()-t0}),flush=True)
        total.append(accumulator)
    global_ss=np.zeros((3,2));global_coefs={}
    for b in [0,1]:
        for li,lam in enumerate(LAMBDAS):
            w,c=solve(total[b],lam);global_coefs[f"band{b}_lambda{lam}"]={"weights":w.tolist(),"intercept":float(c)}
            for j in range(K):
                valid=M[j,b,8192:].astype(bool);x=np.asarray(X[j,b,8192:][valid],np.float64)
                target=np.asarray(Y[j,b,8192:][valid],np.float64)-np.asarray(FS[j,b,8192:][valid],np.float64)
                global_ss[li,b]+=np.square(x@w+c-target).sum()
    result={"kind":args.kind,"scope":"128 FIT maps; frozen supervised encoder.8192 pixels/map fit the diagnostic readout and disjoint8192 evaluate it. Pixel holdout is only for the readout, not independent of encoder training.",
      "feature_width":D,"count":count.tolist(),"baseline_rmse":np.sqrt(base_ss/count).tolist(),"lambdas":LAMBDAS,
      "global_readout_rmse":np.sqrt(global_ss/count).tolist(),"per_map_oracle_rmse":np.sqrt(permap/count).tolist(),
      "seconds":time.time()-t0,"not_competition_score":True,"not_raw_input_information_bound":True,
      "interpretation":"Per-map weights use labels from that same map and cannot be inferred for a new terrain by this diagnostic. All regularization settings reported; no held-out terrain claim."}
    (A/(args.kind+"_ridge.json")).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (A/(args.kind+"_global_coefficients.json")).write_text(json.dumps(global_coefs)+"\n")
    (A/(args.kind+"_per_map_rows.json")).write_text(json.dumps(per_map_rows)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--kind",required=True);main(ap.parse_args())
