"""Export frozen hidden features for a readout recoverability diagnostic."""
from cc_core import ROOT,atomic_json,check_source_unchanged
import argparse,json,time,hashlib
import numpy as np
import torch
from c21_train_r1 import setup,model,gather
from c21_data import Data
from c24_model import expand_readout
A=ROOT/"audit/c26_representation_20261101"
def main(args):
    p=json.loads((A/"protocol.json").read_text());item=p["models"][args.kind]
    gpu=setup(p["seed"]);check_source_unchanged()
    run=ROOT/"features"/("c26_"+args.kind);run.mkdir(exist_ok=False)
    net=model(False)
    if item["wide"]:net=expand_readout(net)
    ckpath=ROOT/item["checkpoint"];ck=torch.load(ckpath,map_location="cpu",weights_only=False)
    net.load_state_dict(ck["model"]);del ck;net.eval()
    data=Data();ids=p["ids"];pixels=np.load(A/"pixels.npy");dim=net.readout[0][-1].in_features
    X=np.lib.format.open_memmap(run/"features.npy",mode="w+",dtype=np.float32,shape=(len(ids),2,16384,dim))
    Y=np.empty((len(ids),2,16384),np.float32);FS=np.empty_like(Y);P=np.empty_like(Y);M=np.empty_like(Y,dtype=np.uint8)
    captured=[[],[]]
    def capture(b):
        def hook(module,args):
            captured[b].append(args[0].detach().cpu().numpy())
        return hook
    handles=[net.readout[b][-1].register_forward_pre_hook(capture(b)) for b in [0,1]]
    t0=time.time();maximum=0.
    for j,i in enumerate(ids):
        x,tx,fs,y,m=[torch.from_numpy(v[None]).cuda() for v in data.sample(i)]
        pix=torch.from_numpy(pixels[j:j+1]).cuda()
        captured[0].clear();captured[1].clear()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            pred=net(x,tx,pix,1024)*20+gather(fs,pix)
        for b in [0,1]:
            h=np.concatenate(captured[b],axis=1)[0]
            X[j,b]=h
            w=net.readout[b][-1].weight.detach().cpu().numpy()[0]
            bias=float(net.readout[b][-1].bias.detach().cpu()[0])
            reconstructed=(h[:64].astype(np.float64)@w.astype(np.float64)+bias)*20
            observed=(pred[0,:64,b]-gather(fs,pix)[0,:64,b]).cpu().numpy()
            maximum=max(maximum,float(np.max(np.abs(reconstructed-observed))))
        Y[j]=gather(y,pix)[0].T.cpu().numpy();FS[j]=gather(fs,pix)[0].T.cpu().numpy()
        M[j]=gather(m,pix)[0].T.cpu().numpy().astype(np.uint8);P[j]=pred[0].T.cpu().numpy()
        if (j+1)%16==0:print(json.dumps({"kind":args.kind,"maps":j+1,"seconds":time.time()-t0}),flush=True)
    X.flush()
    for name,arr in [("targets",Y),("fspl",FS),("mask",M),("baseline",P)]:np.save(run/(name+".npy"),arr)
    for h in handles:h.remove()
    assert maximum<.02,maximum
    atomic_json(run/"complete.json",{"gpu":gpu,"maps":len(ids),"width":dim,"checkpoint_sha256":hashlib.sha256(ckpath.read_bytes()).hexdigest(),"max_linear_reconstruction_difference_db":maximum,"seconds":time.time()-t0,"no_parameter_updates":True})
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--kind",required=True);main(ap.parse_args())
