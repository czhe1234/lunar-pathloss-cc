"""C22 numerical and optimizer audit; disposable updates, no DEV tuning."""
from cc_core import ROOT, atomic_json, check_source_unchanged
import json,time,math
import numpy as np
import torch
from c21_data import Data
from c21_train_r1 import setup,model,gather
A=ROOT/"audit/c22_scale_diagnosis_20261101"
def main():
    gpu=setup(20261124);check_source_unchanged();t0=time.time()
    protocol=json.loads((A/"protocol.json").read_text());data=Data()
    ids=protocol["fit_panel"][:8]
    dat=[data.sample(i) for i in ids]
    x,tx,fs,y,m=[torch.from_numpy(np.stack([d[j] for d in dat])).cuda() for j in range(5)]
    pix=torch.from_numpy(np.random.default_rng(20261124).integers(0,65536,(8,2048))).cuda()
    yy=gather(y,pix);mm=gather(m,pix);base=gather(fs,pix)
    ck=torch.load(ROOT/protocol["parent"],map_location="cpu",weights_only=False)
    net=model(False);net.load_state_dict(ck["model"])
    def pred(amp=True):
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=amp):
            return net(x,tx,pix,512)*20+base
    def loss(p,b=None):
        v=((p-yy).square()*mm).sum((0,1))/mm.sum((0,1))
        return v.mean()/400 if b is None else v[b]/400
    with torch.no_grad():
        net.train();p_train=pred()
        net.eval();p_eval=pred();p_fp32=pred(False)
        diff_mode=torch.sqrt(((p_train-p_eval).square()*mm).sum((0,1))/mm.sum((0,1))).tolist()
        diff_precision=torch.sqrt(((p_eval-p_fp32).square()*mm).sum((0,1))/mm.sum((0,1))).tolist()
        initial_rmse=torch.sqrt(((p_eval-yy).square()*mm).sum((0,1))/mm.sum((0,1))).tolist()
        z=(x-net.mean)/net.std
        clipped_fraction=(z.abs()>8).float().mean((0,2,3)).tolist()
    net.train()
    gradients=[];branch_stats=[]
    shared=[(n,p) for n,p in net.named_parameters() if p.requires_grad and not n.startswith(("readout.","freq"))]
    for b in [0,1]:
        net.zero_grad(set_to_none=True);loss(pred(),b).backward()
        branch={}
        for n,p in net.named_parameters():
            if p.grad is not None:branch[n.split(".")[0]]=branch.get(n.split(".")[0],0.)+float(p.grad.float().square().sum())
        gradients.append(torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for n,p in shared]))
        branch_stats.append({k:math.sqrt(v) for k,v in branch.items()})
    cosine=float(torch.nn.functional.cosine_similarity(gradients[0],gradients[1],dim=0))
    del gradients
    updates=[]
    for lr in [.000025,.0002]:
        net.load_state_dict(ck["model"])
        opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=lr,weight_decay=1e-5)
        opt.load_state_dict(ck["optimizer"])
        norms=[];trace=[]
        before={n:p.detach().clone() for n,p in net.named_parameters() if p.requires_grad}
        for j in range(10):
            for group in opt.param_groups:group["lr"]=lr
            opt.zero_grad(set_to_none=True);p=pred();l=loss(p);trace.append(float(l*400))
            l.backward();norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(),5)));opt.step()
        with torch.no_grad():after_loss=float(loss(pred())*400)
        delta={}
        for n,p in net.named_parameters():
            if n in before:
                key=n.split(".")[0];delta[key]=delta.get(key,0.)+float((p-before[n]).square().sum())
        updates.append({"lr":lr,"batch_mse_trace":trace,"batch_mse_after10":after_loss,
            "grad_norms":norms,"parameter_delta_l2":{k:math.sqrt(v) for k,v in delta.items()}})
        del opt,before
    result={"gpu":gpu,"scope":"8 fixed FIT maps,2048 uniform pixels; optimizer updates are disposable same-batch diagnostics",
        "ids":ids,"initial_rmse":initial_rmse,"train_eval_prediction_rms_db":diff_mode,
        "bf16_fp32_prediction_rms_db":diff_precision,"input_clipped_fraction_per_channel":clipped_fraction,
        "shared_band_gradient_cosine":cosine,"branch_gradient_norms_per_band":branch_stats,
        "same_batch_update_tests":updates,"seconds":time.time()-t0,
        "limits":"Gradient conflict on one batch is descriptive; neither proves nor rules out optimization/capacity limits."}
    atomic_json(A/"optimizer_probe.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":main()
