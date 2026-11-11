"""Matched C23 readout-conditioning check on the fixed128 FIT maps."""
from cc_core import ROOT,atomic_json,check_source_unchanged
import json,time,hashlib,shutil,math
import numpy as np
import torch
from c21_train_r1 import setup,gather,digest_state
from c21_pair_r1 import loader,gpu_batch
from c22_train import monitor
from c23_model import Model
A=ROOT/"audit/c23_conditioning_20261101"
def main():
    protocol=json.loads((A/"protocol.json").read_text())
    gpu=setup(protocol["seed"]);check_source_unchanged()
    parent=ROOT/protocol["small_parent"]
    assert parent.exists()
    run=ROOT/"runs/c23_fit128_pair";run.mkdir(exist_ok=False);(run/"source").mkdir()
    cfg=dict(protocol,gpu=gpu,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
    for name in ["c23_small.py","c23_model.py","c22_train.py","cc_core.py","c21_data.py","c21_model.py","c21_model_r1.py","c21_pair_r1.py","c21_train_r1.py"]:
        shutil.copy2(ROOT/"src"/name,run/"source"/name)
    atomic_json(run/"config.json",cfg)
    ck=torch.load(parent,map_location="cpu",weights_only=False)
    n=np.load(ROOT/"audit/c21_execution_20261101/normalization.npz")
    nets=[];opts=[];init=[]
    for cond in [False,True]:
        torch.manual_seed(protocol["seed"])
        net=Model(torch.from_numpy(n["mean"]),torch.from_numpy(n["std"]),conditional=cond).cuda()
        missing,extra=net.load_state_dict(ck["model"],strict=False)
        assert not extra and all(k.startswith("condition.") for k in missing)
        nets.append(net);opts.append(torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.0002,weight_decay=1e-5));init.append(digest_state(net))
    del ck
    assert init[0]==init[1]
    atomic_json(run/"initialization.json",{"model_sha256":init,"equal":True,"optimizer":"fresh AdamW for both; parent optimizer not reused because new parameters are added"})
    ids=np.array(protocol["fit_panel"]);steps=protocol["small_steps"]
    rng=np.random.default_rng(protocol["seed"])
    order=np.concatenate([rng.permutation(ids) for _ in range(math.ceil(steps*8/len(ids)))])[:steps*8]
    np.save(run/"map_order.npy",order)
    pixgen=torch.Generator().manual_seed(protocol["seed"]+1)
    pixels=np.load(ROOT/"runs/c21r1_pair_s20261123/monitor_pixels.npy")
    t0=time.time();log=(run/"history.jsonl").open("a",buffering=1)
    def emit(row):
        row["seconds"]=time.time()-t0;print(json.dumps(row),flush=True)
        log.write(json.dumps(row)+"\n");atomic_json(run/"progress.json",row)
    def evaluate(step):
        fit=[monitor(net,protocol["fit_panel"],pixels) for net in nets]
        row={"event":"monitor","step":step,"FIT":fit}
        emit(row);atomic_json(run/"latest_eval.json",row)
        for label,net,opt in zip(["constant","conditional"],nets,opts):
            torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),"step":step,"metrics":row,"config":cfg},run/f"last_{label}.pt")
        return row
    start=evaluate(0)
    for step,batch in enumerate(loader(order,8,6),1):
        x,tx,fs,y,m=gpu_batch(batch);pix=torch.randint(65536,(len(x),2048),generator=pixgen).cuda()
        yy=gather(y,pix);mm=gather(m,pix);base=gather(fs,pix)
        losses=[];norms=[]
        lr=.0002*min(step/50.,1)*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        for net,opt in zip(nets,opts):
            for pg in opt.param_groups:pg["lr"]=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):pred=net(x,tx,pix,512)*20+base
            loss=(((pred-yy).square()*mm).sum((0,1))/mm.sum((0,1)).clamp_min(1)).mean()/400
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
            if not torch.isfinite(norm):raise RuntimeError("Nonfinite gradient")
            opt.step();losses.append(float(torch.sqrt(loss*400)));norms.append(float(norm))
        if step<=2 or step%100==0:
            emit({"event":"train","step":step,"sampled_rmse_constant_conditional":losses,"grad_norms":norms,"lr":lr,"peak_GB":torch.cuda.max_memory_allocated()/1e9})
        if step%500==0 or step==steps:final=evaluate(step)
    delta=[b-a for a,b in zip(final["FIT"][0]["rmse"],final["FIT"][1]["rmse"])]
    passed=all(x<=-.05 for x in delta)
    atomic_json(run/"complete.json",{"status":"paired_fit_check_complete","steps_each":steps,
        "initial":start,"final":final,"conditional_minus_constant_db":delta,
        "full_data_gate_passed":passed,"gate":"both bands at least0.05dB lower FIT RMSE, then confirm full valid FIT pixels",
        "seconds":time.time()-t0,"gpu":gpu})
if __name__=="__main__":main()
