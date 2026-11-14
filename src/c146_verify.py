"""Independent NumPy FP64 scoring. C82 appears only here, after training/export."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/"data"
AUDIT=ROOT/"audit/c146_solver_ablation";RUN=ROOT/"runs/c146_solver_ablation"
ARMS=("with_solver","without_solver")
GROUPS=("all","tail","under_tail","over_tail","non_tail","C82_good")
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def atomic(path,obj):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n");temp.replace(path)
def agg(sse,count):
    return np.sqrt(np.divide(sse,count,out=np.zeros_like(sse),where=count>0))
def main():
    plan=json.loads((AUDIT/"protocol.json").read_text())
    frozen=json.loads((RUN/"frozen_checkpoints.json").read_text())
    assert (RUN/"export_complete.json").exists()
    for key,info in frozen.items():assert sha(RUN/f"{key}.pt")==info["sha256"]
    rows=list(csv.DictReader((DATA/"train_index.csv").open()))
    labels=[np.load(DATA/f"train_pl{b}.npy",mmap_mode="r") for b in ("415","58")]
    masks=[np.load(DATA/f"train_mask{b}.npy",mmap_mode="r") for b in ("415","58")]
    base_arrays=[np.load(ROOT/"audit/c88_mesh/base.npy",mmap_mode="r"),
                 np.load(ROOT/"audit/c101_residual/base0of1/prediction.npy",mmap_mode="r")]
    base_ids=[json.loads((ROOT/"audit/c88_mesh/base_index.json").read_text())["ids"],
              json.loads((ROOT/"audit/c101_residual/base0of1/indices.json").read_text())]
    lookup={}
    for source,ids in enumerate(base_ids):
        for j,i in enumerate(ids):
            assert int(i) not in lookup
            lookup[int(i)]=(source,j)
    summary={"status":"verified","protocol_sha256":sha(AUDIT/"protocol.json"),"panels":{},
             "delta_definition":"RMSE_without - RMSE_with; positive means solver features help",
             "C82_used_only_after_freeze":True,"official_VAL_used":False}
    for name,ids in plan["panels"].items():
        terrains=sorted(set(rows[i]["terrain_id"] for i in ids));ti={v:j for j,v in enumerate(terrains)}
        keys=[f"{a}_{tag}" for tag in ("selected","final") for a in ARMS]
        preds={k:np.load(RUN/f"{k}_{name}.npy",mmap_mode="r") for k in keys}
        for k,p in preds.items():
            receipt=json.loads((RUN/f"{k}_{name}.json").read_text())
            assert receipt["checkpoint_sha256"]==frozen[k]["sha256"]
            assert receipt["prediction_sha256"]==sha(RUN/f"{k}_{name}.npy")
            assert p.dtype==np.float32 and p.shape==(len(ids),2,256,256)
        # Last model index is the frozen C82 reference.
        ts=np.zeros((len(terrains),5,len(GROUPS),2),np.float64)
        tc=np.zeros((len(terrains),len(GROUPS),2),np.int64)
        new=np.zeros((4,2,2),np.int64);own_tail=np.zeros((4,2),np.int64)
        own_tail_sse=np.zeros((4,2),np.float64);correct=np.zeros((4,2),np.int64)
        for j,i in enumerate(ids):
            y=np.stack([a[i] for a in labels]).astype(np.float64)
            valid=np.stack([np.unpackbits(a[i]).reshape(256,256) for a in masks]).astype(bool)
            s,b=lookup[i];base=base_arrays[s][b].astype(np.float64);residual=y-base
            tail=np.abs(residual)>=5
            region=(valid,valid&tail,valid&(residual>=5),valid&(residual<=-5),valid&~tail,valid&(np.abs(residual)<=1))
            t=ti[rows[i]["terrain_id"]]
            for g,m in enumerate(region):
                tc[t,g]+=m.sum((1,2));ts[t,4,g]+=np.where(m,residual**2,0).sum((1,2))
            for k,key in enumerate(keys):
                p=preds[key][j].astype(np.float64);assert np.isfinite(p).all()
                e=p-y;sq=e**2
                for g,m in enumerate(region):ts[t,k,g]+=np.where(m,sq,0).sum((1,2))
                for z,threshold in enumerate((3,5)):new[k,z]+=(region[-1]&(np.abs(e)>threshold)).sum((1,2))
                big=valid&(np.abs(e)>=5);own_tail[k]+=big.sum((1,2));own_tail_sse[k]+=np.where(big,sq,0).sum((1,2))
                correct[k]+=(region[1]&((p-base)*residual>0)).sum((1,2))
            if (j+1)%200==0:print("CPU_score",name,j+1,len(ids),flush=True)
        ss=ts.sum(0);cc=tc.sum(0);rmse=agg(ss,cc[None])
        result={"maps":len(ids),"terrains":len(terrains),"groups":list(GROUPS),"count":cc.tolist(),
                "C82":rmse[4].tolist(),"models":{},"paired":{}}
        for k,key in enumerate(keys):
            result["models"][key]={"epoch":frozen[key]["epoch"],"RMSE":{g:rmse[k,j].tolist() for j,g in enumerate(GROUPS)},
                "SSE":ss[k].tolist(),"own_error_ge5_fraction":(own_tail[k]/cc[0]).tolist(),
                "own_error_ge5_SSE_share":(own_tail_sse[k]/ss[k,0]).tolist(),
                "new_error_on_C82good_gt3_gt5_count":new[k].tolist(),
                "tail_correction_direction_accuracy":(correct[k]/np.maximum(cc[1],1)).tolist()}
        rng=np.random.default_rng(146091);bootstrap=[]
        for _ in range(1000):
            ix=rng.integers(0,len(terrains),len(terrains))
            rs=agg(ts[ix].sum(0),tc[ix].sum(0)[None])
            bootstrap.append(np.stack((rs[1]-rs[0],rs[3]-rs[2])))
        ci=np.quantile(np.stack(bootstrap),[.025,.975],axis=0)
        for j,tag in enumerate(("selected","final")):
            delta=rmse[2*j+1]-rmse[2*j]
            result["paired"][tag]={"delta_without_minus_with":{g:delta[z].tolist() for z,g in enumerate(GROUPS)},
                                  "CI95":{g:ci[:,j,z].T.tolist() for z,g in enumerate(GROUPS)}}
        np.savez_compressed(RUN/f"{name}_terrain_statistics.npz",terrains=np.array(terrains),model_keys=np.array(keys+["C82"]),groups=np.array(GROUPS),sse=ts,count=tc)
        atomic(RUN/f"{name}_verified.json",result);summary["panels"][name]=result
        del preds
    # Independent CPU all-pixel scores must match the last/selected monitor GPU metric.
    hist=json.loads((RUN/"history.json").read_text());byepoch={x["epoch"]:x for x in hist}
    initial=json.loads((RUN/"initial.json").read_text())["both_arms"]
    diffs=[]
    for arm in ARMS:
        for tag in ("selected","final"):
            key=f"{arm}_{tag}";ep=frozen[key]["epoch"]
            expected=initial if ep==0 else byepoch[ep]["monitor"][arm]
            actual=summary["panels"]["monitor"]["models"][key]["RMSE"]["all"]
            diffs.extend(abs(np.array(expected["rmse"])-actual).tolist())
    assert max(diffs)<2e-5, diffs
    summary["max_CPU_GPU_RMSE_difference"]=max(diffs)
    primary=summary["panels"]["confirmation"]["paired"]["selected"]
    delta=np.array(primary["delta_without_minus_with"]["all"]);interval=np.array(primary["CI95"]["all"])
    if np.all(delta>=.05) and np.all(interval[:,0]>0):verdict="Solver features help in this paired run."
    elif np.all(delta<=-.05) and np.all(interval[:,1]<0):verdict="Solver features harm in this paired run."
    else:verdict="Mixed or below-materiality result; no uniform net-effect conclusion."
    summary["verdict"]=verdict
    atomic(AUDIT/"comparison_summary.json",summary)
    lines=["# C146：快速求解器特征有／无对照","",verdict,"",
           "两组同一网络、同一随机初始化、同一数据顺序和128轮预算；无C82输入、旧权重或C82残差目标。保留基础地形/TX与共同自由空间距离项。无求解器组45个派生通道置零，模型形状完全相同。",
           "","数字均为全部原生有效像素的CPU FP64 RMSE，顺序415MHz／5.775GHz。新地形确认集为历史484地形1934图，不是全新盲测或官方VAL。","",
           "| 面板／权重 | 使用求解器：全图 | 不用求解器：全图 | 使用求解器：固定大误差区 | 不用求解器：固定大误差区 |",
           "|---|---|---|---|---|"]
    pair=lambda x:" / ".join(f"{v:.6f}" for v in x)
    for name,r in summary["panels"].items():
        for tag in ("selected","final"):
            a=r["models"][f"with_solver_{tag}"]["RMSE"];b=r["models"][f"without_solver_{tag}"]["RMSE"]
            lines.append(f"| {name}/{tag} | {pair(a['all'])} | {pair(b['all'])} | {pair(a['tail'])} | {pair(b['tail'])} |")
    lines+=["","固定大误差区是冻结C82误差≥5dB的位置。C82只在权重和预测冻结后用于评价分区，不参与模型选择或学习。每个模型自身大误差比例、原准确区新增错误和分地形配对区间保存在comparison_summary.json。",
            "",f"CPU/GPU同点RMSE最大差：{max(diffs):.3g} dB。一个配对随机种子；只能判断此架构、特征集合、训练预算下的作用，不能证明所有求解器特征都有害或原始数据不可预测。",
            "","训练轨迹：runs/c146_solver_ablation/history.json；独立统计与预测同目录；协议：audit/c146_solver_ablation/protocol.json。"]
    (ROOT/"results/C146_快速求解器特征有无对照_20261113.md").write_text("\n".join(lines)+"\n")
    atomic(AUDIT/"decision.json",{"status":"complete","verdict":verdict,"next_action":"Use paired all/tail/non-tail effects to decide retain/remove/refine solver channels; no automatic official model replacement or fullTRAIN refit."})
    print("VERIFIED",verdict,"CPU_GPU_maxdiff",max(diffs),flush=True)
if __name__=="__main__":main()
