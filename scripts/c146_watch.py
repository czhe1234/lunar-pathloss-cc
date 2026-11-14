"""Read-only live terminal view for the existing C146 paired training job."""
from pathlib import Path
import argparse,json,os,shutil,sys,time
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"audit/c146_solver_ablation";R=ROOT/"runs/c146_solver_ablation"
def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default
def duration(seconds):
    if seconds is None:return "估算中"
    seconds=max(0,int(seconds));h,rem=divmod(seconds,3600);m,s=divmod(rem,60)
    return f"{h:02d}:{m:02d}:{s:02d}"
def render():
    plan=read(A/"protocol.json",{});p=read(R/"progress.json",{});c=read(A/"controller_status.json",{});history=read(R/"history.json",[])
    total=plan.get("updates_per_arm",45056);epochs=plan.get("epochs",128)
    done=min(total,p.get("updates_per_arm",0));epoch=p.get("epoch",0);batch=p.get("batch",0);batches=p.get("batches",352)
    complete=(A/"comparison_summary.json").exists();exported=(R/"export_complete.json").exists();trained=(R/"training_complete.json").exists()
    last=history[-1]["epoch"] if history else 0
    if complete:stage="训练与独立评价已完成"
    elif exported:stage="训练已完成；正在独立核对分数"
    elif trained:stage="训练已完成；正在导出预测"
    elif c.get("stage") in ("failed","verification_failed"):stage="运行异常，请检查日志"
    elif batch==batches:stage="正在评价或保存断点"
    else:stage="正在训练"
    child=c.get("child_pid")
    if not complete and child and not Path(f"/proc/{child}").exists() and c.get("stage") not in ("waiting_for_memory","waiting_for_CC_device_lock"):
        stage="子进程已退出，等待状态更新" if c.get("stage") not in ("failed","verification_failed") else stage
    if trained:done=total
    fraction=done/total if total else 0
    width=max(12,min(34,shutil.get_terminal_size((90,24)).columns-49))
    fill=int(width*fraction)
    bar="█"*fill+"░"*(width-fill)
    rates=[]
    window=history[-9:]
    for prev,cur in zip(window,window[1:]):
        dt=cur.get("seconds",0)-prev.get("seconds",0)
        ds=cur.get("updates_per_arm",0)-prev.get("updates_per_arm",0)
        if dt>0 and ds>0:rates.append((ds,dt))
    rate=sum(x for x,_ in rates)/sum(t for _,t in rates) if rates else None
    if not rate and p.get("elapsed_seconds",0)>0 and done>0:rate=done/p["elapsed_seconds"]
    eta=(total-done)/rate if rate else None
    try:age=max(0,time.time()-(R/"progress.json").stat().st_mtime)
    except OSError:age=None
    lines=[
        "C146 快速求解器特征对照  |  transformer / CC",
        "",
        f"使用求解器  [{bar}] {100*fraction:5.1f}%  {done:,}/{total:,}",
        f"不用求解器  [{bar}] {100*fraction:5.1f}%  {done:,}/{total:,}",
        "",
        f"第 {epoch}/{epochs} 轮  |  本轮 {batch}/{batches} 批  |  已保存 {last} 轮",
        f"阶段：{stage}",
        f"预计剩余训练：{duration(eta)}  （共享负载会影响耗时）" if not trained else "训练进度 100%；评价状态见上方。",
        f"状态记录距今：{int(age)} 秒" if age is not None else "等待首条训练状态",
        "",
        "每 2 秒刷新。按 Ctrl+C 关闭查看，训练会继续。",
        "更新时间："+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
    ]
    latest=next((x for x in reversed(history) if "monitor" in x),None)
    if latest:
        m=latest["monitor"]
        lines[9:9]=["",f"新地形评价：第 {latest['epoch']} 轮（RMSE dB，越低越好）",
            "                  415 MHz    5.775 GHz",
            "使用求解器       %8.3f     %8.3f" % tuple(m["with_solver"]["rmse"]),
            "不用求解器       %8.3f     %8.3f" % tuple(m["without_solver"]["rmse"])]
    return "\n".join(lines),complete
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--once",action="store_true");args=ap.parse_args()
    interactive=sys.stdout.isatty() and not args.once
    if interactive:sys.stdout.write("\033[2J\033[?25l")
    try:
        while True:
            text,complete=render()
            if interactive:sys.stdout.write("\033[H"+"\n".join(line+"\033[K" for line in text.splitlines())+"\033[J")
            else:sys.stdout.write(text+"\n")
            sys.stdout.flush()
            if args.once or complete:break
            time.sleep(2)
    except KeyboardInterrupt:pass
    finally:
        if interactive:sys.stdout.write("\033[?25h\n");sys.stdout.flush()
if __name__=="__main__":main()
