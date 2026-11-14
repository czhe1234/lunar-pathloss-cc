"""Read-only C146 live curves; no training imports, labels, CUDA or new inference."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import argparse,json,time,threading,fcntl,math
from datetime import datetime
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"runs/c146_solver_ablation";AUDIT=ROOT/"audit/c146_solver_ablation"
OUT=ROOT/"results/c146_live"
ARMS=("with_solver","without_solver")
COLORS={"with_solver":"#648477","without_solver":"#B27F69"}
LABELS={"with_solver":"使用求解器","without_solver":"不用求解器"}
BANDS=("415 MHz","5.775 GHz")

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def reference():
    p=OUT/"reference.json"
    if p.exists():return read(p,{})
    old=read(ROOT/"runs/c145_capacity_native/history.json",[])[0]["evaluation"]
    ref={k:{"rmse":old[k]["groups"]["all"]["C82_RMSE"],"count":old[k]["groups"]["all"]["count"]} for k in ("fit","monitor")}
    p.write_text(json.dumps(ref,indent=2)+"\n")
    return ref

def snapshot():
    h=read(RUN/"history.json",[]);p=read(RUN/"progress.json",{});c=read(AUDIT/"controller_status.json",{})
    plan=read(AUDIT/"protocol.json",{});init=read(RUN/"initial.json",{});ref=reference()
    last_monitor=next((x for x in reversed(h) if "monitor" in x),None)
    last_fit=next((x for x in reversed(h) if "fit" in x),None)
    for name,rec in (("monitor",last_monitor),("fit",last_fit)):
        if rec:
            for arm in ARMS:assert rec[name][arm]["count"]==ref[name]["count"],"Reference panel counts differ"
    trained=(RUN/"training_complete.json").exists();exported=(RUN/"export_complete.json").exists();verified=(AUDIT/"comparison_summary.json").exists()
    stage="训练中"
    if p.get("batch")==p.get("batches"):stage="评价或保存中"
    if trained:stage="训练完成，导出预测中"
    if exported:stage="独立核对分数中"
    if verified:stage="训练与评价完成"
    if c.get("stage") in ("failed","verification_failed"):stage="运行异常，请查看日志"
    total=plan.get("updates_per_arm",45056);done=total if trained else p.get("updates_per_arm",0)
    return {"updated":datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "progress":p,"stage":stage,"percent":100*done/total,"done":done,"total":total,
        "completed_epochs":h[-1]["epoch"] if h else 0,"monitor":last_monitor,"fit":last_fit,
        "last_training":h[-1] if h else None,"history":h,"initial":init,"reference":ref,
        "curve_version":str((RUN/"history.json").stat().st_mtime_ns) if (RUN/"history.json").exists() else "0",
        "verified":verified}

def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,ticker
    font="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font)
    plt.style.use(OUT/"academic.mplstyle")
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=font).get_name(),"axes.unicode_minus":False,
                         "font.size":11,"axes.titlesize":12,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10})
    fig,axs=plt.subplots(2,2,figsize=(11.8,7.8),dpi=150)
    h=data["history"];end=max(8,data["completed_epochs"])
    for row,name in enumerate(("monitor","fit")):
        records=[x for x in h if name in x]
        for band,ax in enumerate(axs[row]):
            all_values=[]
            for arm in ARMS:
                xs=[r["epoch"] for r in records];ys=[r[name][arm]["rmse"][band] for r in records]
                if name=="monitor" and data["initial"]:
                    xs=[0]+xs;ys=[data["initial"]["both_arms"]["rmse"][band]]+ys
                if xs:
                    ax.plot(xs,ys,label=LABELS[arm],color=COLORS[arm],marker="o" if arm==ARMS[0] else "s",markersize=5,linewidth=2.2)
                    all_values.extend(ys)
            baseline=data["reference"][name]["rmse"][band]
            ax.axhline(baseline,color="#92979B",linestyle=(0,(4,3)),linewidth=1.25,label="现有 C82 参考")
            title=("新地形评价" if row==0 else "完整训练集评价")+" · "+BANDS[band]
            ax.set_title(title,pad=10);ax.set_ylabel("RMSE（dB，越低越好）")
            ax.set_xlabel("已完成训练轮数");ax.set_xlim(0,end+max(.4,end*.025));ax.set_ylim(0,max(all_values+[baseline])*1.12)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6,integer=True))
            if not records:ax.text(.5,.65,"第 8 轮评价完成后显示" if row else "等待首次评价",ha="center",va="center",transform=ax.transAxes,color="#67716D")
            ax.grid(axis="x",visible=False)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.925),ncol=3,frameon=False,fontsize=11)
    fig.suptitle("C146｜求解器特征是否有帮助",x=.06,y=.985,ha="left",fontsize=18,fontweight="bold",color="#303B37")
    fig.text(.94,.975,f"已完成 {data['completed_epochs']} / 128 轮",ha="right",va="top",fontsize=11,color="#66716C")
    fig.subplots_adjust(left=.075,right=.97,top=.83,bottom=.18,hspace=.52,wspace=.24)
    fig.text(.075,.027,"新地形：128 块 / 628 图，每 4 轮评价。训练集：1984 块 / 2815 图，每 8 轮评价。\n全部原生有效像素；仅连接已完成的评价点。过程成绩尚未经过本轮最终独立计分。",fontsize=9.5,color="#65706B")
    temp=OUT/"curves.partial.png";fig.savefig(temp,dpi=160,bbox_inches="tight",pad_inches=.16);plt.close(fig);temp.replace(OUT/"curves.png")
    meta=OUT/"curve_data.partial.json";meta.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n");meta.replace(OUT/"curve_data.json")

HTML="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C146 实时训练曲线</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#303b37;background:#f5f6f3}
body{max-width:1150px;margin:0 auto;padding:28px 22px}h1{font-size:26px;margin:0 0 8px}p{line-height:1.6}.muted{color:#69736e;font-size:14px}
.card{background:white;border:1px solid #dfe4dd;border-radius:12px;padding:20px;margin:18px 0}
#status{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}.bar{height:12px;background:#e7ebe5;border-radius:10px;overflow:hidden;margin-top:14px}
#fill{height:100%;background:#648477;transition:width .3s}table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:right;padding:12px;border-bottom:1px solid #e7ebe5}td:first-child,th:first-child{text-align:left}th{color:#65706b;font-weight:500}
.with{color:#648477}.without{color:#a47059}img{width:100%;height:auto;display:block}a{color:#456f5d}.small{font-size:13px}
</style><body>
<h1>C146 实时训练曲线</h1><div class="muted">同一网络、划分、初始化和 128 轮预算。只比较是否使用求解器特征。</div>
<div class="card"><div id="status"><strong id="stage">正在读取</strong><span id="progress"></span></div><div class="bar"><div id="fill"></div></div>
<p class="muted" id="detail"></p></div>
<div class="card"><strong id="evaluation-title">最新新地形评价</strong><p class="muted">RMSE 是平均预测误差的指标，单位 dB。越低越好。两频分别比较。</p>
<table><thead><tr><th>模型</th><th>415 MHz</th><th>5.775 GHz</th></tr></thead><tbody id="metrics"></tbody></table></div>
<div class="card"><img id="curves" alt="两组在新地形和完整训练集上的双频RMSE曲线" src="/curves.png"></div>
<p class="muted"><b>怎么看：</b>新地形误差下降，表示预测能力在改善。训练集误差下降而新地形误差上升，表示模型开始过拟合。先比较两组在相同轮数的差异，再看各组最优结果。</p>
<p class="muted">大误差区域、原准确区域新增错误及 484 块地形确认集，按原协议在训练结束后核对。当前页面不将它们标成已测。</p>
<p class="muted small" id="updated"></p><p class="muted small">页面每 10 秒读取最新记录。曲线只在完成一次评价后新增点。<a href="/curves.png" target="_blank">打开当前曲线图片</a></p>
<script>
let version="";
async function refresh(){
 try{
 const r=await fetch("/metrics.json",{cache:"no-store"});if(!r.ok)throw Error("读取失败");const d=await r.json();
 document.querySelector("#stage").textContent=d.stage;
 document.querySelector("#progress").textContent=d.percent.toFixed(1)+"% · 两组各 "+d.done.toLocaleString()+" / "+d.total.toLocaleString()+" 次更新";
 document.querySelector("#fill").style.width=d.percent+"%";
 document.querySelector("#detail").textContent="当前第 "+(d.progress.epoch||0)+" / 128 轮，本轮 "+(d.progress.batch||0)+" / "+(d.progress.batches||352)+" 批。已完整保存 "+d.completed_epochs+" 轮。";
 const m=d.monitor;
 document.querySelector("#evaluation-title").textContent=m?"最新新地形评价 · 第 "+m.epoch+" 轮":"等待首次新地形评价";
 const rows=[];
 if(m){for(const [key,label,cls] of [["with_solver","使用求解器","with"],["without_solver","不用求解器","without"]]){
 const rm=m.monitor[key].rmse;rows.push('<tr><td class="'+cls+'">'+label+'</td><td>'+rm[0].toFixed(3)+'</td><td>'+rm[1].toFixed(3)+'</td></tr>');}}
 const ref=d.reference.monitor.rmse;rows.push("<tr><td>现有 C82 参考</td><td>"+ref[0].toFixed(3)+"</td><td>"+ref[1].toFixed(3)+"</td></tr>");
 document.querySelector("#metrics").innerHTML=rows.join("");
 if(version!==d.curve_version){document.querySelector("#curves").src="/curves.png?v="+d.curve_version;version=d.curve_version;}
 document.querySelector("#updated").textContent="最近读取："+d.updated+"（北京时间）";
 }catch(e){document.querySelector("#stage").textContent="连接中断，请检查 SSH 连接";}}
refresh();setInterval(refresh,10000);
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split("?",1)[0]
        try:
            if path=="/":body=HTML.encode();content="text/html; charset=utf-8"
            elif path=="/metrics.json":
                data=snapshot();data["curve_version"]=read(OUT/"curve_data.json",{}).get("curve_version","0")
                body=json.dumps(data,ensure_ascii=False).encode();content="application/json; charset=utf-8"
            elif path=="/curves.png":body=(OUT/"curves.png").read_bytes();content="image/png"
            else:self.send_error(404);return
            self.send_response(200);self.send_header("Content-Type",content);self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:print("request_error",repr(e),flush=True);self.send_error(503)
    def log_message(self,*args):pass

def follow(last):
    while True:
        try:
            d=snapshot()
            if d["curve_version"]!=last:render(d);last=d["curve_version"];print("rendered_epoch",d["completed_epochs"],flush=True)
        except Exception as e:print("render_error",repr(e),flush=True)
        time.sleep(5)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--once",action="store_true");ap.add_argument("--port",type=int,default=16006);a=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.once:render(snapshot());return
    lock=(OUT/"server.lock").open("w")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("Live curves already running")
    data=snapshot();render(data)
    server=ThreadingHTTPServer(("127.0.0.1",a.port),Handler)
    (OUT/"server.json").write_text(json.dumps({"pid":os.getpid(),"bind":"127.0.0.1","port":a.port,"read_only_training":True,"CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"]},indent=2))
    threading.Thread(target=follow,args=(data["curve_version"],),daemon=True).start()
    print("serving",a.port,flush=True);server.serve_forever()
if __name__=="__main__":main()
