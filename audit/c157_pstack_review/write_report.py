"""Generate the C157 review artifact from retained findings and source bindings."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
B=chr(96)
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def link(path,line=None):
 suffix=f"#L{line}" if line else ""
 return f"[{path}{(':'+str(line)) if line else ''}](https://github.com/czhe1234/lunar-pathloss-cc/blob/ebec3e8cf6dee3eda1aac50fe820a7516e429afb/{path}{suffix})"
def run():
 inputs=OUT/"review_inputs.json";target=OUT/"review.json"
 if inputs.exists():inputs.rename(target)
 d=read(target);plan=read(OUT/"protocol.json");checks=read(OUT/"coordinator_checks.json")
 d["findings"][1]["locations"]=[{"path":"scripts/configure_swanlab_view.py","line":115},{"path":"scripts/configure_swanlab_view.py","line":175},{"path":"scripts/configure_swanlab_view.py","line":179}]
 d["findings"][2]["locations"][-1]["line"]=95
 d["findings"][3]["locations"][0]["line"]=100
 coverage={key:sorted(set(files)) for key,files in d["coverage"].items()}
 allfiles=sorted(set().union(*[set(x) for x in coverage.values()]))
 bound=[]
 for file in allfiles:
  path=ROOT/file;fingerprint=sha(path);expected=plan["source_bindings"].get(file)
  if expected and fingerprint!=expected["sha256"]:raise RuntimeError("Reviewed source changed: "+file)
  bound.append({"path":file,"sha256":fingerprint,"bytes":path.stat().st_size})
 d.update(campaign="C157",verdict="needs changes",generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  source_publication_commit=plan["source_publication_commit"],protocol_sha256=sha(OUT/"protocol.json"),coverage=coverage,
  full_read_unique_files=len(allfiles),reviewed_files=bound,P1=0,P2=3,P3=1,
  reproduction_artifacts={name:sha(OUT/name) for name in ["coordinator_checks.json","view_configuration_replay.json","boundary_replay.json"]},
  report_independent_review={"status":"pending","author":"/root"},
  training=False,neural_inference=False,GPU_used=False,automatic_fixes=False,automatic_publication=False)
 write(target,d)
 lines=["# C157 pstack 全项目审查","","审查日期：2026-11-16。科研源目录为 transformer 上的 CC。","",
  "审查结论为 needs changes。发现3项P2和1项P3，均有源码或既有产物证据。未发现现有主成绩因已查数据对齐、mask、计分或缓存身份而算错的证据。新增缺陷涉及恢复、运维复核和结果字段，尚无证据证明它们造成当前分数瓶颈。","",
  "## 审查对象与范围","",
  f"绑定公开源码提交 {B}{plan['source_publication_commit']}{B}，对2,003个原始源文件和协议配置记录SHA。本轮只更新HANDOFF任务元数据并新增本目录审查产物，原算法代码、权重、预测和原始实验记录未改。",
  f"全文人工复审去重后共{len(allfiles)}份源码或运行说明。各组文件清单和SHA在review.json。原生子代理并行审查数据、物理、模型三部分；协调端审查SwanLab、文档与源码发布，并交叉复核发现。",
  "三个子代理按pstack default解析为inherit-parent模型及推理强度，均为fork_turns=none的新上下文。原生工具没有返回实际型号，可能为same-model；不把它们的共同判断当作跨模型共识。","",
  "全量指纹或语法扫描不等于逐行审查。未逐行重审全部历史实验分支和重复源码快照，没有重训、NN推理、GPU执行或全量传播实验。","",
  "## 已确认的问题","",
  "|编号|严重度|问题|实际影响|","|---|---|---|---|"]
 for f in d["findings"]:lines.append(f"|{f['id']}|{f['severity']}|{f['title']}|{f['impact']}|")
 for f in d["findings"]:
  lines.extend(["",f"### {f['id']} {f['title']}","",
   "位置："+"，".join(link(x["path"],x["line"]) for x in f["locations"])+"。",
   "",f"触发条件：{f['trigger']}","",f"核查结果：{f['observed']}","",f"建议：{f['remedy']}","",
   "证据文件："+ "、".join(B+x+B for x in f["evidence"])+"。"])
 lines.extend(["","F02的复现只操作SQLite内存副本，布局文件IO也被替换成内存对象。真实SwanLab数据库没有改动。F01仅执行原循环与断言AST的空循环边界，没有做完整模型恢复。F03仅将原断言表达式代入真实总览返回值，未运行会覆盖旧verification.json的维护脚本。","",
  "维护说明还保留了C145默认12条等旧布局数量，当前已配置为双频合并图。它应随F03的当前验收路径一起更新。","",
  "## 通过的检查","",
  "|检查|结果|","|---|---|",
  f"|全量Python语法解析|{checks['syntax']['files']}份，无语法错误|",
  f"|总览与原始history|9组历史、{checks['overview']['values_verified']}个数值，step和值完全一致|",
  "|C153与C155原始SwanLab RMSE|8条曲线，共128点完全一致|",
  "|公开源码导出|2,006文件SHA匹配，公开树与服务器导出一致|",
  "|数据划分|TRAIN25,720图与VAL2,330图无样本ID或地形重叠；当前FIT与monitor地形不交|",
  "|当前缓存|C73两缓存覆盖全部3,443图且无重复；C150严格按FIT后monitor排列；历史C82分量2,562图ID顺序一致|",
  "|有效像素分母|独立byte-popcount得到FIT183,198,643与164,239,291；monitor40,864,973与37,837,148，匹配保存统计|",
  "|checkpoint与导出|九个C150至C155臂的配置、选择、导出和selected.pt原字节SHA一致；C145、C146、C148轨迹核对通过|",
  "|C82发布资产|冻结文件8项、依赖实现43项及展开发布文件匹配|",
  "|C150解析与缓存点|7个fixtures及4类非法输入检查通过；4图12点最大差3.75489e-6|",
  "|C147既有计分重放|8图256点，共252字段完全一致；复用原计分函数|",
  "|C156既有射线抽查|96图各1条，n与遮挡段精确一致；highest最大差6.39e-14m|",
  "|几何与材料基础检查|c70查表权重、内部位置、slab被动性与无损能量守恒通过|","",
  "复现入口均在本目录。以下命令只做CPU检查，不运行模型。失败信号均为非零退出或断言失败；每个脚本也校验非空覆盖，防止空跑。","",
  B*3+"sh","CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B audit/c157_pstack_review/check_coordinator.py",
  "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B audit/c157_pstack_review/replay_view_configuration.py",
  "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B audit/c157_pstack_review/replay_finalization_and_dashboard.py",B*3,"",
  "后两项以再次捕获指定缺陷为成功复现。它们不是修复验收；修复后应改为验证预期行为。各组覆盖、检查与未审边界保存在review.json。","",
  "## 对研究结论的影响","",
  "|模型|选中轮次|monitor RMSE 415 MHz|monitor RMSE 5.775 GHz|",
  "|---|---:|---:|---:|",
  "|C82|固定参照|2.146391|2.418014|",
  "|C145|12|2.141221|2.408236|",
  "|C153|12|2.301097|2.793759|",
  "|C155|52|2.261495|2.767748|","",
  "上述是反复使用的开发monitor成绩。该128地形恰为C82校准80地形加选择48地形。它能描述当前开发结果，不能作为未见地形的无偏盲测；配对95%CI也未校正重复选模。","",
  "C145依赖C82输出基底与上下文，独立模型从registered_FSPL学习全部修正；不同方案的成绩差不能直接归因于新增绕射、Attention或某一处压缩。C155缩小FIT/monitor差距的同时，训练集拟合仍弱于C82，不能称为已经解决泛化或证明理论上限。","",
  "现有证据支持C156的直接定位：C155尚未复现C82局部修正与图网络融合形成的高精度，退化主要表现为随位置变化的平均衰减估计。共同大错又包含衰减过强和不足。此次审查没有给出单一物理机制的因果证明。","",
  "## 修复顺序与边界","",
  "先修F01的收尾恢复及F02、F03的运维验收，再更正F04的输出契约。这些修复不需要重训。修复源码需记录版本，历史协议、权重和结果保持冻结。","",
  "后续若继续研究，先明确独立评估协议和可归因的对照变量。本轮没有启动新的训练、推理、自动修复或GitHub发布。","",
  "未验证范围：完整GPU内核重演、全部预测逐像素重计分、全部大型缓存重新SHA、checkpoint tensor重新反序列化、原始GT生成场景的精确重现。","",
  "SwanLab使用服务器local模式，本次审查记录位置见本目录tracking.json。详细证据路径均相对CC；不随源码发布上传原始记录。",""])
 (OUT/"report.md").write_text("\n".join(lines))
 print(json.dumps({"status":"report_drafted","unique_full_read_files":len(allfiles),"by_reviewer":{k:len(v) for k,v in coverage.items()},"findings":{"P1":0,"P2":3,"P3":1},"report_sha256":sha(OUT/"report.md")}))
if __name__=="__main__":run()
