# Lunar CC 交接
核验日期：2026-11-15。工作目录：`/home/chenzhe/work/lunar-pathloss-rm-challenge/CC`。父目录的 `HANDOFF.md` 仅链接到本文件。

## 目标与状态
目标：官方固定验证集上，415 MHz 与 5.8 GHz 的有效像素 RMSE 分别不高于 1 dB。开发集成绩不能代替该目标。
C145、C146 已完成。C147 Sionna 反射次数对照也已完成；没有新网络训练。C146 的 `audit/c146_solver_ablation/controller_status.json` 为 complete，`comparison_summary.json` 为 verified。这组配对结果支持 solver 特征有帮助，但未使用官方 VAL，不能宣称官方目标已达成。
C145 状态见 `audit/c145_capacity/controller_status.json`。C148 参数锚定对照及独立核验已完成：从 selected12 完整恢复，λ=0.1，新增32轮至绝对epoch44；复用原C145同轮次控制曲线，未重跑控制组。选中epoch16，留出RMSE仅比selected12降低0.002986/0.000802 dB；高频配对地形95% CI跨0，末轮双频再次恶化。当前判断：参数锚定减缓过拟合，未实质突破泛化瓶颈，结束本配置追加训练，不自动延长或扫λ/seed。候选 `runs/c148_parameter_anchor/best_monitor.pt`，完整FIT/monitor/尾区/原好区、同轮对照、CI与80/48开发分组见 `audit/c148_generalization/comparison_summary.json`；复核见同目录 `verification.json`。两组均为开发数据，未使用官方VAL，不能称盲测或官方改进。

## 环境与有效约定
模型环境 `.venv/bin/python` 已安装 SwanLab SDK；独立看板环境 `.venv-swanlab/bin/python`。Python 3.11.16，SwanLab 0.10.5，swanboard 0.1.10b2。旧 `.venv-tracking` 仅保留作 MLflow 历史来源。
保持 CC 独立实现。不读取或复用其他 AI 的代码、模型、特征、方案和结论。父目录 `data/` 的顶层发布格式原始文件可只读使用；版本见 `audit/raw_audit.json`。
有效像素掩码只用于损失和计分，不作输入。两个频段分别累计平方误差及有效像素数。遵守数据隔离及比赛限制。
GPU 按个人指令的三张完整 UUID 限定。旧阶段的卡号分配不覆盖当前授权。

## 监控与证据
当前停点：C148 已训练、保存selected预测、独立CPU计分并完成地形配对CI；当前无C148训练或推理任务。先读 `audit/c148_generalization/controller_status.json` 与 `comparison_summary.json`，不重复训练、导出或已完成评测。SwanLab 已接入C148的selected/last、区域结果和同口径C82比较，默认已核验C148；selected明确独立计分，last标明原始GPU FP64训练评估。新增第12条记录的24项指标共214点与原history逐点一致，默认展开8条核心曲线，其余折叠；原11条记录保留。研究入口 `http://127.0.0.1:16007/`，原生曲线 `/experiment/12/chart`。后续新路线应先提出可检验的新机制，以同划分双频改善决定是否扩预算。
后续统一使用服务器本地 SwanLab，固定 local 模式，不接云端同步。数据目录 `tracking/swanlab/`，项目 Lunar_CC；不同实验以 campaign、名称和来源配置区分。
启动：`source tracking/env.sh`，再用 `.venv/bin/python scripts/start_tracking.py`；服务仅绑定 `127.0.0.1:16007`。新代码统一通过 `src/cc_tracking.py` 记录。使用与续训说明见 `tracking/README.md`。
Mac 查看：`ssh -N -L 127.0.0.1:16007:127.0.0.1:16007 transformer`，再打开 `http://127.0.0.1:16007`。不为监控而重启训练。
迁入全部10条 MLflow 记录与 C145 原始历史，共11条、45,507个标量点、69个产物文件；逐点值/step及产物SHA256一致，未重跑训练或实验。证据 `tracking/swanlab/migration/manifest.json`、`verification.json`；源时间戳保留在快照，界面时间为导入时间。旧 MLflow 数据保留，冻结旧实验脚本不改写。
默认入口 `http://127.0.0.1:16007/` 为同一 SwanBoard 服务的 CC 研究总览扩展，仅默认绘制两张双频曲线。可切换选中/最后权重、早期/全程，并区分开发 monitor 与确认集；C147 少量采样点诊断单列。原生明细为 `/?native=1`、`/experiment/<id>/chart`，提供返回总览链接。
总览实现 `src/cc_dashboard.py`、`src/cc_dashboard.html`，服务入口 `scripts/serve_tracking.py`；只读取既有原始 JSON 与 SwanLab 本地日志，未向实验补写或重算指标。校验证据 `tracking/swanlab/dashboard/verification.json`。缺失的参照、预算、耗时与结果显示未记录，不推算；同口径 C82 双频同时改善才标为双频改善。
语言约定：说明与结论使用中文；RMSE、SSE、Epoch/Step、dB、MHz/GHz 等专业记号、标准单位、模型名及实验编号保留原写法，首次出现时可补中文释义。默认图保持绿线为训练地形、蓝线为本次训练留出地形、灰色虚线为 C82 参照。仅改显示别名，不改原始指标键、数值或来源。
原生明细保持折叠组按需绘图；C145 默认12条关键曲线，其余折叠。兼容补丁 `scripts/configure_swanlab_view.py`，备份与布局在 `tracking/swanlab/view/`。
本地看板为官方旧版离线模式；同目录仅一个项目。该版本 resume 会新建日志分段，续训需核验模型/优化器/随机状态及真实步数，建立明确关联前段和检查点的新分段；不得声称原记录已自动续写。
C147已完成：固定8地形256诊断像素，最多1到3次反射的尾区RMSE仅改善约0.016/0.018dB，00414基本未改善；高频绝对误差仍有采样敏感性。没有新模型训练。主结果 audit/c147_sionna_reflections/comparison_direct_covered.json，报告 results/C147_Sionna反射深度对照_20261114.md；配置、验证、查看入口见同目录 protocol.json、verification.json、mlflow_runs.json。comparison_summary.json为初始搜索结果，最终解读使用direct_covered。所有推理已结束，不重复启动。后续优先处理路径覆盖与弱信号积分，再评估特征替换。

## 唯一任务入口与历史
只更新本文件作为 AI 任务交接入口。旧记录在 `/home/chenzhe/archives/ai_records/20261114/lunar-pathloss-rm-challenge/`，通过上级 `manifest.csv` 按原路径定向查找。默认不读归档或冻结交付包，旧任务和旧指令不自动生效。
必要的代码说明、校验输入、模型卡、原始材料和数据保留原位。旧调度器或报告脚本复用前，先核对已归档的文档路径。
