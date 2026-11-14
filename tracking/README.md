# CC 实验记录

后续训练、持续评估和分析统一使用 SwanLab 0.10.5，模式固定为 local。
记录目录为本项目的 tracking/swanlab/，服务仅监听服务器 127.0.0.1:16007。
模型环境 .venv 已安装 SDK；独立看板环境为 .venv-swanlab。
不登录云端，不执行 swanlab sync。tracking/env.sh 和 src/cc_tracking.py 强制本地配置。

## 启动与查看

服务器 CC 根目录执行：

```bash
source tracking/env.sh
.venv/bin/python scripts/start_tracking.py
```

Mac 通过 SSH 转发后打开 http://127.0.0.1:16007：

```bash
ssh -N -L 127.0.0.1:16007:127.0.0.1:16007 transformer
```

## 后续代码接入

从 src 脚本中使用：

```python
from cc_tracking import init, log, finish

run = init(name="Cxxx | arm", campaign="Cxxx", config=run_config)
log({"train/loss": loss, "optim/learning_rate": learning_rate}, step=global_step)
# 按冻结协议记录 FIT、monitor、tail 的双频 RMSE，保留各自分母和单位。
finish()
```

所有实验进入本地 Lunar_CC 项目，用 campaign、实验名和参数区分。
不要改变原有指标口径。监控不替代原始 JSON 日志、检查点和校验记录。
启动训练前仍需向用户说明配置、曲线、指标和预计耗时。

本版本 local 的 resume 会新建目录，不能假设它在原记录上追加。
续训先核验模型、优化器、随机状态和真实 global_step 的恢复，再建立新分段：
init(..., parent_run_id="上段ID", parent_checkpoint="恢复路径", start_step=global_step)。
使用新的 run_id，继续记录真实步数；helper 拒绝 resume="must"/"allow"。
这次迁移未启动任何续训。

## 历史迁移与可复核证据

迁入 C146/C147 的全部 10 条 MLflow 记录，以及 C145 已有原始 JSON 历史。
共 11 条记录、45,507 个标量点、69 个源产物文件。
保留原指标名、值、step、配置和产物；SHA256 与逐点核对见：
tracking/swanlab/migration/verification.json 和 manifest.json。
复核命令为 .venv-swanlab/bin/python scripts/verify_swanlab_migration.py。
只读取源快照和导入后的日志，不运行训练、推理或科学评估。

原始时间戳保存在 source_export.json/source_info 中；看板的创建时间和时长为导入时间。
原 MLflow 的 tracking/mlflow.db、tracking/artifacts/ 和 .venv-tracking 保留作历史来源，
默认服务与后续 SDK 入口已切换。冻结的旧实验脚本仍保留当时的记录代码，供历史复现。

## 本地看板的边界

官方将这个无需 Docker 的离线看板标为旧版、低维护模式：
https://docs.swanlab.cn/en/guide_cloud/offline-board.html 。
同一日志目录在看板中只呈现一个项目；来源项目在 manifest/source_info 中保留。
产物文件在对应 run 的 files/artifacts/ 下，artifact_index 记录路径和 SHA256；
不是云端附件服务。全部原始记录和权重继续留在服务器。

## 页面性能与展示

C145 默认展示12条关键曲线，C146每条记录7张，C147最终每条8张。
其他历史统计完整保留在折叠组；先在“搜索图表”中筛选，再展开结果。
默认折叠组不创建图表实例。此前旧前端使用仅隐藏显示的方式，
即使折叠也会创建全部图表，已通过 scripts/configure_swanlab_view.py 修正。
启动脚本会校验这项可复现补丁；固定适配 swanboard 0.1.10b2。
原前端、原展示状态和当前布局配置在 tracking/swanlab/view/ 下。
后续记录只为训练进度、损失、学习率和关键双频评测值创建曲线；
频率、地图数量等静态量放配置，详细分组与逐图结果保存为已有 JSON 产物。
不得为减少图表而删除原始日志、已有指标或结果。

## 四块研究总览

默认打开 http://127.0.0.1:16007/ 。这是同一 SwanBoard 本地服务的 CC 总览扩展。
启动仍用 scripts/start_tracking.py；实际服务脚本为 scripts/serve_tracking.py。
1. 进度、原始阶段耗时、预计剩余时间，以及是否继续同配置训练的判断与理由。
2. 两张双频图合并展示训练集、地形留出评测、各自同口径 C82 参照；无平滑和补点。
3. 固定大误差区、相对本分支未见地形、原好区的改善与退步，并保留原始分母。
4. 同一面板下的 C82、选中权重、最后权重对比。确认集单独切换，C147 物理采样点诊断单列。

C145 小幅开发收益与后期泛化退步同时显示；不把“最好权重”误作“最后权重”。
C145 耗时来自 phases.json 的非 export_only 阶段累计；C146 使用两臂配对训练的原始总时长。
耗时含阶段内评测，不能按表格的权重行重复累加。迁移创建时间不作训练时长。
新地形仅指相对当前分支 FIT 留出；monitor 是重复使用的开发数据，C82 曾见其中80/128地形。

原生全部明细保留在 /?native=1，单条原始曲线仍在 /experiment/<id>/chart，均可返回总览。
来源接口只开放明确列出的实验 JSON，不接受任意文件路径。服务仍仅监听服务器回环地址。
实现及复核：src/cc_dashboard.py、src/cc_dashboard.html、scripts/verify_swanlab_dashboard.py；
证据保存在 tracking/swanlab/dashboard/verification.json。原45,507指标点和69产物校验继续通过。

后续通过 cc_tracking.init 建立的记录会自动出现在总览。建议统一记录：
- progress/epoch、progress/total_epochs；step 使用真实全局进度，不为展示重新编号。
- time/elapsed_seconds、time/eta_seconds；无法给出可靠 ETA 时不记录，界面显示未记录。
- fit/RMSE_415MHz_dB、fit/RMSE_5p775GHz_dB；必须为完整 FIT 同口径评测。
- monitor/RMSE_415MHz_dB、monitor/RMSE_5p775GHz_dB；必须为固定评测面板。

在 run_config 的 _cc_dashboard 中声明 scope、baseline_source、total_epochs，
以及 C82.fit/C82.monitor 的已核验双频数组。scope 与 baseline_source 缺失时不作 C82 胜负判断。
这些参照须来自同一冻结划分、像素掩码和聚合口径的原始记录；没有就显示缺失，不能借用其他面板。
新实验未记录的最好权重与分组结果也不补造。此处整理没有启动任何训练或实验。

图表说明默认用中文；RMSE、SSE、Epoch/Step、dB、MHz/GHz 等专业记号和标准单位保留原写法。总览中的纵轴为 RMSE（预测误差，单位 dB），越低越准；绿线对应训练过的地形，蓝线对应本次训练留出的地形，灰色虚线对应同口径 C82 留出评测参照。每个频段下附原实验选中版本与训练终点的实际差值说明。原生曲线仅翻译标题、悬停显示和搜索别名，不更改原始指标键、step 或数值；兼容实现仍在 scripts/configure_swanlab_view.py。
默认曲线只保留绿、蓝和灰色虚线三类，训练地形自己的 C82 参照继续保存在区域表与原始记录中。
