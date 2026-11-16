# Lunar CC 交接
核验日期：2026-11-17。工作目录：`/home/chenzhe/work/lunar-pathloss-rm-challenge/CC`。父目录的 `HANDOFF.md` 仅链接到本文件。

## 目标与状态
CC 依赖核查与源码归档已完成，未训练、神经网络推理、重跑实验或使用 GPU。归档位置为 `archive/20261117_cc_cleanup/`，共414个 Python 源文件：407份历史快照，其中404份与现有源码完全相同；5个已完成的一次性控制/预热脚本；2个已结束的C151b/C152模型定义。未合并或删除重复文件的原始内容。
本次源码发布将整理后的布局、C157审查与归档管理源码同步到现有公开GitHub仓库。新发布流程位于 `deliveries/github_lunar_pathloss_cc_20261117/prepare_snapshot.py`；提交414条Git相对软链及全部归档目标，补齐 `results/c146_live/academic.mplstyle`。发布副本含 `SOURCE_ARCHIVE.json` 干净映射和 `scripts/verify_source_snapshot.py` 只读验收器，不上传原始审查JSON、训练记录、数据库、权重或预测。源清单、独立验收及远端提交核验分别见该交付目录的 `inventory.json`、`publication_review.json`、`upload_verification.json`。
依赖清单、归档映射、整理前文件身份及最终核验见 `audit/cc_cleanup_20261117/` 下的 `active_dependencies.json`、`frozen_references.json`、`archive_plan.json`、`inventory_before.json`、`verification.json`。独立复核见同目录 `review.json`；本地SwanLab整理记录见 `tracking.json`，说明更正见 `tracking_correction.json`。原说明保留，成绩称谓与历史动作时间以更正说明为准。依赖分析采用独立上下文，模型与推理预算继承主代理，运行工具未返回实际型号，可能为同一模型。

每个归档源文件的旧位置保留相对软链，旧路径仍可读取相同字节并核验SHA；同文件系统移动保留跟链stat的device/inode/size/mtime_ns。当前依赖、原生库、核函数、使用自身路径定位资源的代码保持原位普通文件。所有 `releases/`、`deliveries/` 和既有 `audit/` 文件未归档，原始数据、缓存、权重、预测、日志和冻结JSON保持原位。两项既有监控进程未停止或重启。
归档目录用于保存历史源码，不作为运行工作目录。继续从CC根目录使用当前入口，不启动已完成的历史控制器。软链不等于任意旧脚本在任意目录都可直接执行。20261115/16旧发布导出器会拒绝或跳过软链。本次使用20261117新流程，按归档映射保留相对链接并收录全部目标，不执行源码还原，不改写既有发布快照。后续同步复用新流程，并核对Git链接模式、链接文本与目标内容。
在CC根目录核验归档：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B audit/cc_cleanup_20261117/manage_archive.py verify`。源码位置还原用同一命令将末尾 `verify` 改为 `restore`；只还原源码和兼容链接，不改模型或记录。命令非零退出表示失败；核验还须看到非空检查计数及 `failures=0`。

当前入口按用途查找，不按实验编号猜测依赖：

| 用途 | 入口或证据 |
|---|---|
| C82冻结推理与发布复核 | `src/c82_inference.py`、`src/c82_predict.py`、`scripts/c82_release_replay.py`；冻结目录 `runs/c82_selected/`、`releases/C82_20261103/` |
| C145 selected12成绩核查 | `src/c145_report.py`、`src/c145_verify.py`、`runs/c145_capacity_native/selected_export.json` |
| C153与C155既有独立基线 | `src/c153_report.py`、`src/c155_report.py`、`src/c155_grazing.py`；训练源码保留，但不启动 |
| C156瓶颈归因 | `src/c156_*.py`、`src/c156_counts.cpp`；结果 `audit/c156_attribution/result.json`、`report.md` |
| C157项目审查 | `audit/c157_pstack_review/report.md`、`review.json`；检查入口 `check_coordinator.py` |
| SwanLab与看板 | `src/cc_tracking.py`、`scripts/start_tracking.py`、`scripts/serve_tracking.py`、`scripts/c146_curves.py` |

当前主要冻结参照为C145 selected12，双频RMSE为2.141221/2.408236dB；C82参照为2.146391/2.418014dB。C148 selected16的点估计为2.138235/2.407435dB，比C145低0.002986/0.000802dB；高频增益95%CI为[-0.001788,+0.002929]dB，未通过双频CI门槛，不能把主要参照称为最低分。来源为 audit/c148_generalization/comparison_summary.json。monitor为628图/128地形，是反复使用且部分参与C82校准的开发集，不是官方盲测。官方目标仍为双频各≤1dB。

C155轻量ConvNeXt-UNet已完成60epoch/21120更新，3,193,650参数，256/128/64/32多尺度跳跃连接。其余保持C153的raw68输入、registered_FSPL基底、C82槽位零、无D4、划分、图像权重、mask与loss。共同选中epoch52，FIT为2.299718/2.757400，monitor为2.261495/2.767748dB；末60轮FIT为2.297903/2.752470，monitor为2.259307/2.779240。相对C153 selected12改善0.039602/0.026011dB，高频配对CI跨0；相对C82两频都更差。训练与monitor差距缩小同时伴随训练拟合较弱，不能单凭gap称突破。协议、完整CI、Grazing支持域和独立CPU核验见 `audit/c155_unet/` 与 `runs/c155_unet/*_verified.json`。Swan训练界面32，双频RMSE两图及Grazing高频误差两图；原始20条曲线共320点已与history逐点核验。

C153已完成同预算60epoch，从头训练的15,214,786参数Field使用registered_FSPL基底、无C82、无D4。共同最优epoch12的monitor为2.301097/2.793759dB；末轮FIT为1.884180/2.056550，monitor为2.564186/2.965352dB，显示后期泛化退化。协议与结果见 `audit/c153_standalone_c151a/`，Grazing与single/multi纯统计见 `audit/c153_grazing_groups/`。Swan训练界面29，原始双频两图各含16个真实评价点。C154的D4检查未通过严格物理一致性门槛，未训练；原因及独立验证在 `audit/c154_d4/`，不能直接把普通D4加入现有注册几何。

C156纯CPU归因已完成。C82高精度基本盘主要来自local/graph学习修正和互补融合，多尺度校准再小幅改善。C155弱项主要表现为随位置变化的局部均值/衰减修正，统一平移、全局仿射或高通增益只消除小部分误差。两模型共同|err|≥5dB区占全部有效像素3.2075%/3.7437%，贡献C155总SSE的69.58%/61.66%；衰减过强与不足同时存在，不能统一解释为多重深绕射。完整定义、Top3区域和来源见 `audit/c156_attribution/`，Swan分析界面34。只复用了原预测并补做已记录的CPU统计，不再运行。

C157审查未发现已查主成绩、标签、mask、缓存身份或当前计分被算错的证据。已确认但未修复的问题为3项P2与1项P3：C153/C155末轮恢复分支的rate未绑定；Swan布局重配先提交且用旧tag快照校验；看板复核器固定旧11条记录；C153部分LoS命名字段实际用全体分母。完整覆盖与复现见 `audit/c157_pstack_review/`。C156绘图源码的旧协议SHA差异已有display-only修订记录，并匹配C157绑定，保持该既有状态，不改写冻结协议。

C145/C146/C147/C148及C149–C155历史实验均已结束，禁止重复启动、自动续训或增加配置。C149误差拆解、C150衍射、C151空间/极坐标、C152公平对照的定义与结果保留在各 `audit/c149*` 至 `audit/c152*` 目录。没有新模型通过相对C82与C145的双频地形配对门槛。本次只同步源码，发布后停止；不修复C157缺陷，不训练或续训。

## 环境与有效约定
源码发布：`https://github.com/czhe1234/lunar-pathloss-cc` 的 `main` 已公开，保留此前授权公开的旧C144提交历史。2026-11-17同步范围为截至C157的源码与本次归档布局，原始记录、数据库、数据、权重与预测继续留服务器。源码导出、清单和远端核验位于 `deliveries/github_lunar_pathloss_cc_20261117/`；上次20261116交付及旧历史备份保持原样。CC仍是科研源，非Git工作树；Git索引仅位于新的发布副本内，用于核验发布树。通过GitHub连接器以现有main为父提交更新，不重写历史。
模型环境 `.venv/bin/python` 已安装 SwanLab SDK；独立看板环境 `.venv-swanlab/bin/python`。Python 3.11.16，SwanLab 0.10.5，swanboard 0.1.10b2。旧 `.venv-tracking` 仅保留作 MLflow 历史来源。
保持 CC 独立实现。不读取或复用其他 AI 的代码、模型、特征、方案和结论。父目录 `data/` 的顶层发布格式原始文件可只读使用；版本见 `audit/raw_audit.json`。
有效像素掩码只用于损失和计分，不作输入。两个频段分别累计平方误差及有效像素数。遵守数据隔离及比赛限制。
GPU仅限个人指令三张完整UUID：GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009、GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0、GPU-70b6fceb-c7c4-9467-3c81-097c9284c160。允许共享，不要求空闲，不中断其他任务。运行前同时设置CC_GPU_UUID与CUDA_VISIBLE_DEVICES为同一完整UUID，子进程继承；旧卡号分配不覆盖当前授权。

## 监控与证据
C149–C151历史结果：其选中权重均经原始GT/有效mask的独立CPU Float64计分，置信区间与分组已记录local SwanLab。C150–C151新模型均只训练一次固定12fullFIT/4224更新，0/4/8/12轮完整评价；同一checkpoint按双频共同均值选择，不按频段拼权重。选中模型已独立核验，末轮未选中模型指标来自训练内完整评估。这些历史controller_status.json均complete，禁止重跑或重复报告。

| 实验/臂 | 选中Epoch | monitor RMSE：415 /5.775GHz dB |
|---|---:|---:|
| C82参照 | — | 2.146391 / 2.418014 |
| C145 selected12 | 12 | 2.141221 / 2.408236 |
| C148 parameter anchor | 16 | 2.138235 / 2.407435 |
| C152 | 12 | 2.145473 / 2.416657 |
| C153 standalone | 12（训练满60） | 2.301097 / 2.793759 |
| C155 ConvNeXt-UNet | 52（训练满60） | 2.261495 / 2.767748 |
| C150/A | 12 | 2.412450 / 3.070751 |
| C150/B | 12 | 2.433418 / 3.125203 |
| C151a/A | 12 | 2.428188 / 3.008245 |
| C151a/B | 8 | 2.367850 / 2.795412 |
| C151b/A | 8 | 2.359045 / 2.778327 |
| C151b/B | 8 | 2.361196 / 2.809100 |
C149以C145 selected12计分：注册场景点LoS仍贡献约52.45%/49.16% SSE；误差≥5dB像素只占3.75%/4.52%，贡献74.87%/76.10% SSE。报告 audit/c149_error_audit/summary.json、summary.md；无需再次跑几何或模型。 此前已保存的只读统计补充MeanBias及真实GT路损分桶：audit/c149_error_audit/supplement_bias_gt_bins.json（SHA d8924cc123832af3076487544a04c2a4447fa8ebc657c756563b132d262ad03d），复现脚本同目录supplement_bias_gt_bins.py；仅NumPy CPU统计，原冻结summary/模型/预测均未改。
C150：同输入、同网络，A=FSPL、B=FSPL+Deygout。B−A为+0.020968/+0.054451dB，两频CI均>0；共同后续prior选A。报告 audit/c150_diffraction_training/comparison_summary.json 及同目录comparison_summary.md，模型/原始预测 runs/c150_diffraction_prior/{A,B}/；确定性输入缓存 features/c150_diffraction/part0of1/，不重新导出。
C151a：同15,214,786参数，A固定共同打乱Global55物理通道，B保持对齐，Local/FiLM相同。B−A为−0.060339/−0.212833dB，两频CI均<0，支持本匹配控制中的空间对齐收益，但未超过C82/C145。B的epoch12高频FIT和monitor同时上升，保留joint-best epoch8，不能只归因过拟合。报告 audit/c151_global/comparison_summary.json 及同目录comparison_summary.md，模型/预测 runs/c151_global/{A,B}/。
C151b：共同fresh aligned Global+FSPL，不加载历史权重；同15,237,586参数，A Cartesian/B TX-polar，branch20,752参数、256×192/2层。B−A为+0.002151/+0.030773dB；CI分别[−0.000746,+0.004731]、[+0.007556,+0.052188]。低频无明确差别，高频polar更差；固定C82 tail略好、原好区更差，整体未获益。报告 audit/c151_polar/comparison_summary.json 及同目录comparison_summary.md，模型/预测 runs/c151_polar/{A,B}/。均未过双参考双频门槛，不扩预算。
C150–C151新网络均不输入C82预测或权重；历史模型比较只作性能参照，不是新增feature的单变量因果对照。monitor为反复使用的开发集，其中80地形参与过C82校准、48曾作selection，不能称盲测；CI未校正反复选择。 只读源码核查进一步确认：C145=冻结C82完整预测+Field残差，输入含(C82−legacyFSPL)/20双频context；Field从C123随机初始化训练8轮检查点加载，并非从C82网络参数恢复。C150完整保留原45c73，context置零、新增10衍射输入，换FSPL/Ld prior并从随机Field训练。前12轮数据/预算/LR/loss路径对应，但C145全程实际764轮、C150仅12轮，checkpoint搜索历史不等。故可比较成绩，不能把外部退化归因于新增feature或单一原因；仅C150内部A/B隔离Ld prior。
已知输入边界：basic11仍用旧raw-grid TX/高度定义，diffraction/FSPL与polar路由用registered定义；当前C95、C141–C144物理诊断未覆盖仅替换basic11定义的匹配网络训练。两套基准可由已知metadata转换，其并存不构成bug或因果证据，这些历史对照未临时更改basic/FiLM。
物理/路由近似的精确定义、数值校验、source/cache SHA与限制见各protocol.json：Deygout/Bullington不是严格上下界；冻结缓存floor不存在solver反传死梯度；polar比较同时改变采样密度/插值/coverage，不能仅归为径向顺序，也不是无损或所有polar方法的结论。

后续统一使用服务器本地 SwanLab，固定 local 模式，不接云端同步。数据目录 `tracking/swanlab/`，项目 Lunar_CC；不同实验以 campaign、名称和来源配置区分。
启动：`source tracking/env.sh`，再用 `.venv/bin/python scripts/start_tracking.py`；服务仅绑定 `127.0.0.1:16007`。新代码统一通过 `src/cc_tracking.py` 记录。使用与续训说明见 `tracking/README.md`。
Mac 查看：`ssh -N -L 127.0.0.1:16007:127.0.0.1:16007 transformer`，再打开 `http://127.0.0.1:16007`。不为监控而重启训练。
迁入全部10条 MLflow 记录与 C145 原始历史，共11条、45,507个标量点、69个产物文件；逐点值/step及产物SHA256一致，未重跑训练或实验。证据 `tracking/swanlab/migration/manifest.json`、`verification.json`；源时间戳保留在快照，界面时间为导入时间。旧 MLflow 数据保留，冻结旧实验脚本不改写。
默认入口 `http://127.0.0.1:16007/` 为同一 SwanBoard 服务的 CC 研究总览扩展，仅默认绘制两张双频曲线。可切换选中/最后权重、早期/全程，并区分开发 monitor 与确认集；C147 少量采样点诊断单列。原生明细为 `/?native=1`、`/experiment/<id>/chart`，提供返回总览链接。
总览实现 `src/cc_dashboard.py`、`src/cc_dashboard.html`，服务入口 `scripts/serve_tracking.py`；只读取既有原始 JSON 与 SwanLab 本地日志，未向实验补写或重算指标。校验证据 `tracking/swanlab/dashboard/verification.json`。缺失的参照、预算、耗时与结果显示未记录，不推算；同口径 C82 双频同时改善才标为双频改善。 此前已关联C150/C151的原history/result/selected_export与实时progress，正确区分选中/末轮及区域指标；分析记录不混作训练，默认active或最新训练，手选跨刷新保留。服务PID66080。此前配置看板时仅重启viewer，未重启训练；本轮整理未重启服务。
语言约定：说明与结论使用中文；RMSE、SSE、Epoch/Step、dB、MHz/GHz 等专业记号、标准单位、模型名及实验编号保留原写法，首次出现时可补中文释义。默认图保持绿线为训练地形、蓝线为本次训练留出地形、灰色虚线为 C82 参照。仅改显示别名，不改原始指标键、数值或来源。
原始曲线页面默认只显示双频两张合并图：每张为同频FIT绿线、monitor蓝线及C82灰色虚线；其余原始指标隐藏折叠保留。已配置总览内12个训练实验，Swan原始scalar tag/step/值及训练config未改。配置：.venv-swanlab/bin/python scripts/configure_swanlab_view.py --paired-rmse --experiment-id <native_id>；批量当前训练记录用--paired-all-training。原版0.1.10b2的native多指标渲染兼容补丁同脚本，元数据回退快照与配置在 tracking/swanlab/view/paired_rmse_*。脚本只改显示/关联，不生成或迁写训练指标。
本地看板为官方旧版离线模式；同目录仅一个项目。该版本 resume 会新建日志分段，续训需核验模型/优化器/随机状态及真实步数，建立明确关联前段和检查点的新分段；不得声称原记录已自动续写。
C147已完成：固定8地形256诊断像素，最多1到3次反射的尾区RMSE仅改善约0.016/0.018dB，00414基本未改善；高频绝对误差仍有采样敏感性。没有新模型训练。主结果 audit/c147_sionna_reflections/comparison_direct_covered.json，报告 results/C147_Sionna反射深度对照_20261114.md；配置、验证、查看入口见同目录 protocol.json、verification.json、mlflow_runs.json。comparison_summary.json为初始搜索结果，最终解读使用direct_covered。所有推理已结束，不重复启动。该结论只作历史参照，不自动启动后续计算。

## 唯一任务入口与历史
只更新本文件作为 AI 任务交接入口。旧记录在 `/home/chenzhe/archives/ai_records/20261114/lunar-pathloss-rm-challenge/`，通过上级 `manifest.csv` 按原路径定向查找。默认不读归档或冻结交付包，旧任务和旧指令不自动生效。
冻结校验输入、模型卡、原始材料和数据保留原位。历史源码归档及旧路径兼容方式见本文件当前状态和 audit/cc_cleanup_20261117/archive_plan.json。复用旧入口前先核对完成状态、依赖与工作目录。
