# Lunar CC 交接
核验日期：2026-11-16。工作目录：`/home/chenzhe/work/lunar-pathloss-rm-challenge/CC`。父目录的 `HANDOFF.md` 仅链接到本文件。

## 目标与状态
当前任务C156纯CPU归因核查已完成并独立复核，未训练、神经网络推理、GPU计算或重跑旧实验。628 monitor图/128地形、C155 selected52；结果 audit/c156_attribution/result.json、report.md，复核 verification.json、Top3 case_*.npz/region_*.png（均已render-inspect），协议及可复现源 src/c156_*.py/cpp。仅为共同大错并集补算2,015,853条CPU几何穿越次数；原highest/nu/FSPL/Ld均复用，所有受保护权重与预测未变。此任务完成后停止，不开新训练。
C82 base→base+local/graph并行→50:50融合→多尺度校准：good区RMSE415/5.775GHz为0.7441/0.8232→0.4531/0.5581或0.4334/0.5438→0.3911/0.4455→0.3791/0.4185；C155为0.5380/0.9089。高精度主要由学习修正及互补融合形成，校准为最后小幅改善。CPU冻结校准重组与C82 RMS差2.36e-6/4.28e-6dB。统一±2像素整数平移最优(0,0)，全局仿射最多移除约4% MSE，高通增益最多约1.4%；主要短板表现为随位置变化的局部均值/衰减修正，而非简单平移、全局尺度或过平滑。good由最终C82条件选取，需结合全图比较；不是因果训练消融。
共同大错严格为每频两者都|err|≥5，占全部有效像素3.2075%/3.7437%（旧C82 tail为3.7827%/4.5520%），占C155总SSE69.58%/61.66%；其中Clear/Grazing/NLoS为1.51/44.11/54.38%与19.11/24.04/56.85%。平均n1.5359/1.2933（n为穿越次数非山脊/反射数），≥2遮挡段仅18.77%/6.70%。高频c73 floor编码占共同区39.11%（全图10.18%）；floor定义Q=P·d_face²压缩值−3，并非绝对P触底。≥150m富集约1.5倍，图外圈4px约1.5–1.6倍，TX<10m未富集；不能称统一的多重深绕射或边缘问题。
Top3按8连通≥16像素区域的联合SSE固定选取：06501_05高频与05608_00低频主要衰减过强；01128_45高频单段深遮挡则衰减不足（低频共同区仅1像素，不推广）。两方向错误并存，单纯增大绕射损耗不能统一修复；当前几何/相关统计不能唯一证明GT生成机制。Swan服务器local q6zhvio0/界面34，36个最终指标step629与result逐点完全相同；图与报告留在CC，Mac仅三张查看副本。
当前C155轻量ConvNeXt-UNet对照已完成60epoch/21120更新，并独立CPU核验selected/last完整FIT、monitor及Grazing指标；训练PID2922786、核验PID3012164均已退出，未扩预算。网络3,193,650参数、256/128/64/32四级跳跃连接；其余严格保持C153 raw68、registered_FSPL基底、无C82/无D4、划分、图像权重/mask/loss及单阶段AdamW设置。协议与环境 audit/c155_unet/protocol.json，结果/核验 audit/c155_unet/comparison_summary.json、comparison_summary.md、controller_status.json；不重跑旧实验。
共同选中52轮：FIT2.299718/2.757400、monitor2.261495/2.767748dB。相对C153 selected12 ΔRMSE -0.039602/-0.026011dB，128地形配对95%CI分别[-0.044250,-0.035588]、[-0.082353,+0.036924]；低频改善明确，高频跨0，未通过双频CI门槛。相对C82仍差+0.115104/+0.349733dB，两频CI均>0，未突破既有成绩。末60轮FIT2.297903/2.752470、monitor2.259307/2.779240，末轮monitor-FIT=-0.038596/+0.026770dB，C153同预算末轮为+0.680006/+0.908803；缩小差距伴随训练集拟合较弱，不能单凭gap称突破。
选中52的Grazing高频误差FIT1.332283/2.294953、monitor1.306178/2.276631dB，均低于C153 selected12的1.338312/2.297831、1.320228/2.300201，但收益小；monitor GT/Pred高通RMS仍为1.386245/0.479454、2.407706/0.830721。定义/固定支持域/旧预测新统计见 grazing_support.json；新selected/last的独立CPU证据 runs/c155_unet/*_verified.json、*_grazing_verified.json、*_terrain_sse.npz。模型/优化器step21120、末lr1e-5和data-order RNG已验证；20个C82/C145/C153保护SHA全部未变。只此一次候选，结束等待用户选择，不自动追加。
Swan本地训练nff8a052（界面32，/experiment/32/chart），核验分析iv626lzy（界面33）；20条RMSE/Grazing相关曲线各16个点、共320点与原history的step/value逐点完全一致，tracking/swanlab/view/c155_metrics_verification.json。默认双频RMSE两图；Grazing高频误差另列可展开两图，灰线为C153 selected12 monitor。配置 scripts/configure_swanlab_view.py --paired-rmse --paired-grazing --experiment-id 32；仅改显示关联，原始记录保留。C153完整历史高频曲线未保存，仅补算其selected12现有预测，没有旧推理或训练。monitor为重复使用的开发集，CI未校正模型选择；单个网络替换同时改变容量、感受野和归纳偏置，不能归因于Attention或压缩中的单一因素。
历史C154_D4严格物理门槛未通过、未训练；张量/64次群合成及GT/mask/SSE同步通过，但现有TX注册及固定三角剖分不与普通D4交换。全部3443图TX检查及9图接收面/FSPL分量、72点独立复核见 audit/c154_d4/result.json、geometry_verified.json；20个保护SHA未变。Swan本地分析3txtvdaj/界面31。此为增强一致性检查，不是历史预测误差成因结论；当前C155不使用D4。
对照C153两阶段均已完成并独立CPU核验，训练进程970836已退出。60epoch/21120更新，单阶段AdamW实际step21120、末次lr1e-5，未重置optimizer；原C82/C145权重与历史预测SHA全部未变。状态/证据入口 audit/c153_standalone_c151a/controller_status.json、comparison_summary.json、comparison_summary.md；禁止重复启动、自动加训或新增配置。
第一阶段纯CPU统计已先在聊天输出，证据 audit/c153_grazing_groups/result.json、result.md（Swan分析 hfgq83ai）。全9点Grazing的GT/Pred高通RMS为1.386245/0.521167、2.407706/0.908823dB；高通误差1.283874/2.235644、局部均值误差1.194286/2.018708。single118图RMSE2.227796/2.542732，multi510图2.120681/2.378186；像素占比与Bias见原表。统计表用每频全部有效像素为分母，复核了C149总计分及Grazing像素；未训练、推理或跑几何。高频粗糙度不直接证明干涉/噪声；HP/LP非正交。
C153_standalone_c151a复用C151a aligned_global网络15,214,786参数，随机初始化，registered_FSPL+Field(raw68)，两路C82槽位严格零；C82只作计分参照。保持原fit/monitor、图像权重、mask和C145完整FIT归一化masked MSE/25。batch8、200步线性warmup至3e-4、余弦至1e-5；每4epoch完整评估，含epoch0共16点；未重跑历史实验。完整环境、cache/来源SHA与配置见 protocol.json（SHA49662b53e3bb0a3193075f67e9b913213986f73c34b3a648c90df1368c16ee4e）。
同一共同最优checkpoint为epoch12：FIT2.313586/2.790766，monitor2.301097/2.793759dB。相对C82 ΔRMSE +0.154706/+0.375745dB；128地形配对95%CI（2000rep/seed145073）分别[+0.132999,+0.177843]、[+0.302873,+0.442739]，双频均更差。末轮60的训练内完整评估FIT1.884180/2.056550、monitor2.564186/2.965352；长训练持续降低FIT但未改善monitor，选中轮次未更新。该方案未突破瓶颈，固定预算已结束，不自动继续。
独立CPU Float64已核对保存最佳预测的完整FIT/monitor及各固定误差组，未额外推理；证据 runs/c153_standalone_c151a/selected_export.json、selected_verified.json、selected_terrain_sse.npz。保护文件/源SHA结束后再次通过。此为反复使用、部分参与过C82校准的开发monitor，非盲测；CI未校正checkpoint选择；单次60epoch不能证明理论收敛上限。
Swan服务器本地训练 hvuycdb3（界面29，/experiment/29/chart），CPU独立比较 fpwo0slf。原始曲线默认双频两张合并图，四条原始RMSE各16个真实点，和history的step/数值逐点完全一致；其余指标隐藏折叠保留。配置/回退位置见下文监控约定。启动前解释器路径错误仅导入失败、0次更新，记录保留startup_python_error.log/controller；未形成额外训练。历史C153已结束；当前接续任务见上文C155。
历史C152两步均已完成：单候选12epoch/4224更新选中12，monitor2.145473/2.416657，相对C145 Δ+0.004252/+0.008421dB，两频CI均>0，未过双参考门槛。证据 audit/c152_los_diagnostic/result.json、audit/c152_fair_global/comparison_summary.json。其早先接线错误尝试未完成epoch即中止并保留 invalid_global_input* 与 runs/c152_invalid_global_input/，Swan25标INVALID；修正正式run xt81fqsf/界面26，训练PID66081已退出。协议、权重、历史预测保持冻结。
历史C149→C150→C151a→C151b已全部完成并独立核验。没有新模型通过相对C82及C145 selected12的双频terrain-paired95%CI门槛，未突破约2.15/2.42dB开发瓶颈；本轮结束，不自动加训、续训或新增配置。整体协议/全部冻结报告SHA在 audit/c149_c151/protocol.json，原用户指令在同目录directive.md。
长期官方目标仍为415MHz与5.775GHz有效像素RMSE分别≤1dB；本轮未用官方VAL，开发结果不能代替该目标。
旧C145/C146/C147/C148均已结束，禁止重复启动。C146支持solver输入有帮助；C147增加反射深度的尾区收益约0.016/0.018dB；C148参数锚定较selected12仅改善0.002986/0.000802dB，高频CI跨0，不扩预算。原证据分别见 audit/c145_capacity/、audit/c146_solver_ablation/comparison_summary.json、audit/c147_sionna_reflections/comparison_direct_covered.json、audit/c148_generalization/comparison_summary.json；C147初始comparison_summary不是最终direct-covered结论。
C149–C151历史结论只作参照，不重复；不得把原始生成缺失、floor、输入基准差别或训练回升直接当作已证实的误差原因。

## 环境与有效约定
源码发布：`https://github.com/czhe1234/lunar-pathloss-cc` 的 `main` 已公开，按用户授权保留并公开旧C144提交历史。本次源码同步覆盖至C156，包括C149–C156模型/训练/CPU核查源码、不可变算法协议、绘图样式及本地SwanLab集成；新训练原始记录、数据库、原始数据、权重与预测继续留在服务器。清单和远端核验见 `deliveries/github_lunar_pathloss_cc_20261116/inventory.json`、`upload_verification.json`，发布导出为同目录 `repository/`。CC仍是科研源，非Git工作树；本次通过GitHub连接器提交。旧历史完整备份保留在 `deliveries/github_lunar_pathloss_cc_20261115/private_history_before_sync.bundle`。
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
| C152 | 12 | 2.145473 / 2.416657 |
| C153 standalone | 12（训练满60） | 2.301097 / 2.793759 |
| C155 ConvNeXt-UNet | 52（训练满60） | 2.261495 / 2.767748 |
| C150/A | 12 | 2.412450 / 3.070751 |
| C150/B | 12 | 2.433418 / 3.125203 |
| C151a/A | 12 | 2.428188 / 3.008245 |
| C151a/B | 8 | 2.367850 / 2.795412 |
| C151b/A | 8 | 2.359045 / 2.778327 |
| C151b/B | 8 | 2.361196 / 2.809100 |
C149以C145 selected12计分：注册场景点LoS仍贡献约52.45%/49.16% SSE；误差≥5dB像素只占3.75%/4.52%，贡献74.87%/76.10% SSE。报告 audit/c149_error_audit/summary.json、summary.md；无需再次跑几何或模型。 本次只读核查补充MeanBias及真实GT路损分桶：audit/c149_error_audit/supplement_bias_gt_bins.json（SHA d8924cc123832af3076487544a04c2a4447fa8ebc657c756563b132d262ad03d），复现脚本同目录supplement_bias_gt_bins.py；仅NumPy CPU统计，原冻结summary/模型/预测均未改。
C150：同输入、同网络，A=FSPL、B=FSPL+Deygout。B−A为+0.020968/+0.054451dB，两频CI均>0；共同后续prior选A。报告 audit/c150_diffraction_training/comparison_summary.json 及同目录comparison_summary.md，模型/原始预测 runs/c150_diffraction_prior/{A,B}/；确定性输入缓存 features/c150_diffraction/part0of1/，不重新导出。
C151a：同15,214,786参数，A固定共同打乱Global55物理通道，B保持对齐，Local/FiLM相同。B−A为−0.060339/−0.212833dB，两频CI均<0，支持本匹配控制中的空间对齐收益，但未超过C82/C145。B的epoch12高频FIT和monitor同时上升，保留joint-best epoch8，不能只归因过拟合。报告 audit/c151_global/comparison_summary.json 及同目录comparison_summary.md，模型/预测 runs/c151_global/{A,B}/。
C151b：共同fresh aligned Global+FSPL，不加载历史权重；同15,237,586参数，A Cartesian/B TX-polar，branch20,752参数、256×192/2层。B−A为+0.002151/+0.030773dB；CI分别[−0.000746,+0.004731]、[+0.007556,+0.052188]。低频无明确差别，高频polar更差；固定C82 tail略好、原好区更差，整体未获益。报告 audit/c151_polar/comparison_summary.json 及同目录comparison_summary.md，模型/预测 runs/c151_polar/{A,B}/。均未过双参考双频门槛，不扩预算。
C150–C151新网络均不输入C82预测或权重；历史模型比较只作性能参照，不是新增feature的单变量因果对照。monitor为反复使用的开发集，其中80地形参与过C82校准、48曾作selection，不能称盲测；CI未校正反复选择。 只读源码核查进一步确认：C145=冻结C82完整预测+Field残差，输入含(C82−legacyFSPL)/20双频context；Field从C123随机初始化训练8轮检查点加载，并非从C82网络参数恢复。C150完整保留原45c73，context置零、新增10衍射输入，换FSPL/Ld prior并从随机Field训练。前12轮数据/预算/LR/loss路径对应，但C145全程实际764轮、C150仅12轮，checkpoint搜索历史不等。故可比较成绩，不能把外部退化归因于新增feature或单一原因；仅C150内部A/B隔离Ld prior。
已知输入边界：basic11仍用旧raw-grid TX/高度定义，diffraction/FSPL与polar路由用registered定义；当前C95、C141–C144物理诊断未覆盖仅替换basic11定义的匹配网络训练。两套基准可由已知metadata转换，其并存不构成bug或因果证据，本轮未临时更改basic/FiLM。
物理/路由近似的精确定义、数值校验、source/cache SHA与限制见各protocol.json：Deygout/Bullington不是严格上下界；冻结缓存floor不存在solver反传死梯度；polar比较同时改变采样密度/插值/coverage，不能仅归为径向顺序，也不是无损或所有polar方法的结论。

后续统一使用服务器本地 SwanLab，固定 local 模式，不接云端同步。数据目录 `tracking/swanlab/`，项目 Lunar_CC；不同实验以 campaign、名称和来源配置区分。
启动：`source tracking/env.sh`，再用 `.venv/bin/python scripts/start_tracking.py`；服务仅绑定 `127.0.0.1:16007`。新代码统一通过 `src/cc_tracking.py` 记录。使用与续训说明见 `tracking/README.md`。
Mac 查看：`ssh -N -L 127.0.0.1:16007:127.0.0.1:16007 transformer`，再打开 `http://127.0.0.1:16007`。不为监控而重启训练。
迁入全部10条 MLflow 记录与 C145 原始历史，共11条、45,507个标量点、69个产物文件；逐点值/step及产物SHA256一致，未重跑训练或实验。证据 `tracking/swanlab/migration/manifest.json`、`verification.json`；源时间戳保留在快照，界面时间为导入时间。旧 MLflow 数据保留，冻结旧实验脚本不改写。
默认入口 `http://127.0.0.1:16007/` 为同一 SwanBoard 服务的 CC 研究总览扩展，仅默认绘制两张双频曲线。可切换选中/最后权重、早期/全程，并区分开发 monitor 与确认集；C147 少量采样点诊断单列。原生明细为 `/?native=1`、`/experiment/<id>/chart`，提供返回总览链接。
总览实现 `src/cc_dashboard.py`、`src/cc_dashboard.html`，服务入口 `scripts/serve_tracking.py`；只读取既有原始 JSON 与 SwanLab 本地日志，未向实验补写或重算指标。校验证据 `tracking/swanlab/dashboard/verification.json`。缺失的参照、预算、耗时与结果显示未记录，不推算；同口径 C82 双频同时改善才标为双频改善。 新C150/C151已关联原history/result/selected_export与实时progress，正确区分选中/末轮及区域指标；分析记录不混作训练，默认active或最新训练，手选跨刷新保留。当前服务PID66080，仅重启viewer，未重启训练。
语言约定：说明与结论使用中文；RMSE、SSE、Epoch/Step、dB、MHz/GHz 等专业记号、标准单位、模型名及实验编号保留原写法，首次出现时可补中文释义。默认图保持绿线为训练地形、蓝线为本次训练留出地形、灰色虚线为 C82 参照。仅改显示别名，不改原始指标键、数值或来源。
原始曲线页面默认只显示双频两张合并图：每张为同频FIT绿线、monitor蓝线及C82灰色虚线；其余原始指标隐藏折叠保留。已配置总览内12个训练实验，Swan原始scalar tag/step/值及训练config未改。配置：.venv-swanlab/bin/python scripts/configure_swanlab_view.py --paired-rmse --experiment-id <native_id>；批量当前训练记录用--paired-all-training。原版0.1.10b2的native多指标渲染兼容补丁同脚本，元数据回退快照与配置在 tracking/swanlab/view/paired_rmse_*。脚本只改显示/关联，不生成或迁写训练指标。
本地看板为官方旧版离线模式；同目录仅一个项目。该版本 resume 会新建日志分段，续训需核验模型/优化器/随机状态及真实步数，建立明确关联前段和检查点的新分段；不得声称原记录已自动续写。
C147已完成：固定8地形256诊断像素，最多1到3次反射的尾区RMSE仅改善约0.016/0.018dB，00414基本未改善；高频绝对误差仍有采样敏感性。没有新模型训练。主结果 audit/c147_sionna_reflections/comparison_direct_covered.json，报告 results/C147_Sionna反射深度对照_20261114.md；配置、验证、查看入口见同目录 protocol.json、verification.json、mlflow_runs.json。comparison_summary.json为初始搜索结果，最终解读使用direct_covered。所有推理已结束，不重复启动。后续优先处理路径覆盖与弱信号积分，再评估特征替换。

## 唯一任务入口与历史
只更新本文件作为 AI 任务交接入口。旧记录在 `/home/chenzhe/archives/ai_records/20261114/lunar-pathloss-rm-challenge/`，通过上级 `manifest.csv` 按原路径定向查找。默认不读归档或冻结交付包，旧任务和旧指令不自动生效。
必要的代码说明、校验输入、模型卡、原始材料和数据保留原位。旧调度器或报告脚本复用前，先核对已归档的文档路径。
