# lunar-pathloss-cc

月球双频路损预测项目的 CC 源码。当前源码覆盖至 C156，包括 C145–C148 原有对照，以及 C149–C156 的误差拆解、绕射特征、全局输入对齐、独立基线、D4 一致性检查、轻量 ConvNeXt-UNet 和纯 CPU 分量归因核查。

输入为地形高程图、TX 位置及各阶段定义的几何与物理特征；输出为 415 MHz、5.775 GHz 的路损图，单位 dB。官方目标与当前研究停点见 [HANDOFF.md](HANDOFF.md)。开发集结果不等同于官方固定验证成绩。当前 C156 纯 CPU 核查已完成；没有重跑训练或旧实验。C155 的 FIT/monitor 差距缩小，但仍未超过 C82。C156 将高精度基本盘定位到局部/图网络修正及互补融合；共同大错包含衰减过强和不足，不能统一解释为多重深绕射。具体证据与限制见 HANDOFF。当前任务完成后停止，不自动开启训练。

## 代码入口

- [src/](src/)：模型、训练、推理、几何和物理计算，以及科学核验源码。
- [scripts/](scripts/)：数据处理、评价、看板服务与原有实验控制入口。
- [audit/](audit/)：辅助核查源码与不可变 protocol.json 算法/划分配置。
- [runs/](runs/)、[releases/](releases/)、[features/](features/)：现存阶段源码快照；不包含模型权重、运行配置快照或生成数据。
- [tracking/README.md](tracking/README.md)：服务器本地 SwanLab 的接入与查看说明。
- [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json)：公开源文件、来源说明和 SHA256 清单。

近期入口：

- C149–C152：[误差拆解](src/c149_error_audit.py)、[绕射训练](src/c150_train.py)、[全局对齐](src/c151_train.py)、[公平基线](src/c152_train.py)。
- C153–C155：[独立基线](src/c153_train.py)、[D4 一致性检查](src/c154_d4_consistency.py)、[ConvNeXt-UNet](src/c155_field.py) 与 [训练](src/c155_train.py)。
- C156：[CPU 归因](src/c156_attribution.py)、[独立核验](src/c156_verify.py)、[区域绘图](src/c156_figures.py)。

各阶段不自动重跑已有实验。

## 环境与运行资产

模型环境使用 Python 3.11、PyTorch 2.6.0+cu124；现有依赖记录保存在 [requirements.lock.txt](requirements.lock.txt) 与 [requirements.c6.lock.txt](requirements.c6.lock.txt)。看板固定使用 SwanLab 0.10.5、swanboard 0.1.10b2，安装清单见 [tracking/requirements.txt](tracking/requirements.txt) 及其 lock 文件。

此仓库是源码导出，不是附带完整运行资产的可直接复现实验包。正式原始数据、地形/TX/标签数组、冻结模型权重、预测、批量物理特征与缓存仍留在服务器。训练、推理和看板需要相应资产；从空白 clone 直接运行这些入口不会自动获得数据或权重。

代码和 HANDOFF 中保留原始相对路径及服务器路径，以维持源文件身份。某些路径指向未公开的运行资产或原始实验记录；它们不在本仓库中。历史阶段源码可能保留当时的环境接口。GPU 启动脚本针对原服务器的目录和许可 UUID，不能直接视为其他机器的设备配置。

两个 src/c70_centroid_lookup*.npz 是固定网格的索引/插值权重源查表，不是地形、标签或模型权重。三维地形展示仅保留带数据占位符的 HTML 模板，未发布填充后的地形实例。

第三方地形生成示例 notebook 保留全部单元源码；输出、执行计数和附件已清除，原始版本仍在服务器，清单记录原版与导出版 SHA256。

## 实验记录留在服务器

后续实验记录统一使用服务器本地 SwanLab，模式固定为 local。不会向云端同步。此源码快照不携带 SwanLab/MLflow 数据库、训练 history/progress、日志、PID/状态文件、指标明细、核验记录或模型产物。

旧 C144 提交保留当时已有的实验记录和诊断资产，本次按用户授权保留历史并公开；上述排除规则针对当前源码快照。

[tracking/env.sh](tracking/env.sh) 与 [src/cc_tracking.py](src/cc_tracking.py) 保留本地记录配置；[src/cc_dashboard.py](src/cc_dashboard.py)、[src/cc_dashboard.html](src/cc_dashboard.html) 和 [scripts/serve_tracking.py](scripts/serve_tracking.py) 为看板源码。查看已有记录仍需访问原服务器本地记录目录。

## 第三方源码

部分核查目录保留第三方源码的原版权、来源或 SPDX 标记。Sionna 修改版求解器对应的 Apache-2.0 许可证保存在 [audit/c95_alignment/solver_variants/LICENSE](audit/c95_alignment/solver_variants/LICENSE)。本次同步不改变这些文件的许可。
