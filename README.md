# lunar-pathloss-cc

月球双频路损预测项目的 CC 源码与研究记录。源码快照覆盖 C82 冻结模型及 截至 C144 的训练、评测和物理排查。

输入是一块 256×256 高程图和发射机 TX 的像素位置。输出是 415 MHz、5.775 GHz 的 256×256 路损图，单位 dB。目标是两个频段的官方固定验证 RMSE 都不高于 1 dB。

## 当前结果

冻结版本为 C82。完整官方固定验证集共 680 块地形、每频段 2330 张图。

| 频段 | 有效像素 RMSE |
| --- | ---: |
| 415 MHz | 2.246377 dB |
| 5.775 GHz | 2.579150 dB |

这些数值来自本地按官方算法计算的完整固定验证。最新 C144 检查的是 00414_00 的连续范围视线净空。结果和适用条件见对应报告。

## 文件入口

- [HANDOFF.md](HANDOFF.md) 记录当前接续点、数据范围、模型组成和已完成检查。
- [STATUS.md](STATUS.md) 保存当前状态和历史实验。
- [src/](src/) 保存模型、训练、推理、物理计算和排查源码。
- [scripts/](scripts/) 保存评价、核查和数据处理入口。
- [results/](results/) 保存研究报告和图表。
- [audit/](audit/) 保存检查代码、小型配置、指标和选定的复现输入。
- [runs/](runs/) 和 [releases/](releases/) 保存对应阶段的源码、配置与小型记录。
- [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) 列出本次上传文件的大小和 SHA256。

## 数据与运行环境

科研源目录为服务器 `/home/chenzhe/work/lunar-pathloss-rm-challenge/CC`。原始官方数据位于同级 `../data/`。代码通过各阶段固定配置定位训练权重、批量特征和缓存。这些运行资产仍由服务器的实验目录提供。

已使用 Python 3.11、PyTorch 2.6.0+cu124。当前 CPU 检查环境的 NumPy 为 2.2.6，SciPy 为 1.15.3。固定 RT 环境为 Sionna RT 1.1.0、Mitsuba 3.6.2、DrJit 1.0.3。原依赖记录见 [requirements.lock.txt](requirements.lock.txt) 和 [requirements.c6.lock.txt](requirements.c6.lock.txt)。

`run_gpu.sh` 沿用 transformer 的项目路径和允许设备 UUID。服务器上运行时应先按 [GPU_POLICY.md](GPU_POLICY.md) 核查空闲设备。其他机器需要配置自己的项目路径与设备。

## 最近的单样本检查

- [C142/C143 报告](results/C142_C143_CUDA与00414逐级核查_20261111.md) 保存固定 LLVM/CUDA 对照、TX 高度和接收导出核查。
- [C144 报告](results/C144_00414连续视线净空核查_20261111.md) 保存连续 TX 区域、两种地形剖分和三个接收单元的净空界。
- 主入口为 [src/c144_visibility_certificate.py](src/c144_visibility_certificate.py)。
- 独立核验入口为 [audit/c144_visibility/independent_check.py](audit/c144_visibility/independent_check.py)。

C144 只需要 NumPy、SciPy、官方发布数据和已保存的轻量复现输入。仓库保留了该检查用到的 C141/C142 输入及 C144 证据。主脚本保护已有输出，重新计算时应在工作副本中使用新的输出目录。

## 快照范围

仓库包含项目源码、构建脚本、配置、研究文档、小型指标和选定复现资产。完整训练、推理和旧实验复现还需要相应的官方数据、训练权重与生成缓存。第三方文件保留原有许可证或 SPDX 标记。
