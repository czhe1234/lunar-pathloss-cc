# lunar-pathloss-cc

月球双频路损预测项目的 CC 源码。本次同步包含截至 C157 的审查工具和 2026-11-17 源码整理，保留当前流水线与历史路径。输入为地形高程图、TX 位置及派生几何/物理特征，输出为 415 MHz 与 5.775 GHz 路损图。

研究状态、评价口径及当前入口见 [HANDOFF.md](HANDOFF.md)。开发集成绩不能视为官方盲测成绩。现有训练与诊断已结束，不自动重跑。

## 当前入口

- [src/](src/) 保存当前模型、训练、推理和几何/物理实现。
- C82 冻结链使用 [src/c82_inference.py](src/c82_inference.py)、[src/c82_predict.py](src/c82_predict.py)。
- C145、C153、C155 的已保存成绩由各自的 report/verify 源码读取服务器资产。
- C156 归因源码见 [src/c156_attribution.py](src/c156_attribution.py)。
- C157 审查源码见 [audit/c157_pstack_review/](audit/c157_pstack_review/)。
- 本地 SwanLab 配置与查看说明见 [tracking/README.md](tracking/README.md)。

## 归档与旧路径

414 份历史源码正文位于 [archive/20261117_cc_cleanup/](archive/20261117_cc_cleanup/)，包括 407 份快照、5 个一次性控制/预热脚本、2 个已结束模型定义。旧位置以 Git mode 120000 保留相对软链，每个链接目标都包含在当前提交中。当前流水线、使用自身位置定位资源的代码及冻结发布包保持原路径。

[SOURCE_ARCHIVE.json](SOURCE_ARCHIVE.json) 列出旧路径、归档路径、相对链接文本和源码 SHA256。[SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) 区分链接文本的 SHA256 与跟随链接后源码的 SHA256。此映射不包含服务器 inode、原始实验记录或训练指标。

先激活 Python 3.11 或更新的环境，并用 python3 --version 确认版本。再从 clone 根目录执行以下命令，只读取源码并核对路径、哈希和 Python 语法，不导入项目模块、不训练、不加载权重或数据：

~~~sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/verify_source_snapshot.py
~~~

非零退出表示失败。成功输出必须包含 status=pass 与 archive_links=414。验收器拒绝 -O、-OO 或非零 PYTHONOPTIMIZE；较旧的 Python 版本也会明确报错退出。checkout 必须保留真实软链；将软链文本当普通 Python 文件的导出不满足此检查。若采用解链复制，需相应调整验证格式，不能冒称保留了原链接布局。归档目录不作为旧控制器的启动工作目录。

## 环境与运行资产

Python 3.11、PyTorch 2.6.0+cu124 的依赖记录见 [requirements.lock.txt](requirements.lock.txt) 与 [requirements.c6.lock.txt](requirements.c6.lock.txt)。SwanLab 0.10.5 与 swanboard 0.1.10b2 的依赖见 [tracking/requirements.txt](tracking/requirements.txt) 及 lock 文件。实验记录固定留在服务器，使用 local 模式。

本仓库是源码导出。原始地形、TX/标签数组、有效像素数据、模型权重、拟合系数、预测、批量物理特征、训练日志和状态、SwanLab/MLflow 数据库、审查/整理原始记录均不随本次提交上传。算法 protocol.json 沿用既有公开范围，不新增 C157/本轮整理的原始审查 JSON。已公开的旧 C144 提交历史按此前授权保留。

源码依赖齐全不等于空白 clone 可以复现实验。模型推理、训练、报告和看板仍需服务器上的运行资产及原环境。C157 的审查工具和 audit/cc_cleanup_20261117 下的管理源码依赖服务器原始审查记录；其中部分管理命令会写核验记录，不能替代上面的只读公开源码检查器。服务器管理源码保持原字节，未把公开简表伪装成其原始输入。

原生库对应的 .cpp/.cu 源码已收录，编译后的 .so 留服务器。本次仅检查这些源依赖存在，未重新编译或验证构建命令。两个 src/c70_centroid_lookup*.npz 是固定网格索引和插值权重源查表，不是标签、预测或模型权重；文本样式 [results/c146_live/academic.mplstyle](results/c146_live/academic.mplstyle) 已补齐。三维展示只发布带数据占位符的 HTML 模板，不发布填充地形的实例。

## 许可与来源

[SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) 给出源文件与生成文档清单。第三方文件保留原版权和来源；Sionna 求解器的 Apache-2.0 许可证见 [audit/c95_alignment/solver_variants/LICENSE](audit/c95_alignment/solver_variants/LICENSE)。此次同步不改变其许可。
