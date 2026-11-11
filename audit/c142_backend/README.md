# C142 固定几何与方向的后端对照

## 当前结果

2026-11-11 已完成全部 CUDA 固定面板。22 个反例仍在双频超界。局部供体最大差 1.247926 dB，普通点 RMSE 仅改变约 0.00005 / 0.00006 dB。完整结果见 [compare.json](compare.json) 和 [decision.json](decision.json)。00414_00 的后续核查见 [C142/C143 报告](../../results/C142_C143_CUDA与00414逐级核查_20261111.md)。

固定面板包含 10 块训练地形、22 个 C141 有限反例和同图 80 个普通点。按地形累计保存了 C141 用过的 193,112,036 条固定射线的全局编号和 Float32 世界方向。另保存 5,543 条固定射线的 LLVM 轨迹，含接收面编号及顺序、交点、推进后起点和方向、首个地表交点。轨迹按供体附近和方向编号抽取，只用于配对诊断。

新 LLVM 入口只在 1819 地形重算了供体功率。7,430,870 个起始方向与原版生成方式逐位一致。有限供体的最大路损差为 9.69×10⁻⁷ dB，零功率集合一致。其余地形的供体功率沿用 C141 缓存。见 [LLVM 入口核验](llvm_smoke_verification.json) 和 [固定面板核验](llvm_panel_verification.json)。

另用 C94 已归档的 Sionna 质心供体表做缓存对照。22 个反例中，12 个的正权重供体集合相同，最弱供体在两个频段仍各超过标签允许功率 1 dB。其余 10 个各缺一个供体的缓存功率。在这个候选供体表下，12 个已知反例足以排除单靠此变更解决全部反例。80 个普通点中，41 个可比较最弱供体功率与标签上限，其中 0 个超界。此次比较未计算普通点 RMSE。见 [供体表核查](export_stencil_check.json)。

OptiX 550.135 已按 NVIDIA 发布的 SHA256 校验，并解包到 CC。禁用 GPU 时已验证两个库可加载，私有沙箱可读数据文件。GPU 上的 OptiX 场景已实际运行。启动脚本使用同驱动版本的本地库和私有挂载。见 [环境核验](optix_preflight.json) 和 [NVIDIA 550.135 组件清单](https://download.nvidia.com/XFree86/Linux-x86_64/550.135/README/installedcomponents.html)。

## 已完成的执行入口

在服务器 CC 中设置 card 为已确认空闲的实体 1、3 或 4 号卡。启动脚本会重新核查计算进程、显存和利用率，再按完整 UUID 启动。

~~~bash
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
./scripts/c142_run_cuda.sh "$card" --maps 1819
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B src/c142_compare.py
~~~

确认 1819 的 CUDA 实际运行、输入哈希、射线编号和方向一致后，用同一张卡接续固定面板。启动脚本会复用已完成地图。若该卡被占用，等待同一许可列表内的空闲卡。比较脚本区分已有结果和缺失地图。

~~~bash
./scripts/c142_run_cuda.sh "$card"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B src/c142_compare.py
~~~

两后端共享 C141 的候选场景、接收面、供体、N=4×10⁹、每批最多 50 万方向、seed=13、max_depth=0 和原版全部接收面穿越。功率仍乘每批 M/N，并按两个波长换算。主对照用 LLVM 生成的相同世界方向，隔离方向生成误差。脚本另记录 CUDA 自身方向生成与固定方向的差异。若该差异足以影响结论，再单独比较原生 CUDA 方向的功率。

先看 22 个反例在两个频段是否仍超过 1 dB，同时单列零功率和普通点的同分母误差。零功率不算解释反例。只有固定训练面板出现可解释的功率修复，且普通点准确性得到保留，才冻结修复并进入原留出集。当前已完成固定 CUDA 诊断，尚无通过相容性的新模型。
