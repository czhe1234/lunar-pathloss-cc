# C76 月球双频路损预测

官方固定验证覆盖 680 块地形，每频段 2,330 张图。415 MHz 为 2.253301 dB，5.775 GHz 为 2.596454 dB。双频综合 RMSE 为 2.424877 dB。

模型选择与权重清单见 freeze.json。正式输出顺序是 415 MHz、5.775 GHz。输入只需 256×256 米制高度图和整数发射机行列坐标。验证标签和掩码仅供外部评价。

在 transformer 的 CC 目录中运行以下命令。输入 NPZ 含 heightmap、tx_row、tx_col。输出 NPY 是两个 256×256 的 dB 路损图。

    ./run_gpu.sh 3 releases/C76_20261103/infer.py --input input.npz --output prediction.npy

包内含用于已授权 A800 和 H100 的二进制与源码。首次运行包括模型初始化与 PyTorch 编译。build_cuda.sh 可在对应 Linux CUDA 环境中重编译。正式平台部署时还需按平台配置设备。

评分使用官方原始脚本与独立 float64 复算。官方固定验证已用于本次版本对比。隐藏测试和网站分数另行产生。
