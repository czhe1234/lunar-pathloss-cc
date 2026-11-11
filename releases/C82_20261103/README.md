# C82 月球双频路损预测

完整官方固定验证的双频 RMSE 为 2.246377 / 2.579150 dB，综合 2.412764 dB。相对 C76 改善 0.006924 / 0.017304 dB。两个频段都降低 0.30 dB 才达到大幅提分标准，最终目标是各自不高于 1 dB。

输入为一个 npz 文件，含 256×256 的 float32 高程 heightmap 和整数 tx_row、tx_col。输出为形状 2×256×256 的 npy 文件，依次为 415 MHz 和 5.775 GHz。已包括权重、源文件、已编译的 CUDA 库和编译脚本。验证集标签只在冻结后用于计分。

在 transformer 的 CC 目录，使用 ./run_gpu.sh 3 releases/C82_20261103/infer.py --input 输入.npz --output 预测.npy。完整计时和超过 500 ms 的次数见 timing.json。
