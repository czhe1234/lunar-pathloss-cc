# C10 服务器复现包

完整官方固定验证：415 MHz 2.967202 dB；5.8 GHz 3.387979 dB。
双频均≤1 dB目标尚未达到。

输入：256×256 float32 高度图，单位米，每图最小高程为0；TX为整数行列像素。
保持官方高度小数，不要整数化或按高差缩放。
输出：(2,256,256) float32，顺序415 MHz、5.775 GHz，单位正路径损耗dB。
不需要标签、掩码、索引表或特征缓存。

在服务器 CC 目录执行：
~~~bash
./run_gpu3.sh releases/C10_20261101/scripts/c10_predict_one.py --model-directory models --heightmap /home/chenzhe/work/lunar-pathloss-rm-challenge/data/val_hm.npy --hm-row 0 --tx-row 205 --tx-col 148 --output example_prediction.npy
~~~
输出位于本包目录。程序拒绝覆盖已有文件。
示例使用 val_index.csv 第一个样本的原始高度图与TX，未读取其标签。
Python接口：src.c10_inference.C10Predictor(model_directory).predict(heightmap,tx_row,tx_col)。

沿用 CC/run_gpu3.sh，以完整UUID限定实体3并复核身份。显存分配上限为总显存15%。
使用CC独立虚拟环境，依赖版本见requirements.lock.txt。
这是服务器复现包，未提交Kaggle或制作最终Notebook。

本机预热后完整双频单图：平均313.8ms，
p95 400.0ms；首次调用1373.7ms。
包含当场特征计算和输出。官方硬件与计时规则仍需正式核对。
