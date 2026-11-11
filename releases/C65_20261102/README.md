# C65 服务器复现包

官方固定VAL：415MHz 2.5434139064dB；5.775GHz 2.9327636194dB；综合2.7380887629dB。
相比C38改善0.129304/0.147774dB。
目标双频均≤1dB尚未达到；这是小幅改进版本。

模型：冻结全TRAIN C38周期数值嵌入/8成员BatchEnsemble MLP基底；
低频用有序地形剖面注意力残差，高频用定向地表功率输运注意力残差。
按频段固定组合；不读取标签、有效掩码、训练特征缓存或数据索引进行预测。
C65两个残差专家都以全部6120块TRAIN地形、25720图再拟合固定17770步。
原内部DEV已作为TRAIN，官方VAL未用于梯度、选步数或路由。
官方VAL只在权重和路由冻结后计分。没有网站提交。

本包用于transformer服务器复现，保留用户1/3/4号GPU完整UUID限制。
它不是隐藏测试集最终平台适配包。
从CC运行：
./run_gpu.sh 3 releases/C65_20261102/infer.py --input input.npz --output prediction.npy

input.npz含heightmap(float32,256x256米)、tx_row、tx_col(从0开始的整数像素)。
输出prediction.npy为[2,256,256]的路径损耗dB，频段顺序415MHz、5.775GHz。
运行依赖见requirements.lock.txt。第一次调用包括编译，不能把首次编译耗时当作预热后延迟。
原始模型文件及SHA256见model/manifest.json。数值、延迟及官方评分证据在CC/audit/c65_full_refit_20261102。
