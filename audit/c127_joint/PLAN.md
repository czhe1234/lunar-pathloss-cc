# C127：按地形组织多 TX、双频监督的四组对照

## Objective and constraints
用户要求解决既有方法的信息组织和建模范围问题。以同一 TRAIN 留出集验证明显提分后才扩训。
仅在 CC 独立实现；父目录 data 顶层原始数据只读。只用当时空闲的实体 1/3/4 UUID，经 run_gpu.sh 核验。无官方 VAL 开发，无 mask 输入，无另一 AI 代码，无 C95 重建或融合权重搜索。

## Hypothesis
最近的点级残差训练没有充分约束同一地形多个 TX 的共同/差异响应。
输入纯净的地形分支 S(H,RX) 与 TX 条件分支 T(H,TX,RX)，通过 TRAIN 多 TX 共同均值、TX 偏离和双频差异监督，可能建立更可迁移的关系。
它是监督和模型分解假设，不宣称恢复真实物理路径。单纯成组和均值/差分的等权正交重写等价于 MSE，不被当成新方法。

## Four-arm comparison
Factors: output reference {direct H/TX physical prior, frozen C82 residual}; objective {independent pixel MSE, joint component/dual-band supervision}.
All use identical 56 H/TX inputs, C99 pretrained TX backbone, same new H-only scene branch, batch map order, sampled maps and seeds.
Direct starts from the existing C99 absolute predictor; residual starts from C82 with zero TX readout. Their initial predictions necessarily differ: within-output objective comparisons are primary; direct-vs-residual is a route comparison, not pure equal-initialization proof. No C82 map is an input channel in either arm.
In joint arm, for the 277 multi-TX FIT terrains, S is supervised by the observed TX-average of target minus the selected reference; T is supervised by the per-TX deviation. Group means are TRAIN-only supervision, never inference inputs. Different band masks are respected; S receives no mask.
A fixed additional loss supervises the predicted cross-band difference at jointly valid pixels. Independent arm has the same architecture and grouped batches but only ordinary MSE. Existing inclusion weights remain.

## Data and metric
Same C105/C101 FIT 2815 maps /1984 terrains (1707 single TX,277 four TX), monitor 628 maps /128 terrains; no new training labels outside this FIT list.
C82 full-native monitor RMSE=2.1463905073/2.4180144849 dB.
Primary: native valid-pixel per-band RMSE, independent NumPy scoring; tail defined by |Y-C82|>=5 only in evaluation.
Additional: frozen FIT-panel RMSE, TX-difference and band-difference error, tail precision/recall and amplitude/sign.
Gate: both monitor gains >=0.20 dB then original484-terrain confirmation both>=0.30 and paired95%CI positive; no expansion otherwise.
Existing monitor reused historically, not a fresh blind test. Official VAL unchanged.

## Implementation and bounded run
Files src/c127_joint.py, src/c127_train.py, scripts/c127_prepare.py, scripts/c127_score.py.
Freeze CPU batch orders. Save exact source/initial/data hashes and logs.
Smoke: initial predictor match; finite gradients; joint decomposition algebra/masks; predictions independent of other TX in batch; bounded throughput test.
Main budget: max24 epochs, min8, patience6 on actual monitor mean, same schedule per arm; no retuning auxiliary weights. Batch selected from smoke then common across arms.
Joint loss: ordinary risk +0.25*(scene component MSE+TX deviation MSE)/2 +0.25*cross-band difference MSE/2. Each normalized by25 and own valid denominator. This explicitly supervises latent components, not only a regrouped scalar MSE.
If underfitting remains, record it; do not turn a bounded negative result into input-insufficiency proof.
If any arm qualifies, complete confirmation and native verification before extending. Otherwise inspect whether component learning or transfer failed and close this branch without full refit.

## Runtime
GPU1/3 paired comparisons, GPU4 smoke/diagnostics or scoring when free; independent processes, no unrelated termination.
Use GPU cached input panels, BF16 backbone, FP32 readouts/labels/loss, original packed masks. Monitor progress at least each minute.
Skill bash_exec/artifact/memory APIs are unavailable (tool discovery empty). Use existing authorized SSH executor and CC run/log/JSON equivalents. No new worktree: explicit user CC scope and existing workspace reused. No memory writes.
Expected first substantive result after smoke/first epochs; final duration estimated after throughput.

### Smoke revision
First AMP batch-invariance max threshold failed before training. Preserve source and log; test functional independence in FP32 and separately bound AMP RMS (<0.03 dB), record both maximum and RMS. No weakening of RMSE promotion gate.

### Smoke outcome and execution
16-map smoke passed; parameters1,529,860; A800 joint step0.214s,peak10.81GiB. Initial direct AMP replay RMS0.00779dB. Batch-shape AMP RMS0.011744/max0.17524dB; FP32 RMS0.00007867/max0.0023346dB; functional decomposition oracle loss<2e-14. Paired main arms run on H1001/3, two waves, same CPU orders; GPU4 used smoke and later targeted diagnostics if needed. Main training estimated about0.5–1min/epoch per arm; 8–24epochs plus checks.

### Capacity diagnostic added after epoch7
FIT improves slightly while monitor reverses in both residual arms. To check whether the new decomposition itself is executable beyond gradients, run direct_joint on the fixed first4multi-TX FIT terrains/16maps,lr4e-4,wd.01,batch16,max3000updates,stopwhenbothaggregateFIT<0.95dB. FIT-only selection; report ordinary aggregate and tailRMSE, not a heldout result or full-data capacity proof. One GPU4 run, expected<=11min; it does not change four-arm training or promotion.
