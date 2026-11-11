# C129：小集与开发训练集拟合差距
## User scope
所有工作在CC，原始data顶层只读；只用空闲授权1/3/4，UUID/run_gpu核验；不干扰其他任务。官方VAL不开发，mask仅loss/score。C95、共享固定RX残差、简单TX平移、融合调参路线保持关闭。
当前“完整集”指2815图/1984地形的开发FIT，不是全部官方TRAIN。目标仍是双频官方VAL<=1dB。

## Questions and order
A. 训练口径/梯度目标：用原始数据独立复核已选权重及末端在全部2815 FIT的真实误差，区分running loss、192图panel、加权与非加权。核对每图曝光次数、实际更新数、原采样权重与逐batch归一化引入的有效权重。
B. 冻结表征和读出优化：在固定C127 residual_independent最优骨干上，采集原C101 FIT固定1536点/map的72维原生读出前特征，求独立双频加权线性最小二乘头。不调整骨干、不增加输入、不调参选lambda。若闭式头明显降低FIT而Adam头不行，支持读出优化问题；若两者相近且FIT仍大，说明当前冻结表征的线性可读性不足，不证明原始H/TX缺信息。
C. 控制每图训练强度与容量：必要时用嵌套16/64 FIT图，同一普通MSE、相同batch、相同初始化来源、固定CPU顺序，比较相同每图曝光与相同更新两个切面。容量对照仅在同64图改变同一骨干宽度，保留深度/注意力head_dim和初始预测；先核验扩宽初始化。不得用不同数据量固定步数冒充容量结论。
D. 表征归因：结合B的冻结读出与C的端到端拟合，定位输入到特征或共享容量的问题。只在明确证据之后选择一项针对性修改，冻结原128地形monitor，明显收益后才进原484确认/扩训。

## Baseline and gates
Formal C82 officialVAL=2.246377/2.579150dB, unchanged.
Same128monitor628maps fullnative C82=2.1463905073/2.4180144849.
Both monitor gain>=.20dB then original484 confirmation each>=.30dB plus paired95%CI positive; no fullTRAIN extension otherwise.
Monitor historically reused; C82 calibration used80monitor terrains. Do not call it a new blindtest.

## Initial evidence
C127 small16batch16 repeated1700updates=1700exposures/map; main24epochs=24exposures/map but~4380parameterupdates. These are different budget axes. C127 capacity RMSE .847/.950 is aggregate, not all maps/tails below1.

## Runtime and artifacts
First pure CPU protocol/weight audit; GPU4 fullFIT/frozen-readout audit after live check. All new scripts src/c129*, scripts/c129*, audit/c129_fitgap. Existing parent sources/weights read-only.
C slices concrete budget frozen after A/B; no open-ended high-resource launch.
Run bounded smoke before slice execution, logs/checkpoints/JSON durable, progress within60s. Skill analysis-campaign applies; bash_exec/artifact tools absent, use authorized SSH and canonical CC equivalents. User CC scope overrides worktree boilerplate. No memory writes.

### A finding and normalization repair control
Original grouped batch ratios give multiTX valid-pixel gradient mass79.79/79.80%, whereas intended inclusion-weighted fullFIT mass89.22/89.34%. This is a training-estimator bias, not a scoring error and not yet the explanation of all failures. Add same-parent residual continuation on H1001/3: original batch denominator vs fixed full-denominator/epoch-batch-count. Same C127bestepoch4 parent, AdamW3e-4 wd.01,first8frozenorders,8max/min4/patience3,initial included. Ordinary MSE only; no joint shared-target branch. This controlled implementation repair is not fullTRAIN expansion and does not settle exposure/capacity by itself.

### Capacity/exposure slice frozen after readout results
Closed global72feature head changes weighted FIT sampledRMSE2.196107/2.457630 to2.194822/2.454840; fullFIT ordinaryRMSE essentially unchanged. Norm repair/control both retain their common initial best. Proceed with nested16/64FIT maps and width1/2 same-family capacity intervention, max1700exposures/map matching small-run budget,constantAdamW4e-4wd.01,batch16,ordinaryMSE with global normalization. Same64map comparison isolates width within a shared pretrained family; attention head_dim and depth fixed, function-preserving expansion checked before small jitter. Width can change optimization conditioning too, so do not call it a universal capacity proof. Monitor onlyinitial/finalFIT-selected, no monitor-based stopping. SuccessstopbothFIT<.95 after100exposures; reportper-map/tail separately. Curves expose both per-map visits andtotalupdates; no claim that these are simultaneously equal across16/64.

### Representation discriminator
After frozen global head found no remaining linear optimization margin, measure global vs per-terrain vs per-map linear heads on exactly the same72features. Fit first768 of existing1536FITpoints/map,check the disjoint768. Parent encoder already saw all nativeFITlabels: this is a readout-only split, not independent validation. Per-scene/map coefficients are label-derived diagnostic privileged information, not deployable input. Fixedstandardizedridge1e-3; no tuning. If even oracle conditional heads do not materially improve the check points, do not launch a geometry-conditioned hypernetwork; if large gain, that would motivate a separately input-only conditional decoder test. GPU4 after16mapcapacity releases, no competing launches.

### Predeclared shared-anchor/transfer check
At250exposures, narrow64FIT=1.384106/1.522479 andwide64=1.205335/1.346634; traininggain alone is not promotion. Before inspecting those models onmonitor, freeze comparison ofn16ep250,n16ep1000,n64ep250,w64ep250. All score identical16FITanchors andoriginal128monitor. This gives equal250exposures and equal1000updates cuts; cases are fixed fromsavedcheckpoints,not chosen frommonitorlabels. Capacityruns continue theirFIT-only stop unchanged.

### Conditional-readout sample-size check
The768point headcheck failed: terrain/map readouts worsen disjoint points. There are only~30large-error points/map/fold, which leaves sampling variance unresolved. Without tuning ridge or selecting cases, extend the same diagnostic once to all nativepixels:32768checkerboardpixels/headfit,32768complement/headcheck, same72features/fixedridge1e-3/global/terrain/map heads. This remains a privileged TRAINreadout diagnostic and does not certify unseen-terrain transfer. No new candidate training is authorized by oracle gain alone.
