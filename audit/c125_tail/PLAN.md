# C125: resolve the four-percent tail failure
Date: 2026-11-08. Parent: C124. Work remains inside CC.
## Objective
Separate (A) misleading interpretation of the all-point MSE sign score, (B) optimization/capacity failure, and (C) inadequate learned representation of H/TX. Success is a deployable reduction of both held-out RMSEs, not oracle-tail accuracy alone.
## Boundary and comparability
Freeze C101 FIT (1984 terrains,2815 maps,1536 points/map), monitor (128 terrains,628 maps,2048 points/map), C82 predictions and masks. Tail means valid and abs(Y-C82)>=5 dB, as in the prior 4% claim. No official VAL access for selection. Never use true tail membership at deployment. Scores using truth are explicitly diagnostic. No C95 coordinate-reconstruction route, no other-AI sources.
## Slice plan
1. Recompute C124 sign confusion, weighted sign quality and correction amplitude; compare same-point C120 conditional sign models and majority baseline. Counterfactual true sign/magnitude/membership locate the bottleneck; these are nondeployable diagnostics.
2. Train a tail-only conditional continuous residual regressor on old169 features, with a strong tree learner. Compare fit vs monitor and actual all-point C124. This isolates all-point conditional averaging from learnability on tails. Reuse C120 sign models rather than retraining them.
3. Based on slice 1/2, compare old representation with raw H/TX spatial views under the same tail objective and fixed learner; verify with a different learner only if required to distinguish capacity from representation. Raw views must use H/TX only. Freeze exact inputs, budget, seed and gates before launch.
4. If conditional prediction meaningfully improves, build/evaluate deployable gating on all fixed monitor points. Only a real all-valid-pixel gain may enter confirmation or expanded training.
## Assets and dependencies
C101 point cache; C120 old/native sign models and predictions; C124 spectrum/total models and predictions. All local CC artifacts. Dedicated bash_exec / artifact campaign tools are unavailable; use authorized SSH, isolated CC scripts/logs and JSON records. This is a competition diagnostic, not a paper-writing campaign.
## Execution
Check idle allowed UUIDs immediately before each GPU launch. Only run_gpu.sh 1/3/4. Bounded checks before heavy runs; inspect losses, fit and holdout metrics. Report meaningful results within 60 seconds during active work. No external task changes.
## Reporting and gates
Report all negative results and the distinction between actual prediction and truth-assisted diagnosis. Monitor gate for expansion: >=0.20 dB each, followed by same484 confirmation >=0.30 dB each with positive paired-terrain 95% CI, then full-native scoring.
A failed model does not prove missing information in raw inputs. A high fit score alone does not prove transfer.
## Checklist
See CHECKLIST.md.
## Revision log
Initial: first resolve the mismatch between C124 55% all-point regression sign and C120 64-70% conditional sign classification before choosing the intervention.

Slice2 frozen: old169 versus old169+C118 input-only native latent96 (265 total), identical CatBoost conditional residual regression depth10,max3000,lr.05,l2=10,border254,seed125041,early-stop200 on conditional monitor RMSE. Fit inclusion weights unchanged. Save fit and monitor predictions, conditional errors and truth-gated all-point diagnostic. This reuses C120 inputs but changes classification to continuous conditional regression; no retraining of C120.

Slice3 inputs frozen before training: published-raster receiver 33x33 and TX17x17 patches, each raw relative height /5 and departure from the centre tangent plane /.1; 129-point TX-to-RX line with nine perpendicular offsets [-16,-8,-4,-2,0,2,4,8,16] grid cells, sampled bilinearly, height relative to Htx+3 to Hrx+1 line /5. All stored float32, no clipping, no target/mask read by extractor. This is an input representation, not a claim to repair physical TX/receiver coordinates.
Slice3 controls: identical architecture and initialization with old169 numeric input, (a) zero additional views, (b) correct raw views, (c) fixed independent cross-terrain shuffled views. Same conditional tail MSE + 0.1 sign BCE; shared fit data, seeds and budget. Inspect fit vs monitor to distinguish optimization from transfer. A conditional improvement alone never authorizes final deployment; subsequent all-point gate required.

Neural slice frozen after throughput check: 1,907,908 parameters; batch2048 (61.5k points/s on H100, batch4096 only 2.1% faster with 41% higher peak memory), max4000 steps, AdamW lr3e-4 wd.01, 100-step warmup then cosine to10%, evaluation every250; minimum1500 and stop after1000 steps without conditional monitor improvement. All three use exact saved initial SHA, samples and budgets. Additional raw branch includes native patch convolutions and a TX-RX strip attention encoder. First projections use float32, later layers BF16; no clipping of stored raw views.
Interpretation gate: compare raw against both old and shuffled and against the existing native265 tree conditional benchmark. A real-gate deployment check must follow any worthwhile conditional gain. Neither truth-assisted scores nor a feature classifier alone can promote full training.

Membership probe added after slice2: reuse frozen C118 location probabilities p_tail=p_negative+p_positive; combine with C125 old169 conditional mean and FIT-only constant non-tail mean via total expectation. Evaluate all valid monitor points, no blend/threshold search. Compare replacing p_tail by true membership as a labelled diagnostic. This measures the selection bottleneck without training a new gate.

Sampling correction: the first neural set used GPU RNG. A800 produced different evaluation/sample index sequences despite equal seeds (FIT panel denominators differed). Preserve those runs as preliminary; do not use shuffled-vs-raw as an exact paired control. Repeat all three with the same precomputed NumPy index array (4000x2048) and fixed 65536-point FIT panel saved in sampling.npz. Initialization and all hyperparameters unchanged; use *_fixed output directories.

Capacity diagnosis frozen: old169 inputs only, full fixed FIT tails, depth12 CatBoost with 3000 trees,lr.15,l2=1,no bootstrap,no early stopping. Monitor is logged but never selects the checkpoint; use all3000. Purpose: distinguish inability to fit from inability to transfer, not promote an overfit model. Report unweighted and original inclusion-weighted FIT RMSE plus monitor RMSE. Run only once per band.

Targeted implementation C126 added after the controlled failures: train the C123 full-native global terrain operator with explicit ordinary/positive-tail/negative-tail state and conditional-amplitude supervision, versus an identical state-head architecture optimized only for total risk. This directly tests full-field recognition after frozen two-stage gating failed. Detailed frozen protocol is audit/c126_statefield/PLAN.md. Main result is deployable full-native monitor RMSE; diagnostic oracle state scores are separate. Same FIT/monitor, no expansion.
