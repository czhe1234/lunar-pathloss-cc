# C116: identify a repairable transfer failure before a new version
Scope: independent CC, raw published TRAIN only, GPU UUIDs 1/3/4 only, no C95 restart, no official VAL.
Baseline C82 read-only. Original128 monitor and484 confirmation retained.
Completed diagnostics: seen/unseen C72 map residual distributions are close; no large marginal train/deployment residual shift. Local radius4 oracle still leaves7.24/7.71dB weak-point error, and73.0/71.5percent of true weak powers fall below all nearby C82 candidate powers by>1dB. Reject pure relocation as a sufficient repair.
C116 actionable test: learn hard conditional attenuation boundaries from the same fullprecision169 H/TX/C82 inputs used by C101precise. GPU depth10 CatBoost residual readout, one perband. Earlier C33 learned FSPL residual onold294 descriptors; it did not test C82 remaining errors on these inputs. This is a bounded algorithm-family check, not a single-variable capacity comparison or guaranteed improvement.
Same4323840 FIT points, map inclusion weights and same128monitor. Zero residual atiteration0; 3000trees maximum, patience200, learningrate.05, L2=10,254borders. Freeze perband bestiterations onmonitor before any484confirmation. No tail reweighting or blending.
C117 candidate if threshold readout is insufficient: separate terrain encoding from TX-conditioned decoding, directly testing whether joint early conditioning encourages terrain/TX memorization. Exact hypothesis and controls must be frozen before training.
Promotion: both gains>=0.30dB on same484 pixels and positive terrain-paired95CI; native verification before fullTRAIN expansion.
Resources: avoid GPU1 externalPID590322. UseGPU3/4 for bands/controls. CC launcher, JSONlogs; skill bash_exec/Canvas unavailable.
