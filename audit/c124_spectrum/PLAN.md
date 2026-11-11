# C124: candidate delay and arrival-angle power distribution
2026-11-08. Independent work inside CC. Existing C82 formal model stays unchanged.
Question: do summed one-bounce powers discard path composition that predicts the signed continuous residual on held-out terrain?
Inputs: published TRAIN height and TX only for path rendering; existing C101 169 predictors from CC. No target or mask read by renderer. Mask only filters loss/scoring.
Representation: 8192 Fibonacci receiver rays, unchanged C69 source height and C70 three-donor mapping; 8 log2(1+excess_length_m) bins x 16 relative-azimuth bins x 2 bands. Excess length uses actual ray hit, source emission uses unchanged C67 face approximation. This is candidate scattered power, not measured PDP/PAS and not guaranteed official propagation truth.
Control: uniform redistribution over the 128 bins within each band, preserving total power. Both 427-column inputs (old169 + explicit totals2 + spectrum256); same depth10 CatBoost residual regressor, seed124041, max3000 trees, lr.05, l2=10, border254, early stop200. Includes old169 in both arms.
Fixed data: C101 FIT 2815 maps / 1984 terrains / 1536 points per map; monitor628 maps /128 terrains /2048 points. Sampled valid-pixel RMSE is screening; full native scoring required before claim.
Gate: both monitor bands improve >=0.20 dB against C82, then confirmation484 terrains. Require both >=0.30 dB with positive paired-terrain 95% CI before expanded training. If signal is absent, stop this branch; no fullTRAIN / official VAL.
Before fitting: separate CUDA library; check binned sum against unchanged 8192-ray diffuse integration on fixed training maps; finite features; exact point/map alignment; no labels in generation; freeze source hashes.
Resource: use only idle authorized UUIDs via run_gpu.sh. Build features with bounded GPU jobs; train two arms on 1/3, other authorized card can do validation. Do not interfere with others.

Post-training diagnosis: both arms failed the monitor gate. Infer only (no fit/selection) the same 24-map C121 power-probe panel frozen before C122, to measure signed correction and power contrast at fixed opposite-error endpoints. This is selected failure analysis, not an unbiased performance estimate.

Terminology: the regression label is pathloss in dB. A positive label residual means pathloss was underestimated (received power was overestimated). Candidate spectrum bins are linear scattered powers; they are inputs, not the regression labels.
