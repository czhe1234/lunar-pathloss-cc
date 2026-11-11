# C46 residual-specific screen

C44 learns useful structured geometry but still underfits and is linearly redundant with C38. Test the different objective of correcting C38 remaining errors, rather than relearning full pathloss. Frozen teacher is the internal-FIT C38 model; no officialVAL targets are used.

Inputs: original11 channels plus2 input-only C38-FSPL prediction channels. Geometry/profile computation sees only original11 channels. Warm all shared field weights from C44 selected checkpoint, insert zero weights for2 extra channels, initialize final residual heads to zero. Verify step0 exactly reproduces C38. All target labels/masks remain outside forward. Teacher was trained on FIT so training residual estimates may be optimistic; use fixedDEV64, require a new548-terrain confirmation before promoting any gain.

Screen1000 updates with4000-step cosine horizon, LR1e-4, same map/pixel order and masks. Keep zero-step baseline as a selectable checkpoint. Require both fullDEV64 improvements>=0.10dB and no FIT degradation; otherwise stop this screen. Do not infer an input-noise floor or model-family ranking from a negative result. No automatic large training.
