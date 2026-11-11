# C43 architecture experiment

User authorization: try high-resolution multiscale terrain encoding and path-guided sparse attention. All work in CC; GPUs 1/3/4 by UUID only. Official VAL remains outside fitting and selection. Existing C38 weights and prediction files stay read-only.

A/B/C: CNN only / uniform-reference sparse attention / path-reference sparse attention. B/C have identical parameterization and initial state. The intervention is reference-coordinate selection. All variants share data order, supervision points, optimizer schedule and checkpoint rule; hardware differences recorded. This is an independently implemented HRNet/Deformable-Attention-inspired design, not a reproduced published whole model or a validated propagation solver.

Freeze terrain-stratified nested FIT64 and FIT256 plus disjoint DEV64, all from official TRAIN. Preserve every TX map of selected terrains. Pilot tests check pixel coordinates, finite gradients, nonzero offset/backbone gradients, spatial-feature sensitivity and actual throughput. Choose one batch from measured common candidates. Stage FIT64: 64000 map presentations. Continue each LAST model on FIT256 for 128000 additional map presentations, reset optimizer. Evaluate full-pixel DEV64 and fixed FIT panel, independently score saved predictions. The matched C38 reference is its internal-FIT model, not the full-TRAIN model that has seen these DEV labels.

Promotion and stopping rules are in protocol.json. Positive pilot results do not complete the task. If there is no promotion, diagnose train fitting, scaling and feature use before spending a full-FIT budget. Main claim and full validation remain pending actual results.

The optional deepscientist experiment skill was inspected but its required bash_exec/artifact runtime is unavailable in this session. Execution uses the existing authorized CC SSH runner and durable JSON/source/log records.

Precision revision after measured interface discrepancy: final candidate inference uses FP32 with TF32 disabled. At same single input, independent and training-side interfaces agree exactly. Warm H100 measured latency was0.0878s for BOTH maps. All original BF16 results are retained; final full DEV replay uses canonical FP32. Training/selection stays unchanged for all groups.
