# Résumé positioning

Project title: **GPU Architecture Explorer — Workload Characterization & Performance Modeling**

Use measured figures only after the CUDA run. The exporter writes `resume_candidates.md` from successful cases in that run. Pick two compact bullets after reviewing the raw measurements, test logs, skipped work, and prediction errors.

Accurate topics this implementation supports after verification:

- PyTorch/Triton GPU microbenchmarks for dense GEMM, useful-byte memory throughput, access stride, and reductions.
- CUDA-event timing, correctness checks, source fingerprints, metadata, and reproducible JSON/CSV exports.
- Precision-specific empirical rooflines and calibrated runtime estimates evaluated on held-out configurations.
- A self-contained interactive report with searchable measurements and model comparisons.

Do not describe it as a cycle-accurate GPU simulator, hardware-counter HBM profiler, measured occupancy study, custom GEMM implementation, or multi-GPU comparison unless those capabilities are separately implemented and verified. Avoid promising prediction accuracy solely because a model exists.

The code is a forward-looking portfolio implementation. Dates and résumé claims should reflect the work actually completed.
