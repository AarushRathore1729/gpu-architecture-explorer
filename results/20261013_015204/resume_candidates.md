# Résumé candidates generated from this run

- Built a reproducible NVIDIA A100-SXM4-40GB characterization suite covering coalescing, gemm, memory, reduction; correctness-checked 55 measured configurations using CUDA-event timing.
- Measured up to 268.1 dense float16 TFLOP/s and 1386 effective streaming-copy GB/s; visualized arithmetic intensity and precision-dependent empirical rooflines.
- Implemented calibrated compute/memory runtime estimates and evaluated 34 held-out configurations; exported raw timing samples, model errors, CSV data, and a standalone interactive report.

Run status: completed
Timestamp: 2026-10-13T01:52:18+00:00
Source SHA256: 8696ec40c9811b18a538359ae5b894f32266da24b940e872cad900b4eeb52d09
Model MAPE: 41.3%; median APE: 45.9%. Do not describe this as an accurate predictor without reviewing errors.

Review before use: these are measured-suite claims, not prior project history. Custom kernels, traces, and optional modes may be absent if skipped. Bandwidth is logical-byte throughput, not measured DRAM traffic.
