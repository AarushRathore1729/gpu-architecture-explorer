# Résumé candidates generated from this run

- Built a reproducible NVIDIA A100-SXM4-40GB characterization suite covering gemm, memory; correctness-checked 33 measured configurations using CUDA-event timing.
- Measured up to 269.8 dense bfloat16 TFLOP/s and 1380 effective streaming-copy GB/s; visualized arithmetic intensity and precision-dependent empirical rooflines.
- Implemented calibrated compute/memory runtime estimates and evaluated 27 held-out configurations; exported raw timing samples, model errors, CSV data, and a standalone interactive report.

Run status: running
Timestamp: 2026-10-13T01:42:03+00:00
Source SHA256: 43966ccf164d38097b0e5a9435c28a646262a8ff186d2f153293dc93284ea103
Model MAPE: 45.2%; median APE: 48.4%. Do not describe this as an accurate predictor without reviewing errors.

Review before use: these are measured-suite claims, not prior project history. Custom kernels, traces, and optional modes may be absent if skipped. Bandwidth is logical-byte throughput, not measured DRAM traffic.
