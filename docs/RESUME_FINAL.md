# GPU Architecture Explorer

- Built PyTorch and Triton microbenchmarks for GEMM, memory access and reductions; validated 55 configurations on NVIDIA A100 across FP32, TF32, FP16 and BF16.
- Implemented a two-stage Triton FP32 reduction, cutting median latency from 9.32 to 4.45 µs for a 4 MiB input, a 2.09× speedup over PyTorch.
- Characterized GPU workloads with CUDA-event timing, empirical roofline models and profiler traces; measured 268 TFLOP/s FP16 GEMM and 1.39 TB/s effective streaming-copy bandwidth.
