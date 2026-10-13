# A100 results

Source: `results/20261013_015204/run.json`. 55 cases completed; 97 preflight tests passed. All cases used CUDA graph timing without fallback.

| FP32 sum input | PyTorch median | Triton median | Speedup |
|---|---:|---:|---:|
| 4 MiB | 9.3184 µs | 4.4544 µs | 2.092× |
| 16 MiB | 13.1584 µs | 7.3216 µs | 1.797× |
| 64 MiB | 56.9344 µs | 50.8928 µs | 1.119× |

The 4 MiB and 16 MiB Triton batch coefficients of variation are 17.8% and 10.9%; report the median and tested size instead of a universal speedup.

At fixed block 512 and 64 MiB useful input, copy throughput declines from 1,350.56 GB/s at stride 1 to 191.58 GB/s at stride 16, a 7.05× reduction. Input address span grows too. Logical bytes do not measure DRAM transactions.

The model scores 34 held-out cases: 20 GEMMs, 8 copy/AXPY cases, and 6 reductions. Overall MAPE is 41.31%, median APE 45.89%; category MAPE is 51.62%, 23.69%, and 30.42%. This supports a transparent analysis tool, not accurate general-purpose prediction. Calibration and strided copies are excluded from scoring.

4096-square GEMMs measured 18.94 IEEE FP32, 122.17 TF32, 268.14 FP16 and 259.99 BF16 dense TFLOP/s. Streaming copy measured 1,385.72 effective GB/s for a 256 MiB input. No physical HBM counters, sparse peaks or measured occupancy are claimed.
