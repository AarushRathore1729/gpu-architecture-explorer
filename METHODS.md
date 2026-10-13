# Measurement and modeling methods

## Scope

This project characterizes workloads on one CUDA GPU. It does not simulate GPU pipelines, estimate physical occupancy, measure DRAM counters, implement a custom GEMM, or claim cross-architecture validation without actual runs from multiple devices.

## Timing

Inputs and outputs are allocated before measurement. Compilation and warm-up happen before CUDA events. The default captures a batch of 20 operations in a CUDA graph, then times seven graph replays with CUDA events. Each sample is the batch duration divided by 20. Eager mode records events around a Python-submitted batch instead.

The reported median, P10, and P90 summarize **batch averages per operation**. They are not single-request tail latencies. Raw batch samples and coefficient of variation are preserved. Graph capture exceptions trigger a documented eager fallback; cases whose timing methods differ from their compute or copy calibration are excluded from prediction scoring. A fill floor with a different timing method is omitted.

There is no per-iteration cache flush. The same buffers are reused, which approximates a steady workload but permits cache residency. The dedicated copy calibration uses 256 MiB input plus 256 MiB output in the full preset. Small-buffer cases are presented separately and may exceed its effective throughput because of cache effects. Neither is a physical DRAM counter measurement.

Device metadata includes compute capability, SM count, VRAM, software versions, and available `nvidia-smi` snapshots before and after the run. GPU sharing, dynamic clocks, power limits, and thermal changes can affect results. The suite does not lock clocks or guarantee exclusive access.

## Work and traffic conventions

- Dense GEMM counts `2 × M × N × K` FLOPs and `element_bytes × (M×K + K×N + M×N)` ideal traffic, with beta zero.
- Copy counts one useful input read and one output write.
- AXPY computes `X + 3Y`, counting two FLOPs, two input reads, and one output write per element. Inputs and output do not alias.
- Strided copy counts useful input reads and output writes. Its input address span is recorded separately; span is not actual DRAM traffic.
- Sum counts `N−1` useful additions. Triton traffic adds writes and reads of the explicit partial buffer. PyTorch internal temporary traffic is unknown and omitted.
- Decimal units are used: `GB/s = logical_bytes / (latency_ms × 1e6)` and `TFLOP/s = FLOPs / (latency_ms × 1e9)`.

FP32 IEEE and TF32 are distinct modes. FP16 and BF16 have separate empirical ceilings. TF32 uses float32 operands; it is never presented as IEEE FP32 throughput. Dense results are never compared to a sparse marketing peak.

## Correctness

Every successful measurement has a correctness record. Contiguous and strided copies check the full output for exact equality. AXPY checks 8,192 evenly spaced output positions. GEMM checks a 16×16 region using the same quantized operands converted to IEEE FP32; tolerance and maximum error are recorded. Reductions compare their scalar result with a full-input FP64 sum on positive inputs.

The GPU tests additionally check full outputs at odd lengths and masked boundaries. A successful benchmark is evidence for its checked configuration and precision, not proof for all possible inputs.

## Model calibration and evaluation

The full preset calibrates compute with a 4096-square GEMM per supported precision. Held-out evaluation uses squares from 256 through 2048 and a skinny matrix shape. Copy calibration uses a 256 MiB input; evaluation copy/AXPY sizes are 4, 16, 64, and 128 MiB. The measured one-element fill supplies a heuristic per-kernel latency floor.

Predicted runtime is the maximum of compute time, ideal logical-byte memory time, and a per-launch floor. This is a calibrated estimate, not a strict hardware lower bound. The roofline can be exceeded if a measured calibration did not achieve the device's true peak or if cache reuse differs.

Calibration cases are never included in prediction error summaries. Strided copies are excluded because useful-byte counts omit traffic amplification. The evaluated held-out sizes share the GPU session and software with calibration; this is not cross-session, cross-device, or statistical population generalization.

The report preserves absolute percentage error per case, mean APE (MAPE), median APE, and per-category MAPE. Low error should only be claimed after reviewing these outputs and measurement variability. Comparing A100 to L4 requires both real result files and should account for clocks and software differences.

## Optional profiling

PyTorch Profiler records separate CPU/CUDA traces after timing. Trace collection failures are recorded and do not alter measured benchmark numbers. Nsight Compute commands are in the runbook; successful collection requires a suitable environment and permissions. No occupancy or DRAM-counter result is claimed unless such a collection was actually performed.
