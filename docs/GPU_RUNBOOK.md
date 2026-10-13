# GPU runbook

1. Open the supplied self-contained notebook and select A100 in Colab.
2. Keep `PRESET="full"`, `TIMING="cuda_graph"`, and `PROFILE=True`.
3. Run all cells. The notebook first runs correctness tests, then measurements.
4. If correctness tests fail, benchmarks are skipped and a diagnostic bundle is still produced.
5. Send the exported result ZIP for review. It contains source and all saved logs.

The exported run folder is `results/<UTC timestamp>/`. `run.json` is the measurement file. `trace.json` is the optional profiler trace; do not import the trace into the measurement viewer.

## If something fails

- **No GPU:** change the Colab runtime to A100, then rerun setup. The suite refuses to substitute CPU measurements.
- **Missing Triton:** setup uses the exact Triton dependency declared by the installed PyTorch distribution, when available. Do not upgrade PyTorch for this project under the deadline.
- **Graph capture failure:** the case falls back to eager timing and records the exception. Rerun the complete suite with `TIMING="eager"` for consistent calibration/evaluation.
- **GPU test failure:** share the exported ZIP; its test log has the failing configuration. Do not use its missing benchmark values in a résumé.
- **Benchmark failure:** successful cases remain in checkpointed JSON, while failed cases keep errors. Share the ZIP for a focused fix.
- **Profiler unavailable:** measurements are unaffected. Report the profiling step as unavailable.

## Optional Nsight follow-up

On a machine with Nsight Compute and performance-counter access:

```bash
ncu --set basic --target-processes all --launch-count 10 \
  python -m benchmarks.run --preset quick --timing eager --rounds 1 --iterations 1 \
  --output results/nsight/run.json
```

Nsight adds overhead, so this is a diagnostic run, not the résumé latency dataset. Inspect available sections with `ncu --list-sections` before selecting detailed memory or roofline sections. Colab may restrict counters; this is optional.
