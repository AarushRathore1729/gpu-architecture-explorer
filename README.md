# GPU Architecture Explorer

[Live interactive demo](https://aarushrathore1729.github.io/gpu-architecture-explorer/) · [Verified A100 results](results/20261013_015204/run.json)

A reproducible single-GPU characterization suite connecting **dense GEMM, memory streaming, access stride, and reductions** with empirical rooflines and transparent runtime estimates.

**Verified on NVIDIA A100-SXM4-40GB:** all **97 tests** passed (50 CPU + 47 GPU), and all **55 benchmark configurations** completed with correctness checks. PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0. Raw samples, test logs, source fingerprint, and profiler trace are in [`results/20261013_015204/`](results/20261013_015204/).

| Measurement | Result |
|---|---:|
| IEEE FP32 dense GEMM, 4096 square | 18.94 TFLOP/s |
| TF32 dense GEMM, 4096 square | 122.17 TFLOP/s |
| FP16 dense GEMM, 4096 square | 268.14 TFLOP/s |
| BF16 dense GEMM, 4096 square | 259.99 TFLOP/s |
| Streaming copy, 256 MiB input | 1,385.72 effective GB/s |
| Triton FP32 sum versus PyTorch, 4 MiB | 9.32 → 4.45 µs; 2.09× speedup |
| Held-out model evaluation | 34 cases; 41.31% overall MAPE |

[Measured interactive report](results/20261013_015204/report.html). The reduction speedup is a median from this run and tested size. Variability and model limitations are preserved in the raw evidence.

## Fastest path: Colab

1. Open `notebooks/GPU_Architecture_Explorer_Colab.ipynb` in Colab (File → Upload notebook).
2. Select **Runtime → Change runtime type → A100 GPU**. The notebook embeds the entire source tree; no source ZIP upload or GitHub clone is required.
3. Run all cells. Keep `PRESET = "full"` for the complete suite.
4. Download `gpu_architecture_explorer_results.zip`. It contains the source tree, correctness-test logs, measurements, model comparisons, CSV, interactive HTML, and an optional profiler trace.
5. Open the generated `report.html` in your browser. Read the generated résumé candidates only after checking the run status and measured results.

On an A100 with all four precision modes and Triton available, the full suite attempts **55 configurations**: 6 calibration cases and 49 evaluation cases. The model scores 34 held-out cases, excluding the 15 strided copies. These counts were verified in the saved full A100 run.

## What it measures

| Family | Cases | Evidence |
|---|---|---|
| GEMM | IEEE FP32, TF32, FP16, BF16; square and skinny shapes | Dense TFLOP/s, minimum logical traffic, sampled FP32 correctness reference |
| Memory | Contiguous copy and AXPY across buffer sizes | Effective useful-byte GB/s; preallocated outputs |
| Strided access | Triton copy; strides 1–16; blocks 128, 256, 512 | Full-output equality, footprint, input address span |
| Reduction | PyTorch and two-stage Triton sum | FP64-reference scalar check, latency, explicit partial-buffer traffic |
| Timing | CUDA events; batched graph replay or eager launches | Raw samples, median, batch P10/P90, variability, recorded graph fallbacks |
| Modeling | Separate calibration GEMMs + large-buffer copy + fill floor | Precision-specific roofline, held-out predictions, category error summaries |

## Run locally or on Lightning AI

Use a CUDA-enabled PyTorch installation. Triton should match the version required by that installation. Colab normally supplies both already; `requirements.txt` deliberately does not replace them.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m benchmarks.run --preset quick --output results/quick/run.json
python -m benchmarks.run --preset full --profile --output results/full/run.json
```

For environments where graph capture is unavailable:

```bash
python -m benchmarks.run --preset full --timing eager --output results/eager/run.json
```

For explicit PyTorch-only characterization, use `--skip-triton`. The result records skipped custom kernels and cannot support Triton performance claims.

The benchmark checkpoints JSON after every case. A failed case keeps its error instead of a fabricated latency. The CLI exits with code 2 when cases failed. Missing CUDA produces a clear error, without a CPU substitute.

## Inspect the results

Open `results/explorer.html` for the measured interactive viewer, then load a measurement JSON. It is self-contained, works offline, and opens the verified A100 run by default.

Every completed run writes:

```text
run.json               GPU/software metadata, correctness checks, raw samples
measurements.csv       Flattened measurements and model errors
summary.json           Calibration parameters and held-out error summaries
report.html            Interactive roofline, memory, prediction, and data views
resume_candidates.md   Run-grounded candidate bullets and their limitations
trace.json             Optional PyTorch CPU/CUDA profiler trace
```

Regenerate a report from an existing measurement file without a GPU:

```bash
python -m analysis.report results/full/run.json --output results/full/report.html
python -m analysis.compare results/a100/run.json results/l4/run.json
```

An optional Streamlit wrapper is included:

```bash
python -m pip install -e '.[dashboard]'
python -m streamlit run dashboard/app.py
```

Upload `run.json` to the app, or place it at `results/run.json`. The portable HTML report provides the same interactive measurement exploration without hosting dependencies.

## Model and interpretation

For a case with useful work `F` and minimum logical traffic `B`, the model estimates:

```text
compute_ms = F / (P_TFLOPS × 1e9)
memory_ms  = B / (W_GBPS × 1e6)
estimate   = max(compute_ms, memory_ms, launches × fill_floor_ms)
```

`P` comes from a dedicated large GEMM of the matching precision; `W` comes from the large-buffer copy calibration. Those calibration cases are excluded from prediction scoring. Different shapes and sizes from the same session form the evaluation set.

The model is an **empirical useful-byte estimate**, not a guaranteed latency bound or a cycle-accurate simulator. Small working sets can be cache-resident. Strided accesses amplify transactions beyond useful bytes. Reductions may use multiple kernels. Each limitation is visible in the report and methodology; see [METHODS.md](METHODS.md).

## Repository

```text
benchmarks/   Reusable runner, timing, GEMM, memory, stride and reduction cases
kernels/      Triton strided copy and two-stage sum
models/       Pure-Python roofline and runtime estimate
profiling/    GPU metadata and optional PyTorch profiler trace
analysis/     Schema validation, comparison, CSV and HTML export
dashboard/    Portable interactive viewer and optional Streamlit wrapper
notebooks/    Self-contained Colab notebook
scripts/      Rebuild the notebook and export bundles
tests/        CPU model/schema tests and GPU kernel/timing checks
docs/         Runbook, GitHub instructions, and résumé guidance
results/      Empty viewer and verification artifacts; real GPU data goes here
```

GPU tests cover odd lengths, masked boundaries, multiple strides and block sizes, reduction buffers, CUDA-event timing, and precision-specific GEMM validation. CPU CI checks models and result formats; it cannot validate CUDA kernels.

## API

```python
from benchmarks.run import RunConfig, run_suite

result = run_suite(RunConfig(preset="full", rounds=7, iterations=20), "results/run.json")
```

Analysis modules do not import PyTorch, so collected data can be reviewed on a CPU-only machine.

## Technical references

- [PyTorch CUDA semantics and CUDA graph guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [Triton vector addition programming and validation](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton reduction API](https://triton-lang.org/main/python-api/generated/triton.language.sum.html)
- [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

See [docs/GITHUB.md](docs/GITHUB.md) for publication commands and [docs/RESUME.md](docs/RESUME.md) for claim boundaries.
