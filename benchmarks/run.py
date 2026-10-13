"""Run the suite: python -m benchmarks.run --preset full --output results/run.json."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from analysis.schema import validate_result
from benchmarks.timing import measure
from models.performance_predictor import annotate
from models.roofline import calibration_from_rows, gbps, tflops
from profiling.gpu_info import gpu_info, set_matmul_precision, smi_snapshot


@dataclass(frozen=True)
class RunConfig:
    preset: str = "full"
    rounds: int = 7
    iterations: int = 20
    warmup: int = 5
    timing: str = "cuda_graph"
    seed: int = 42
    device: int = 0
    skip_triton: bool = False
    profile: bool = False

    def __post_init__(self):
        if self.preset not in {"quick", "full"} or self.timing not in {"cuda_graph", "eager"}:
            raise ValueError("Invalid preset or timing method")
        if min(self.rounds, self.iterations, self.warmup) <= 0 or self.device < 0:
            raise ValueError("Invalid repetitions or device")


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for package in ["benchmarks", "kernels", "models", "profiling", "analysis"]:
        for path in sorted((root / package).glob("*.py")):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


class Collector:
    def __init__(self, torch, config: RunConfig, result: dict, output: Path):
        self.torch, self.config, self.result, self.output = torch, config, result, output

    def checkpoint(self):
        self.result["calibration"] = calibration_from_rows(self.result["benchmarks"])
        self.result["prediction_summary"] = annotate(self.result["benchmarks"], self.result["calibration"])
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.result, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.output)

    def add(self, case: dict, fn, validator):
        print(f"[{len(self.result['benchmarks']) + 1:02d}] {case['case_id']}", flush=True)
        try:
            timing = measure(self.torch, fn, rounds=self.config.rounds, iterations=self.config.iterations,
                             warmup=self.config.warmup, method=self.config.timing)
            correctness = validator()
            if correctness.get("passed") is not True:
                raise AssertionError("Correctness check did not pass")
            row = {**case, "status": "ok", "latency_ms": timing["median_ms"], "timing": timing,
                   "validation": correctness, "effective_gbps": gbps(case["logical_bytes"], timing["median_ms"]),
                   "tflops": tflops(case["flops"], timing["median_ms"]),
                   "arithmetic_intensity": case["flops"] / case["logical_bytes"]}
            print(f"     {row['latency_ms'] * 1000:.2f} us; {row['effective_gbps']:.1f} effective GB/s; validated", flush=True)
        except Exception as exc:
            row = {**case, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"     FAILED: {row['error']}", flush=True)
        self.result["benchmarks"].append(row)
        self.checkpoint()


def run_suite(config: RunConfig, output: str | Path) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing. Run this suite in a CUDA-enabled Colab/Lightning environment.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. In Colab select Runtime > Change runtime type > A100 GPU.")
    torch.cuda.set_device(config.device)
    torch.manual_seed(config.seed)
    quick = config.preset == "quick"
    device = gpu_info(torch, config.device)
    output = Path(output)
    result = {
        "schema_version": 1, "project_version": "0.1.1", "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device, "settings": asdict(config), "source_sha256": source_fingerprint(),
        "precision_api": set_matmul_precision(torch, "ieee"),
        "benchmarks": [], "skipped": [],
        "notes": [
            "TFLOP/s is dense GEMM throughput; sparse peaks are not used.",
            "Bandwidth uses useful logical bytes; it is not hardware-counter-measured HBM traffic.",
            "Empirical compute ceilings are calibrated separately for IEEE FP32, TF32, FP16 and BF16.",
            "Calibration cases are excluded from prediction scoring; evaluation uses held-out sizes in this session.",
            "The roofline estimate assumes minimum useful traffic and reuse. It is not a cycle-accurate simulator.",
            "Buffers are reused without cache flushing. Small-buffer results may include cache effects.",
            "Timing percentiles describe per-operation batch means, not request-tail latency.",
            "CUDA graph timing amortizes Python submission overhead; graph fallback is recorded per case.",
            "Strided copies are excluded from prediction errors because logical bytes omit amplified traffic.",
            "The launch floor is a measured one-element fill, not an isolated hardware launch counter.",
            "GEMM correctness checks cover a 16x16 region; custom-kernel boundary tests cover full small outputs.",
        ],
    }
    collector = Collector(torch, config, result, output)
    collector.checkpoint()
    started = time.monotonic()
    from benchmarks.gemm import run_gemm
    from benchmarks.memory_bandwidth import run_launch_floor, run_memory
    from benchmarks.reductions import run_reduction
    modes = ["float32_ieee", "float16"]
    if torch.cuda.get_device_capability()[0] >= 8:
        modes.append("float32_tf32")
    else:
        result["skipped"].append({"category": "gemm", "reason": "TF32 requires an Ampere-or-newer CUDA GPU"})
    if torch.cuda.is_bf16_supported():
        modes.append("bfloat16")
    else:
        result["skipped"].append({"category": "gemm", "reason": "BF16 is not supported on this GPU"})
    has_triton = not config.skip_triton
    if has_triton:
        try:
            import triton  # noqa: F401
        except ImportError:
            has_triton = False
            result["skipped"].append({"category": "triton", "reason": "Triton is not installed; custom kernels not run"})
    else:
        result["skipped"].append({"category": "triton", "reason": "Explicit --skip-triton"})

    with torch.inference_mode():
        run_launch_floor(torch, collector)
        run_memory(torch, collector, 128 if quick else 256, "copy", "calibration")
        n_cal = 2048 if quick else 4096
        sizes = [256, 512, 1024] if quick else [256, 512, 1024, 2048]
        for mode in modes:
            run_gemm(torch, collector, n_cal, n_cal, n_cal, mode, "calibration")
            for n in sizes:
                run_gemm(torch, collector, n, n, n, mode, "evaluation")
            run_gemm(torch, collector, 1024 if quick else 2048, 128, 512 if quick else 1024, mode, "evaluation")
        set_matmul_precision(torch, "ieee")
        for mib in [4, 64] if quick else [4, 16, 64, 128]:
            for operation in ["copy", "axpy"]:
                run_memory(torch, collector, mib, operation)
        if has_triton:
            from benchmarks.memory_patterns import run_strides
            run_strides(torch, collector, 16 if quick else 64,
                        [1, 4, 8] if quick else [1, 2, 4, 8, 16],
                        [256, 512] if quick else [128, 256, 512])
        for mib in [4, 16] if quick else [4, 16, 64]:
            run_reduction(torch, collector, mib, "pytorch")
            if has_triton:
                run_reduction(torch, collector, mib, "triton")
    result["duration_seconds"] = time.monotonic() - started
    result["device"]["nvidia_smi_after"] = smi_snapshot()
    failed = [r for r in result["benchmarks"] if r["status"] == "failed"]
    result["status"] = "completed_with_failures" if failed else ("completed_with_skips" if result["skipped"] else "completed")
    if config.profile:
        try:
            from profiling.profile import export_trace
            trace = output.parent / "trace.json"
            export_trace(trace, device=config.device, use_triton=has_triton)
            result["profiler"] = {"status": "ok", "file": trace.name}
        except Exception as exc:
            result["profiler"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    collector.checkpoint()
    validate_result(result)
    from analysis.report import write_outputs
    write_outputs(result, output.parent)
    ok = len(result["benchmarks"]) - len(failed)
    print(f"\n{ok} validated measurements; {len(failed)} failures; status={result['status']}")
    print(f"Results: {output.resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["quick", "full"], default="full")
    parser.add_argument("--output", default="results/run.json")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timing", choices=["cuda_graph", "eager"], default="cuda_graph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    try:
        result = run_suite(RunConfig(**{k: v for k, v in vars(args).items() if k != "output"}), args.output)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    if any(r["status"] == "failed" for r in result["benchmarks"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
