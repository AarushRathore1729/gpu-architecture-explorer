"""Empirical roofline helpers; bandwidth is logical bytes, not DRAM counters."""
from __future__ import annotations

import math


def positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def gemm_work(m: int, n: int, k: int, element_bytes: int) -> dict:
    for value, name in [(m, "m"), (n, "n"), (k, "k"), (element_bytes, "element_bytes")]:
        positive(value, name)
    # Dense GEMM convention: two FLOPs per multiply-add, beta=0.
    flops = 2 * m * n * k
    logical_bytes = element_bytes * (m * k + k * n + m * n)
    return {"flops": flops, "logical_bytes": logical_bytes,
            "arithmetic_intensity": flops / logical_bytes}


def tflops(flops: float, latency_ms: float) -> float:
    return nonnegative(flops, "flops") / (positive(latency_ms, "latency_ms") * 1e9)


def gbps(logical_bytes: float, latency_ms: float) -> float:
    return nonnegative(logical_bytes, "logical_bytes") / (positive(latency_ms, "latency_ms") * 1e6)


def ceiling_tflops(intensity: float, bandwidth_gbps: float, compute_tflops: float) -> float:
    return min(nonnegative(intensity, "intensity") * positive(bandwidth_gbps, "bandwidth_gbps") / 1000,
               positive(compute_tflops, "compute_tflops"))


def calibration_from_rows(rows: list[dict]) -> dict:
    calibrated = [r for r in rows if r.get("role") == "calibration" and r.get("status") == "ok"]
    copies = [r for r in calibrated if r.get("operation") == "copy"]
    floors = [r for r in calibrated if r.get("operation") == "fill_one"]
    compute = {}
    sources = {}
    methods = {}
    for row in calibrated:
        if row.get("category") == "gemm":
            mode = row["compute_mode"]
            if row["tflops"] > compute.get(mode, 0):
                compute[mode] = row["tflops"]
                sources[mode] = row["case_id"]
                methods[mode] = row.get("timing", {}).get("method")
    copy = max(copies, key=lambda r: r["effective_gbps"], default=None)
    floor = min(floors, key=lambda r: r["latency_ms"], default=None)
    return {
        "kind": "empirical_logical_byte_model",
        "compute_tflops": compute,
        "compute_sources": sources,
        "compute_timing_methods": methods,
        "bandwidth_gbps": copy["effective_gbps"] if copy else None,
        "bandwidth_source": copy["case_id"] if copy else None,
        "bandwidth_timing_method": copy.get("timing", {}).get("method") if copy else None,
        "launch_floor_ms": min((r["latency_ms"] for r in floors), default=0),
        "launch_floor_timing_method": floor.get("timing", {}).get("method") if floor else None,
        "calibration_case_ids": [r["case_id"] for r in calibrated],
    }
