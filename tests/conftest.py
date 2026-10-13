"""Synthetic fixtures are test inputs only, never exported as GPU measurements."""
import copy

import pytest


def row(case_id, category="gemm", role="evaluation", mode="float16", latency=1.0):
    return {"case_id": case_id, "category": category, "role": role, "operation": "gemm" if category == "gemm" else "copy",
            "status": "ok", "compute_mode": mode, "latency_ms": latency, "logical_bytes": 1_000_000,
            "flops": 1_000_000_000, "tflops": 1.0 / latency, "effective_gbps": 1.0 / latency,
            "arithmetic_intensity": 1000, "validation": {"passed": True, "coverage": "SYNTHETIC TEST FIXTURE"},
            "timing": {"method": "cuda_graph", "samples_ms": [latency, latency], "p10_batch_ms": latency,
                       "p90_batch_ms": latency, "coefficient_of_variation": 0}}


@pytest.fixture
def fixture_result():
    compute = row("cal_float16", role="calibration")
    ieee = row("cal_float32", role="calibration", mode="float32_ieee", latency=10)
    copy_row = row("cal_copy", category="memory", role="calibration")
    copy_row.update(flops=0, logical_bytes=1_000_000_000, effective_gbps=1000, arithmetic_intensity=0)
    return copy.deepcopy({"schema_version": 1, "device": {"name": "SYNTHETIC TEST DEVICE"},
                          "status": "completed", "benchmarks": [compute, ieee, copy_row, row("eval_float16")],
                          "settings": {}, "notes": [], "skipped": []})
