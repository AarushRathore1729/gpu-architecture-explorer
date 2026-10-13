"""Roofline runtime estimate, evaluated on configurations excluded from calibration."""
from __future__ import annotations

from statistics import mean, median

from .roofline import nonnegative, positive


def predict(flops: float, logical_bytes: float, compute_tflops: float,
            bandwidth_gbps: float, launch_floor_ms: float = 0, launches: int = 1) -> dict:
    flops = nonnegative(flops, "flops")
    logical_bytes = nonnegative(logical_bytes, "logical_bytes")
    compute_ms = flops / (positive(compute_tflops, "compute_tflops") * 1e9)
    memory_ms = logical_bytes / (positive(bandwidth_gbps, "bandwidth_gbps") * 1e6)
    floor = nonnegative(launch_floor_ms, "launch_floor_ms") * positive(launches, "launches")
    ideal = max(compute_ms, memory_ms)
    if floor >= ideal:
        bottleneck = "launch floor"
    elif compute_ms >= memory_ms:
        bottleneck = "compute"
    else:
        bottleneck = "memory"
    return {"compute_ms": compute_ms, "memory_ms": memory_ms,
            "ideal_roofline_ms": ideal, "launch_floor_ms": floor,
            "predicted_ms": max(ideal, floor), "modeled_bottleneck": bottleneck}


def annotate(rows: list[dict], calibration: dict) -> dict:
    """Mutate evaluation rows with predictions; never score calibration rows.

    Strided copies are deliberately excluded from prediction scoring: useful
    bytes do not describe transaction amplification. Cache-resident cases remain
    visible, and category errors expose the limits of this simple model.
    """
    bandwidth = calibration.get("bandwidth_gbps")
    compute = calibration.get("compute_tflops", {})
    if not bandwidth or not compute:
        return {"available": False, "reason": "Compute or bandwidth calibration missing", "count": 0}
    errors: list[float] = []
    by_category: dict[str, list[float]] = {}
    for row in rows:
        row.pop("prediction", None)
        row.pop("model_note", None)
        if row.get("status") != "ok" or row.get("role") != "evaluation":
            continue
        if row.get("category") == "coalescing":
            row["model_note"] = "Excluded from model scoring: logical bytes omit stride-dependent traffic."
            continue
        mode = row.get("compute_mode", "float32_ieee")
        if mode not in compute:
            row["model_note"] = f"No calibration for {mode}"
            continue
        timing_method = row.get("timing", {}).get("method")
        if timing_method != calibration.get("bandwidth_timing_method") or timing_method != calibration.get("compute_timing_methods", {}).get(mode):
            row["model_note"] = "Excluded from scoring: calibration and evaluation timing methods differ."
            continue
        floor = calibration.get("launch_floor_ms", 0) if calibration.get("launch_floor_timing_method") == timing_method else 0
        result = predict(row.get("flops", 0), row["logical_bytes"], compute[mode], bandwidth,
                         floor, row.get("kernel_launches", 1))
        measured = positive(row["latency_ms"], "latency_ms")
        error = abs(result["predicted_ms"] - measured) / measured * 100
        result["absolute_percentage_error"] = error
        result["measured_over_predicted"] = measured / result["predicted_ms"]
        row["prediction"] = result
        errors.append(error)
        by_category.setdefault(row["category"], []).append(error)
    return {"available": bool(errors), "count": len(errors),
            "mape_percent": mean(errors) if errors else None,
            "median_ape_percent": median(errors) if errors else None,
            "by_category": {key: {"count": len(v), "mape_percent": mean(v)} for key, v in by_category.items()},
            "scoring": "Held-out configuration sizes from the same GPU session; not a cross-device model."}
