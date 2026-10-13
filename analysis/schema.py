"""Validate portable measurement JSON before it is analyzed or displayed."""
from __future__ import annotations

import math


def validate_result(data: dict) -> dict:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected GPU Architecture Explorer schema_version=1")
    if not isinstance(data.get("device"), dict) or not data["device"].get("name"):
        raise ValueError("Result is missing device.name")
    if not isinstance(data.get("benchmarks"), list):
        raise ValueError("benchmarks must be a list")
    ids = set()
    for row in data["benchmarks"]:
        if not isinstance(row, dict) or not row.get("case_id") or row["case_id"] in ids:
            raise ValueError("Each benchmark needs a unique case_id")
        ids.add(row["case_id"])
        if row.get("status") not in {"ok", "failed"}:
            raise ValueError(f"Invalid status in {row['case_id']}")
        if row.get("role") not in {"calibration", "evaluation"}:
            raise ValueError(f"Invalid role in {row['case_id']}")
        if row["status"] == "ok":
            for key in ["latency_ms", "logical_bytes", "flops"]:
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Missing/non-finite {key} in {row['case_id']}")
                if value < 0 or (key == "latency_ms" and value == 0):
                    raise ValueError(f"Invalid {key} in {row['case_id']}")
            if row.get("validation", {}).get("passed") is not True:
                raise ValueError(f"Successful benchmark {row['case_id']} lacks a passed correctness check")
            samples = row.get("timing", {}).get("samples_ms", [])
            if not samples or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError(f"Invalid timing samples in {row['case_id']}")
    return data
