import math

import pytest

from models.performance_predictor import annotate, predict
from models.roofline import calibration_from_rows, ceiling_tflops, gbps, gemm_work, tflops


def test_rectangular_gemm_counts_every_operand():
    work = gemm_work(16, 32, 64, 2)
    assert work["flops"] == 65536
    assert work["logical_bytes"] == 7168
    assert work["arithmetic_intensity"] == pytest.approx(65536 / 7168)


def test_square_gemm_ai_matches_known_formula():
    assert gemm_work(1024, 1024, 1024, 2)["arithmetic_intensity"] == pytest.approx(1024 / 3)
    assert gemm_work(1024, 1024, 1024, 4)["arithmetic_intensity"] == pytest.approx(1024 / 6)


def test_throughput_units_are_decimal_not_binary():
    assert tflops(1e12, 1000) == 1
    assert gbps(1e9, 1000) == 1
    assert tflops(0, 1) == 0


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_time_is_rejected(value):
    with pytest.raises(ValueError):
        gbps(1024, value)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_work_is_rejected(value):
    with pytest.raises(ValueError):
        tflops(value, 1)


def test_roofline_ridge_is_continuous():
    assert ceiling_tflops(200, 1000, 200) == 200
    assert ceiling_tflops(100, 1000, 200) == 100
    assert ceiling_tflops(1000, 1000, 200) == 200
    assert ceiling_tflops(0, 1000, 200) == 0


def test_runtime_model_keeps_compute_memory_units_consistent():
    p = predict(2e12, 1e9, 200, 1000)
    assert p["compute_ms"] == 10
    assert p["memory_ms"] == 1
    assert p["predicted_ms"] == 10
    assert p["modeled_bottleneck"] == "compute"


def test_copy_has_zero_compute_and_memory_runtime():
    p = predict(0, 1e9, 200, 1000)
    assert p["predicted_ms"] == 1
    assert p["modeled_bottleneck"] == "memory"


def test_two_stage_launch_floor_is_explicit():
    p = predict(10, 10, 200, 1000, launch_floor_ms=0.003, launches=2)
    assert p["predicted_ms"] == 0.006
    assert p["modeled_bottleneck"] == "launch floor"


def test_calibration_excludes_fast_evaluation_cases(fixture_result):
    fixture_result["benchmarks"][-1]["tflops"] = 5000
    c = calibration_from_rows(fixture_result["benchmarks"])
    assert c["compute_tflops"] == {"float16": 1, "float32_ieee": 0.1}
    assert "eval_float16" not in c["calibration_case_ids"]


def test_each_precision_uses_its_own_calibration(fixture_result):
    rows = fixture_result["benchmarks"]
    summary = annotate(rows, calibration_from_rows(rows))
    assert summary["count"] == 1
    assert rows[-1]["prediction"]["predicted_ms"] == 1
    rows[-1]["compute_mode"] = "float32_ieee"
    annotate(rows, calibration_from_rows(rows))
    assert rows[-1]["prediction"]["predicted_ms"] == 10


def test_no_scoring_calibration_or_strided_cases(fixture_result):
    rows = fixture_result["benchmarks"]
    rows[-1]["category"] = "coalescing"
    summary = annotate(rows, calibration_from_rows(rows))
    assert summary["count"] == 0
    assert all("prediction" not in r for r in rows)


def test_mixed_timing_methods_are_not_scored(fixture_result):
    rows = fixture_result["benchmarks"]
    rows[-1]["timing"]["method"] = "eager"
    assert annotate(rows, calibration_from_rows(rows))["count"] == 0


def test_empty_calibration_explains_missing_prediction():
    assert annotate([], calibration_from_rows([]))["available"] is False


def test_eager_launch_floor_is_not_mixed_with_graph_calibration(fixture_result):
    import copy
    rows = fixture_result["benchmarks"]
    floor = copy.deepcopy(rows[0])
    floor.update(case_id="fill_floor", category="launch", operation="fill_one", latency_ms=10)
    floor["timing"]["method"] = "eager"
    rows.append(floor)
    annotate(rows, calibration_from_rows(rows))
    assert rows[3]["prediction"]["predicted_ms"] == 1
    assert rows[3]["prediction"]["launch_floor_ms"] == 0
