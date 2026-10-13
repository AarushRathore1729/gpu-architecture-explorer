import copy
import csv
import io

import pytest

from analysis.compare import compare
from analysis.report import prepare, resume_candidates, to_csv, to_html
from analysis.schema import validate_result


def test_schema_rejects_duplicate_cases(fixture_result):
    fixture_result["benchmarks"].append(copy.deepcopy(fixture_result["benchmarks"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_result(fixture_result)


def test_success_requires_a_correctness_check(fixture_result):
    fixture_result["benchmarks"][0]["validation"]["passed"] = False
    with pytest.raises(ValueError, match="correctness"):
        validate_result(fixture_result)


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf"), True])
def test_result_latency_must_be_a_positive_finite_number(fixture_result, value):
    fixture_result["benchmarks"][0]["latency_ms"] = value
    with pytest.raises(ValueError):
        validate_result(fixture_result)


def test_failed_cases_survive_without_fabricated_timings(fixture_result):
    fixture_result["benchmarks"].append({"case_id": "failure", "status": "failed", "role": "evaluation", "error": "test failure"})
    validate_result(fixture_result)
    assert "failure" in to_csv(fixture_result)


def test_csv_preserves_quoted_identifiers(fixture_result):
    fixture_result["benchmarks"][-1]["case_id"] = 'shape,"quoted"'
    csv_rows = list(csv.DictReader(io.StringIO(to_csv(fixture_result))))
    assert csv_rows[-1]["case_id"] == 'shape,"quoted"'


def test_report_escapes_closing_script_in_imported_metadata(fixture_result):
    fixture_result["device"]["name"] = "</script><script>alert('test')</script>"
    html = to_html(fixture_result)
    assert fixture_result["device"]["name"] not in html
    assert "\\u003c/script\\u003e" in html
    assert "__DATA_JSON__</script>" not in html


def test_empty_viewer_has_no_synthetic_measurements():
    html = to_html()
    assert '<script id="embedded-data" type="application/json">null</script>' in html
    assert "SYNTHETIC TEST DEVICE" not in html


def test_resume_candidates_only_use_successful_measurements(fixture_result):
    prepare(fixture_result)
    text = resume_candidates(fixture_result)
    assert "4 measured configurations" in text
    assert "1 held-out configurations" in text
    assert "hardware specification" not in text


def test_comparison_only_matches_valid_cases_and_timing_modes(fixture_result):
    b = copy.deepcopy(fixture_result)
    b["benchmarks"][-1]["latency_ms"] = 0.5
    matches = compare(fixture_result, b)
    assert matches[-1]["a_over_b_latency"] == 2
    b["benchmarks"][-1]["timing"]["method"] = "eager"
    assert len(compare(fixture_result, b)) == 3
