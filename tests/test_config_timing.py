import subprocess
import sys

import pytest

from benchmarks.run import RunConfig
from benchmarks.timing import percentile, summarize


@pytest.mark.parametrize("kwargs", [{"rounds": 0}, {"iterations": -1}, {"warmup": 0}, {"device": -1}, {"preset": "unknown"}, {"timing": "host_clock"}])
def test_invalid_configuration_cannot_reach_gpu(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs)


def test_batch_statistics_are_not_request_percentiles():
    summary = summarize([1, 2, 3, 4, 5], "eager", 20)
    assert summary["median_ms"] == 3
    assert summary["p10_batch_ms"] == pytest.approx(1.4)
    assert summary["p90_batch_ms"] == pytest.approx(4.6)
    assert summary["rounds"] == 5
    assert summary["iterations_per_round"] == 20


def test_cli_help_does_not_need_pytorch():
    result = subprocess.run([sys.executable, "-m", "benchmarks.run", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "cuda_graph" in result.stdout


def test_percentile_rejects_empty_samples():
    with pytest.raises(ValueError):
        percentile([], 0.5)
