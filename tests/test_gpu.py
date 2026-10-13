"""GPU-only correctness and timing checks. Run these in Colab before benchmarking."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)
pytest.importorskip("triton")

from benchmarks.gemm import run_gemm
from benchmarks.memory_bandwidth import sample_positions
from benchmarks.run import Collector, RunConfig
from benchmarks.timing import measure
from kernels.triton_memory import launch_copy
from kernels.triton_reduction import launch_sum

pytestmark = pytest.mark.gpu


def test_sampling_128mib_element_count_avoids_float32_endpoint_rounding():
    elements = 33_554_432
    source = torch.zeros(elements, dtype=torch.uint8, device="cuda")
    source[-1] = 7
    index = torch.tensor(sample_positions(elements), dtype=torch.int64, device="cuda")
    assert source[index][-1].item() == 7


@pytest.mark.parametrize("elements", [1, 127, 129, 1027, 65537])
@pytest.mark.parametrize("stride", [1, 3, 8])
@pytest.mark.parametrize("block", [128, 512])
def test_strided_copy_masked_boundaries(elements, stride, block):
    src = torch.randn(elements * stride, device="cuda", dtype=torch.float32)
    dst = torch.empty(elements, device="cuda", dtype=torch.float32)
    launch_copy(src, dst, stride, block)
    assert torch.equal(dst, src[::stride])


@pytest.mark.parametrize("elements", [1, 257, 8193, 65539])
@pytest.mark.parametrize("block", [1024, 4096])
def test_two_stage_reduction_masked_boundaries(elements, block):
    import triton
    src = torch.rand(elements, device="cuda")
    partial = torch.empty(triton.cdiv(elements, block), device="cuda")
    out = torch.empty((), device="cuda")
    launch_sum(src, partial, out, block)
    torch.testing.assert_close(out.double(), src.sum(dtype=torch.float64), rtol=3e-6, atol=1e-3)


def test_copy_rejects_undersized_input():
    src = torch.empty(8, device="cuda")
    dst = torch.empty(8, device="cuda")
    with pytest.raises(ValueError):
        launch_copy(src, dst, 2)


def test_sum_rejects_undersized_partial_buffer():
    src = torch.empty(8193, device="cuda")
    partial = torch.empty(1, device="cuda")
    out = torch.empty((), device="cuda")
    with pytest.raises(ValueError):
        launch_sum(src, partial, out, 4096)


@pytest.mark.parametrize("method", ["cuda_graph", "eager"])
def test_gpu_timer_records_valid_raw_samples(method):
    out = torch.empty(1024, device="cuda")
    timing = measure(torch, lambda: out.zero_(), rounds=3, iterations=5, warmup=2, method=method)
    assert len(timing["samples_ms"]) == 3
    assert timing["median_ms"] > 0
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("mode", ["float32_ieee", "float32_tf32", "float16", "bfloat16"])
def test_gemm_precision_and_sampled_correctness(mode, tmp_path):
    if mode == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    if mode == "float32_tf32" and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("TF32 unavailable")
    result = {"benchmarks": []}
    collector = Collector(torch, RunConfig(rounds=2, iterations=3, warmup=2), result, tmp_path / "test.json")
    run_gemm(torch, collector, 128, 128, 128, mode, "evaluation")
    assert result["benchmarks"][0]["status"] == "ok", result["benchmarks"][0].get("error")
