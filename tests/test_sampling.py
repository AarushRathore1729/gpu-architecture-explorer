import pytest

from benchmarks.memory_bandwidth import sample_positions


@pytest.mark.parametrize("elements", [1, 7, 8192, 16_777_216, 33_554_432, 67_108_864])
def test_large_buffer_sampling_is_exact_and_within_bounds(elements):
    indices = sample_positions(elements)
    assert len(indices) == min(elements, 8192)
    assert indices[0] == 0
    assert indices[-1] == elements - 1
    assert all(isinstance(i, int) and 0 <= i < elements for i in indices)
    assert all(a < b for a, b in zip(indices, indices[1:]))
