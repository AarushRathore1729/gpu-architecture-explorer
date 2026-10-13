from __future__ import annotations


def sample_positions(elements: int, maximum_samples: int = 8192) -> list[int]:
    """Inclusive, evenly spaced indices without float32 endpoint rounding."""
    if elements < 1 or maximum_samples < 1:
        raise ValueError("elements and maximum_samples must be positive")
    samples = min(elements, maximum_samples)
    denominator = max(1, samples - 1)
    return [i * (elements - 1) // denominator for i in range(samples)]


def run_memory(torch, collector, mib: int, operation: str, role: str = "evaluation"):
    elements = mib * 1024 ** 2 // 4
    src = torch.randn(elements, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    other = torch.randn_like(src) if operation == "axpy" else None
    logical_bytes = elements * 4 * (3 if other is not None else 2)
    case = {"case_id": f"{operation}_{mib}mib", "category": "memory", "operation": operation,
            "backend": "pytorch", "role": role, "dtype": "float32", "size_mib": mib,
            "elements": elements, "logical_bytes": logical_bytes,
            "flops": elements * 2 if operation == "axpy" else 0,
            "footprint_bytes": logical_bytes, "kernel_launches": 1,
            "cache_note": "Repeated buffers; no cache flush. Footprint controls cache effects."}

    def validate():
        if operation == "copy":
            if not torch.equal(src, dst):
                raise AssertionError("Copy output differs from input")
            return {"passed": True, "coverage": "entire output, exact equality"}
        index = torch.tensor(sample_positions(elements), device=src.device, dtype=torch.int64)
        expected = src[index] + 3 * other[index]
        torch.testing.assert_close(dst[index], expected, rtol=1e-5, atol=1e-6)
        return {"passed": True, "coverage": "8192 evenly spaced elements", "rtol": 1e-5, "atol": 1e-6}

    fn = (lambda: dst.copy_(src)) if operation == "copy" else (lambda: torch.add(src, other, alpha=3, out=dst))
    collector.add(case, fn, validate)


def run_launch_floor(torch, collector):
    out = torch.empty(1, device="cuda", dtype=torch.float32)

    def validate():
        if out.item() != 0:
            raise AssertionError("Fill output is incorrect")
        return {"passed": True, "coverage": "single element"}

    collector.add({"case_id": "launch_fill_one", "category": "launch", "operation": "fill_one",
                   "backend": "pytorch", "role": "calibration", "logical_bytes": 4, "flops": 0,
                   "kernel_launches": 1, "footprint_bytes": 4}, lambda: out.zero_(), validate)
