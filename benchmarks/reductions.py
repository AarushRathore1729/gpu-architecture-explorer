from __future__ import annotations


def run_reduction(torch, collector, mib: int, backend: str, block: int = 4096):
    elements = mib * 1024 ** 2 // 4
    # Positive inputs avoid cancellation when checking a scalar sum.
    src = torch.rand(elements, device="cuda", dtype=torch.float32)
    out = torch.empty((), device="cuda", dtype=torch.float32)
    partial = None
    if backend == "triton":
        import triton
        from kernels.triton_reduction import launch_sum
        tiles = triton.cdiv(elements, block)
        partial = torch.empty(tiles, device="cuda", dtype=torch.float32)
        fn = lambda: launch_sum(src, partial, out, block)
    else:
        fn = lambda: torch.sum(src, dim=0, out=out)
    traffic = elements * 4 + 4 + (partial.numel() * 8 if partial is not None else 0)

    def validate():
        reference = src.sum(dtype=torch.float64)
        torch.testing.assert_close(out.double(), reference, rtol=3e-6, atol=1e-3)
        return {"passed": True, "coverage": "scalar vs full-input FP64 sum", "rtol": 3e-6, "atol": 1e-3,
                "relative_error": float(((out.double() - reference).abs() / reference.abs()).item())}

    collector.add({"case_id": f"reduce_{backend}_{mib}mib", "category": "reduction", "operation": "sum",
                   "backend": backend, "role": "evaluation", "dtype": "float32", "size_mib": mib,
                   "elements": elements, "logical_bytes": traffic, "flops": elements - 1,
                   "kernel_launches": 2 if partial is not None else 1,
                   "block_size": block if partial is not None else None,
                   "footprint_bytes": elements * 4 + (partial.numel() * 4 if partial is not None else 0) + 4,
                   "traffic_note": "Ideal useful input/output bytes plus explicit Triton partial buffers; not measured DRAM bytes."},
                  fn, validate)
