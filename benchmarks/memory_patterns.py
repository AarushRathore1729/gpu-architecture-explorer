from __future__ import annotations


def run_strides(torch, collector, mib: int, strides: list[int], blocks: list[int]):
    from kernels.triton_memory import launch_copy
    elements = mib * 1024 ** 2 // 4
    for stride in strides:
        src = torch.randn(elements * stride, device="cuda", dtype=torch.float32)
        dst = torch.empty(elements, device="cuda", dtype=torch.float32)
        for block in blocks:
            case = {"case_id": f"stride_{stride}_block_{block}_{mib}mib", "category": "coalescing",
                    "operation": "strided_copy", "backend": "triton", "role": "evaluation",
                    "dtype": "float32", "size_mib": mib, "elements": elements,
                    "stride": stride, "block_size": block, "num_warps": 4 if block <= 512 else 8,
                    "logical_bytes": elements * 8, "flops": 0, "kernel_launches": 1,
                    "footprint_bytes": (src.numel() + dst.numel()) * 4,
                    "input_address_span_bytes": ((elements - 1) * stride + 1) * 4,
                    "cache_note": "Triton .cg loads; L2 is not flushed. Logical bytes exclude transaction amplification."}

            def validate():
                if not torch.equal(dst, src[::stride]):
                    raise AssertionError(f"Incorrect strided copy: stride={stride}, block={block}")
                return {"passed": True, "coverage": "entire output, exact equality"}

            collector.add(case, lambda: launch_copy(src, dst, stride, block), validate)
        del src, dst
