from __future__ import annotations

from models.roofline import gemm_work
from profiling.gpu_info import set_matmul_precision


def run_gemm(torch, collector, m: int, n: int, k: int, mode: str, role: str):
    dtype = {"float32_ieee": torch.float32, "float32_tf32": torch.float32,
             "float16": torch.float16, "bfloat16": torch.bfloat16}[mode]
    precision = "tf32" if mode == "float32_tf32" else "ieee"
    set_matmul_precision(torch, precision)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / k ** 0.5
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    case = {"case_id": f"gemm_{mode}_{m}x{n}x{k}", "category": "gemm", "operation": "gemm",
            "backend": "pytorch_cublas", "role": role, "compute_mode": mode,
            "dtype": str(dtype).removeprefix("torch."), "precision": precision,
            "m": m, "n": n, "k": k, "kernel_launches": 1,
            "footprint_bytes": sum(x.numel() * x.element_size() for x in (a, b, out)),
            **gemm_work(m, n, k, a.element_size())}

    def validate():
        # Compare a 16x16 output region with an IEEE FP32 reference; validate
        # after timing and restore the benchmark's precision setting.
        set_matmul_precision(torch, "ieee")
        try:
            reference = torch.mm(a[:16].float(), b[:, :16].float())
            actual = out[:16, :16].float()
            rtol, atol = {"float32_ieee": (2e-4, 2e-4), "float32_tf32": (1e-2, 1e-2),
                          "float16": (1e-2, 5e-3), "bfloat16": (3e-2, 2e-2)}[mode]
            torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
            return {"passed": True, "coverage": "16x16 output region vs IEEE FP32 reference",
                    "max_abs_error": float((actual - reference).abs().max().item()), "rtol": rtol, "atol": atol}
        finally:
            set_matmul_precision(torch, precision)

    collector.add(case, lambda: torch.mm(a, b, out=out), validate)
