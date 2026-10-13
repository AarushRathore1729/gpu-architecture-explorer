"""Optional PyTorch profiler trace, collected separately from timing."""
from pathlib import Path


def export_trace(path: str | Path, device: int = 0, use_triton: bool = True):
    import torch
    torch.cuda.set_device(device)
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    src = torch.randn(1024 * 1024, device="cuda")
    dst = torch.empty_like(src)
    if use_triton:
        from kernels.triton_memory import launch_copy
        launch_copy(src, dst)
    for _ in range(3):
        torch.mm(a, b, out=c)
        dst.copy_(src)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as profile:
        for _ in range(3):
            with torch.profiler.record_function("explorer/fp16_gemm"):
                torch.mm(a, b, out=c)
            with torch.profiler.record_function("explorer/contiguous_copy"):
                dst.copy_(src)
            if use_triton:
                with torch.profiler.record_function("explorer/triton_copy"):
                    launch_copy(src, dst)
        torch.cuda.synchronize()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(path))
