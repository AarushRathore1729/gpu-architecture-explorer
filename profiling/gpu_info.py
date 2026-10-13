from __future__ import annotations

import platform
import subprocess


def gpu_info(torch, index: int) -> dict:
    p = torch.cuda.get_device_properties(index)
    result = {"name": p.name, "index": index, "compute_capability": f"{p.major}.{p.minor}",
              "multiprocessors": p.multi_processor_count, "vram_bytes": p.total_memory,
              "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "python_version": platform.python_version(), "platform": platform.platform(),
              "l2_cache_bytes": getattr(p, "L2_cache_size", None)}
    try:
        import triton
        result["triton_version"] = triton.__version__
    except ImportError:
        result["triton_version"] = None
    result["nvidia_smi_before"] = smi_snapshot()
    return result


def smi_snapshot() -> str | None:
    try:
        output = subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        return output.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def set_matmul_precision(torch, precision: str) -> str:
    if precision not in {"ieee", "tf32"}:
        raise ValueError("precision must be ieee or tf32")
    # PyTorch recommends the new API and disallows mixing it with allow_tf32.
    try:
        getattr(torch.backends.cuda.matmul, "fp32_precision")
    except (AttributeError, AssertionError):
        # Older backend attribute proxies raise AssertionError rather than
        # AttributeError for unknown settings, so hasattr alone is unsafe.
        pass
    else:
        torch.backends.cuda.matmul.fp32_precision = precision
        return "torch.backends.cuda.matmul.fp32_precision"
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    return "torch.backends.cuda.matmul.allow_tf32 (legacy compatibility)"
