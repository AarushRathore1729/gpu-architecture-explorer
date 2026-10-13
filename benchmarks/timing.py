"""CUDA event timing with compilation/warmup excluded and raw batch samples."""
from __future__ import annotations

from statistics import mean, median, pstdev


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError("Need samples and a quantile in [0, 1]")
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(samples: list[float], method: str, iterations: int, fallback: str | None = None) -> dict:
    import math
    if iterations <= 0 or not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError("Timing needs positive, finite samples and iterations")
    return {"method": method, "samples_ms": samples, "median_ms": median(samples),
            "p10_batch_ms": percentile(samples, 0.1), "p90_batch_ms": percentile(samples, 0.9),
            "min_batch_ms": min(samples), "max_batch_ms": max(samples),
            "coefficient_of_variation": pstdev(samples) / mean(samples),
            "rounds": len(samples), "iterations_per_round": iterations,
            "graph_fallback_reason": fallback}


def measure(torch, fn, *, rounds: int = 7, iterations: int = 20,
            warmup: int = 5, method: str = "cuda_graph") -> dict:
    if min(rounds, iterations, warmup) <= 0 or method not in {"cuda_graph", "eager"}:
        raise ValueError("Invalid timing settings")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    actual_method = method
    fallback = None
    graph = None
    if method == "cuda_graph":
        try:
            # Warm the library on a non-default stream before capture, as in
            # PyTorch's graph guidance. fn must reuse preallocated outputs.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(iterations):
                    fn()
        except Exception as exc:
            graph = None
            actual_method = "eager"
            fallback = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize event resources outside the timed samples.
    start.record()
    end.record()
    end.synchronize()
    samples = []
    for _ in range(rounds):
        start.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(iterations):
                fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return summarize(samples, actual_method, iterations, fallback)
