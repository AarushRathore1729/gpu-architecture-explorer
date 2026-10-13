import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(X, P, N: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + indices, mask=indices < N, other=0).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(P + tl.program_id(0), total)


@triton.jit
def finish_sum_kernel(P, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.arange(0, BLOCK)
    values = tl.load(P + indices, mask=indices < N, other=0)
    tl.store(Y, tl.sum(values, axis=0))


def launch_sum(src, partial, out, block: int = 4096):
    if block < 128 or block > 8192 or block & (block - 1):
        raise ValueError("Reduction block must be a power of two from 128 to 8192")
    tiles = triton.cdiv(src.numel(), block)
    if src.numel() == 0 or partial.numel() < tiles or out.numel() != 1:
        raise ValueError("Invalid reduction buffers")
    if not all(x.is_cuda and x.is_contiguous() and x.device == src.device for x in (src, partial, out)):
        raise ValueError("Reduction buffers must be contiguous on the same CUDA device")
    if str(partial.dtype) != "torch.float32" or str(out.dtype) != "torch.float32":
        raise ValueError("Partial and output buffers must be float32")
    partial_sum_kernel[(tiles,)](src, partial, N=src.numel(), BLOCK=block, num_warps=4 if block <= 1024 else 8)
    finish_sum_kernel[(1,)](partial, out, N=tiles, BLOCK=triton.next_power_of_2(tiles),
                          num_warps=4 if tiles <= 1024 else 8)
