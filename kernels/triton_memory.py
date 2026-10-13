import triton
import triton.language as tl


@triton.jit
def strided_copy_kernel(X, Y, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + indices * STRIDE, mask=indices < N, other=0, cache_modifier=".cg")
    tl.store(Y + indices, values, mask=indices < N)


def launch_copy(src, dst, stride: int = 1, block: int = 256):
    if stride < 1 or block < 128 or block > 2048 or block & (block - 1):
        raise ValueError("stride must be positive; block must be a power of two from 128 to 2048")
    if not src.is_cuda or not dst.is_cuda or src.device != dst.device:
        raise ValueError("Copy buffers must be on the same CUDA device")
    if not src.is_contiguous() or not dst.is_contiguous() or src.dtype != dst.dtype:
        raise ValueError("Copy buffers must be contiguous and have matching dtypes")
    if src.numel() < (dst.numel() - 1) * stride + 1 or dst.numel() == 0:
        raise ValueError("Input backing storage is too small or output is empty")
    return strided_copy_kernel[(triton.cdiv(dst.numel(), block),)](
        src, dst, N=dst.numel(), STRIDE=stride, BLOCK=block, num_warps=4 if block <= 512 else 8)
