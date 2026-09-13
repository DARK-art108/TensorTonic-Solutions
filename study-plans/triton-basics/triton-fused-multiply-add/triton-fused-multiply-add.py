import torch
import triton
import triton.language as tl


@triton.jit
def fma_kernel(x_ptr, y_ptr, out_ptr, n, a, BLOCK_SIZE: tl.constexpr):
    pid= tl.program_id(axis=0)
    bs = pid * BLOCK_SIZE
    of = bs + tl.arange(0, BLOCK_SIZE)
    mask = of < n
    x = tl.load(x_ptr + of, mask=mask)
    y = tl.load(y_ptr + of, mask=mask)
    result = a * x + y
    tl.store(out_ptr + of, result, mask=mask)

    
    
def solve(a: float, x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch fma_kernel: out = a * x + y."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fma_kernel[grid](x, y, out, n, a, BLOCK_SIZE=BLOCK_SIZE)