import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(axis=0) #0
    bs = pid * BLOCK_SIZE 
    shifts = bs + tl.arange(0,BLOCK_SIZE)
    masks = shifts < n
    x = tl.load(x_ptr+shifts, masks)
    y = tl.load(y_ptr+shifts, masks)
    tl.store(out_ptr+shifts, x+y, masks)

def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vector_add_kernel on the provided tensors."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)