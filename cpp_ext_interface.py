import torch

from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Tuple

import lut_transform
import bilateral_slicing


class TrilinearLUTTransformFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, 
                lut: torch.Tensor, 
                x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        lut = lut.contiguous()

        assert x.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert lut.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        
        output = x.new_zeros((x.size(0), lut.size(1), x.size(2), x.size(3)))
        output.contiguous()
               
        lut_transform.tri_forward(lut, x, output)
        
        ctx.save_for_backward(lut, x)
        
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        
        grad_output = grad_output.contiguous()
        
        lut, x = ctx.saved_tensors
                
        grad_img = torch.zeros_like(x)
        grad_lut = torch.zeros_like(lut)          
        
        lut_transform.tri_backward(grad_output, lut, x, grad_lut, grad_img)

        return grad_lut, grad_img



def trilinear_lut_transform(
    lut: torch.Tensor,
    img: torch.Tensor) -> torch.Tensor:
    r"""Trilinear 3D Lookup Table Transform.

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut (torch.Tensor): output values of the 3D LUT, shape (b, 3, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, h, w).
    """
    return TrilinearLUTTransformFunction.apply(lut, img)

class TetrahedralLUTTransformFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, 
                lut: torch.Tensor, 
                x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        lut = lut.contiguous()

        assert x.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert lut.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        
        output = x.new_zeros((x.size(0), lut.size(1), x.size(2), x.size(3)))
        output.contiguous()
               
        lut_transform.tetra_forward(lut, x, output)
        
        ctx.save_for_backward(lut, x)
        
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        
        grad_output = grad_output.contiguous()
        
        lut, x = ctx.saved_tensors
                
        grad_img = torch.zeros_like(x)
        grad_lut = torch.zeros_like(lut)          
        
        lut_transform.tetra_backward(grad_output, lut, x, grad_lut, grad_img)

        return grad_lut, grad_img
    
def tetrahedral_lut_transform(
    lut: torch.Tensor,
    img: torch.Tensor) -> torch.Tensor:
    r"""Tetrahedral 3D Lookup Table Transform.

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut (torch.Tensor): output values of the 3D LUT, shape (b, 3, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, h, w).
    """
    return TetrahedralLUTTransformFunction.apply(lut, img)


class TrilinearSliceFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, 
                grid: torch.Tensor, 
                x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        grid = grid.contiguous()

        assert x.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert grid.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        
        output = x.new_zeros((x.size(0), grid.size(1), x.size(2), x.size(3)))
        output.contiguous()
               
        bilateral_slicing.tri_forward(grid, x, output)
        
        ctx.save_for_backward(grid, x)
        
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        
        grad_output = grad_output.contiguous()
        
        grid, x = ctx.saved_tensors
                
        grad_img = torch.zeros_like(x)
        grad_grid = torch.zeros_like(grid)          
        
        bilateral_slicing.tri_backward(grad_output, grid, x, grad_grid, grad_img)

        return grad_grid, grad_img


def trilinear_slice_function(
    grid: torch.Tensor,
    img: torch.Tensor) -> torch.Tensor:
    r"""Trilinear Bilateral Grid Transform.

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        grid (torch.Tensor): output values of the Bilateral grid, shape (b, 3*N, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3*(N + 1), h, w).
    """
    return TrilinearSliceFunction.apply(grid, img)

class TetrahedralSliceFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, 
                grid: torch.Tensor, 
                x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        grid = grid.contiguous()

        assert x.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert grid.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        
        output = x.new_zeros((x.size(0), grid.size(1), x.size(2), x.size(3)))
        output.contiguous()
               
        bilateral_slicing.tetra_forward(grid, x, output)
        
        ctx.save_for_backward(grid, x)
        
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        
        grad_output = grad_output.contiguous()
        
        grid, x = ctx.saved_tensors
                
        grad_img = torch.zeros_like(x)
        grad_grid = torch.zeros_like(grid)          
        
        bilateral_slicing.tetra_backward(grad_output, grid, x, grad_grid, grad_img)

        return grad_grid, grad_img


def tetrahedral_slice_function(
    grid: torch.Tensor,
    img: torch.Tensor) -> torch.Tensor:
    r"""Tetrahedral Bilateral Grid Transform.

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut (torch.Tensor): output values of the 3D LUT, shape (b, 3*N, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3*(N + 1), h, w).
    """
    return TetrahedralSliceFunction.apply(grid, img)
