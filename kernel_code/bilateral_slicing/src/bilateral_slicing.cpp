//#include "trilinear_kernel.h"
//#include <torch/extension.h>
//#include <THC/THC.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <ATen/cuda/CUDAEvent.h>

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void TriLinearSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor output); 
void TriLinearSliceBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &grid, const torch::Tensor &input,
     torch::Tensor grad_grid, torch::Tensor grad_image);
void TriLinearCPUSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor output); 
void TriLinearCPUSliceBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &grid, const torch::Tensor &input,
     torch::Tensor grad_grid, torch::Tensor grad_image);

void TetrahedralSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor output); 
void TetrahedralSliceBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &grid, const torch::Tensor &input,
     torch::Tensor grad_grid, torch::Tensor grad_image);
void TetrahedralCPUSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor output); 
void TetrahedralCPUSliceBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &grid, const torch::Tensor &input,
     torch::Tensor grad_grid, torch::Tensor grad_image);

void trilinearslice_forward_cuda(const torch::Tensor &grid,
    const torch::Tensor &input,
    torch::Tensor output)
{
    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(output);
        
        TriLinearSliceForwardLaucher(grid, input, output);
    }
    else
    {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(output);
        
        TriLinearCPUSliceForwardLaucher(grid, input, output);
    }
}

void trilinearslice_backward_cuda(const torch::Tensor &grad_output,
    const torch::Tensor &grid,
    const torch::Tensor &input,
    torch::Tensor grad_grid,
    torch::Tensor grad_inp)
{
    

    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_grid);
        
        TriLinearSliceBackwardLaucher(grad_output, grid, input, grad_grid, grad_inp);
    }
    else
    {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_grid);
        
        TriLinearCPUSliceBackwardLaucher(grad_output, grid, input, grad_grid, grad_inp);
    }
}

void tetrahedralslice_forward_cuda(const torch::Tensor &grid,
    const torch::Tensor &input,
    torch::Tensor output)
{
    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(output);
        
        TetrahedralSliceForwardLaucher(grid, input, output);
    }
    else
    {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(output);
        
        TetrahedralCPUSliceForwardLaucher(grid, input, output);
    }
}

void tetrahedralslice_backward_cuda(const torch::Tensor &grad_output,
    const torch::Tensor &grid,
    const torch::Tensor &input,
    torch::Tensor grad_grid,
    torch::Tensor grad_inp)
{
    

    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_grid);
        
        TetrahedralSliceBackwardLaucher(grad_output, grid, input, grad_grid, grad_inp);
    }
    else
    {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_grid);
        
        TetrahedralCPUSliceBackwardLaucher(grad_output, grid, input, grad_grid, grad_inp);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_forward", &trilinearslice_forward_cuda, "Trilinear Slice forward");
  m.def("tri_backward", &trilinearslice_backward_cuda, "Trilinear Slice backward");
  m.def("tetra_forward", &tetrahedralslice_forward_cuda, "Tetrahedral Slice forward");
  m.def("tetra_backward", &tetrahedralslice_backward_cuda, "Tetrahedral Slice backward");
}

