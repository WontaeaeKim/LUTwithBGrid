//#include "trilinear_kernel.h"
//#include <torch/extension.h>
//#include <THC/THC.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <ATen/cuda/CUDAEvent.h>

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void TriLinearForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, torch::Tensor output); 
void TriLinearBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &lut, const torch::Tensor &input,
     torch::Tensor grad_lut, torch::Tensor grad_image);
void TriLinearCPUForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, torch::Tensor output);
void TriLinearCPUBackwardLaucher(const torch::Tensor &grad_output, 
    const torch::Tensor &lut, const torch::Tensor &input, 
    torch::Tensor grad_lut, torch::Tensor grad_image);

void TetrahedralForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, torch::Tensor output); 
void TetrahedralBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &lut, const torch::Tensor &input,
     torch::Tensor grad_lut, torch::Tensor grad_image);
void TetrahedralCPUForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, torch::Tensor output);
void TetrahedralCPUBackwardLaucher(const torch::Tensor &grad_output, 
    const torch::Tensor &lut, const torch::Tensor &input, 
    torch::Tensor grad_lut, torch::Tensor grad_image);


void trilinear_forward_cuda(const torch::Tensor &lut,
    const torch::Tensor &input,
    torch::Tensor output)
{
    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(output);

        TriLinearForwardLaucher(lut, input, output);
    }
    else
    {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(output);

        TriLinearCPUForwardLaucher(lut, input, output);
    }
}
   
    

void trilinear_backward_cuda(const torch::Tensor &grad_output,
    const torch::Tensor &lut,
    const torch::Tensor &input,
    torch::Tensor grad_lut,
    torch::Tensor grad_inp)
{
    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_lut);
        
        TriLinearBackwardLaucher(grad_output, lut, input, grad_lut, grad_inp);
    }
    else
    {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_lut);

        TriLinearCPUBackwardLaucher(grad_output, lut, input, grad_lut, grad_inp);
    }
}

void tetrahedral_forward_cuda(const torch::Tensor &lut,
    const torch::Tensor &input,
    torch::Tensor output)
{
    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(output);

        TetrahedralForwardLaucher(lut, input, output);
    }
    else
    {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(output);

        TetrahedralCPUForwardLaucher(lut, input, output);
    }
}
   
    

void tetrahedral_backward_cuda(const torch::Tensor &grad_output,
    const torch::Tensor &lut,
    const torch::Tensor &input,
    torch::Tensor grad_lut,
    torch::Tensor grad_inp)
{
    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_lut);
        
        TetrahedralBackwardLaucher(grad_output, lut, input, grad_lut, grad_inp);
    }
    else
    {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_lut);

        TetrahedralCPUBackwardLaucher(grad_output, lut, input, grad_lut, grad_inp);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_forward", &trilinear_forward_cuda, "Trilinear forward");
  m.def("tri_backward", &trilinear_backward_cuda, "Trilinear backward");
  m.def("tetra_forward", &tetrahedral_forward_cuda, "Tetrahedral forward");
  m.def("tetra_backward", &tetrahedral_backward_cuda, "Tetrahedral backward");
}

