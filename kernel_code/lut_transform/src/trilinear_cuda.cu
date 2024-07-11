#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  return min(optimal_block_num, max_block_num);
}

/* std::clamp is only available since c++17 */
template <typename scalar_t>
inline __device__ constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void TriLinearForward(const int nthreads, 
                                const scalar_t* __restrict__ lut, 
                                const scalar_t* __restrict__ image, 
                                scalar_t* __restrict__ output, 
                                const int dim, 
                                const int shift, 
                                const scalar_t binsize, 
                                const int width, 
                                const int height, 
                                const int num_channels) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

        const scalar_t r = image[index];
	    const scalar_t g = image[index + width * height];
	    const scalar_t b = image[index + width * height * 2];

	    const int32_t r_id = clamp((int32_t)floor(r * (dim-1)),0, dim-2);
	    const int32_t g_id = clamp((int32_t)floor(g * (dim-1)),0, dim-2);
	    const int32_t b_id = clamp((int32_t)floor(b * (dim-1)),0, dim-2);

        const scalar_t  r_d = (r - binsize * r_id) / binsize;
        const scalar_t  g_d = (g - binsize * g_id) / binsize;
        const scalar_t  b_d = (b - binsize * b_id) / binsize;

        const int id000 = r_id     + g_id * dim       + b_id * dim * dim;
        const int id100 = r_id + 1 + g_id * dim       + b_id * dim * dim;
        const int id010 = r_id     + (g_id + 1) * dim + b_id * dim * dim;
        const int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        const int id001 = r_id     + g_id * dim       + (b_id + 1) * dim * dim;
        const int id101 = r_id + 1 + g_id * dim       + (b_id + 1) * dim * dim;
        const int id011 = r_id     + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        const int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;

        const scalar_t  w000 = (1-r_d)*(1-g_d)*(1-b_d);
        const scalar_t  w100 = r_d    *(1-g_d)*(1-b_d);
        const scalar_t  w010 = (1-r_d)*g_d    *(1-b_d);
        const scalar_t  w110 = r_d*g_d*(1-b_d);
        const scalar_t  w001 = (1-r_d)*(1-g_d)*b_d;
        const scalar_t  w101 = r_d*(1-g_d)*b_d;
        const scalar_t  w011 = (1-r_d)*g_d*b_d;
        const scalar_t  w111 = r_d*g_d*b_d;


        for(int i =0; i<num_channels;++i){
            output[index + width * height * i] = w000 * lut[id000 + shift * i] + w100 * lut[id100 + shift * i] + 
                                                 w010 * lut[id010 + shift * i] + w110 * lut[id110 + shift * i] + 
                                                 w001 * lut[id001 + shift * i] + w101 * lut[id101 + shift * i] + 
                                                 w011 * lut[id011 + shift * i] + w111 * lut[id111 + shift * i];
        }

    }
}


void TriLinearForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, torch::Tensor output) {
    c10::cuda::CUDAGuard device_guard(input.device());
    
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    int dim   = lut.size(2);
    int shift   = dim * dim * dim;
   
    int num_kernels = height * width;
    
    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "trilinear_cuda_forward", ([&] {
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *data_output = output[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                TriLinearForward<<<GET_BLOCKS(num_kernels),
                                              THREADS_PER_BLOCK, 0,
                                              at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, data_lut, data_image, data_output,
                    dim, shift, binsize,
                    width, height, num_channels);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void TriLinearBackward(const int nthreads,
                                  const scalar_t* __restrict__  output_grad, 
                                  const scalar_t* __restrict__ lut, 
                                  const scalar_t* __restrict__ image,                                                                    
                                  scalar_t* __restrict__  lut_grad,
                                  scalar_t* __restrict__  image_grad, 
                                  const int dim, 
                                  const int shift, 
                                  const scalar_t binsize, 
                                  const int width, 
                                  const int height, 
                                  const int num_channels) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

    const scalar_t r = image[index];
	const scalar_t g = image[index + width * height];
	const scalar_t b = image[index + width * height * 2];

	const int32_t r_id = clamp((int32_t)floor(r * (dim-1)),0, dim-2);
	const int32_t g_id = clamp((int32_t)floor(g * (dim-1)),0, dim-2);
	const int32_t b_id = clamp((int32_t)floor(b * (dim-1)),0, dim-2);

    const scalar_t  r_d = (r - binsize * r_id) / binsize;
    const scalar_t  g_d = (g - binsize * g_id) / binsize;
    const scalar_t  b_d = (b - binsize * b_id) / binsize;

    const int id000 = r_id + g_id * dim + b_id * dim * dim;
    const int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
    const int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
    const int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
    const int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
    const int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
    const int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
    const int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;

    const scalar_t  w000 = (1-r_d)*(1-g_d)*(1-b_d);
    const scalar_t  w100 = r_d*(1-g_d)*(1-b_d);
    const scalar_t  w010 = (1-r_d)*g_d*(1-b_d);
    const scalar_t  w110 = r_d*g_d*(1-b_d);
    const scalar_t  w001 = (1-r_d)*(1-g_d)*b_d;
    const scalar_t  w101 = r_d*(1-g_d)*b_d;
    const scalar_t  w011 = (1-r_d)*g_d*b_d;
    const scalar_t  w111 = r_d*g_d*b_d;

    /* derivatives: w to rd */
    const scalar_t w000_rd = - (1 - g_d) * (1 - b_d);
    const scalar_t w100_rd =   (1 - g_d) * (1 - b_d);
    const scalar_t w010_rd = - (    g_d) * (1 - b_d);
    const scalar_t w110_rd =   (    g_d) * (1 - b_d);
    const scalar_t w001_rd = - (1 - g_d) * (    b_d);
    const scalar_t w101_rd =   (1 - g_d) * (    b_d);
    const scalar_t w011_rd = - (    g_d) * (    b_d);
    const scalar_t w111_rd =   (    g_d) * (    b_d);
    
    /* derivatives: w to gd */
    const scalar_t w000_gd = - (1 - r_d) * (1 - b_d);
    const scalar_t w100_gd = - (    r_d) * (1 - b_d);
    const scalar_t w010_gd =   (1 - r_d) * (1 - b_d);
    const scalar_t w110_gd =   (    r_d) * (1 - b_d);
    const scalar_t w001_gd = - (1 - r_d) * (    b_d);
    const scalar_t w101_gd = - (    r_d) * (    b_d);
    const scalar_t w011_gd =   (1 - r_d) * (    b_d);
    const scalar_t w111_gd =   (    r_d) * (    b_d);

    /* derivatives: w to bd */
    const scalar_t w000_bd = - (1 - r_d) * (1 - g_d);
    const scalar_t w100_bd = - (    r_d) * (1 - g_d);
    const scalar_t w010_bd = - (1 - r_d) * (    g_d);
    const scalar_t w110_bd = - (    r_d) * (    g_d);
    const scalar_t w001_bd =   (1 - r_d) * (1 - g_d);
    const scalar_t w101_bd =   (    r_d) * (1 - g_d);
    const scalar_t w011_bd =   (1 - r_d) * (    g_d);
    const scalar_t w111_bd =   (    r_d) * (    g_d);

    for(int i=0;i<num_channels;++i)
    {
        scalar_t grad_o_ = output_grad[index + width * height * i];

        atomicAdd(lut_grad + id000 + shift * i, grad_o_ * w000);
        atomicAdd(lut_grad + id100 + shift * i, grad_o_ * w100);
        atomicAdd(lut_grad + id010 + shift * i, grad_o_ * w010);
        atomicAdd(lut_grad + id110 + shift * i, grad_o_ * w110);
        atomicAdd(lut_grad + id001 + shift * i, grad_o_ * w001);
        atomicAdd(lut_grad + id101 + shift * i, grad_o_ * w101);
        atomicAdd(lut_grad + id011 + shift * i, grad_o_ * w011);
        atomicAdd(lut_grad + id111 + shift * i, grad_o_ * w111);

        scalar_t grad_d = 0;
        const scalar_t lut000 = lut[id000 + shift * i];
        const scalar_t lut100 = lut[id100 + shift * i];
        const scalar_t lut010 = lut[id010 + shift * i];
        const scalar_t lut110 = lut[id110 + shift * i];
        const scalar_t lut001 = lut[id001 + shift * i];
        const scalar_t lut101 = lut[id101 + shift * i];
        const scalar_t lut011 = lut[id011 + shift * i];
        const scalar_t lut111 = lut[id111 + shift * i];
        // r
        grad_d = grad_o_ *
                (w000_rd * lut000 + w100_rd * lut100 + w010_rd * lut010 + w110_rd * lut110 +
                 w001_rd * lut001 + w101_rd * lut101 + w011_rd * lut011 + w111_rd * lut111);
        atomicAdd(image_grad + index, grad_d * 1 / binsize);
        // g
        grad_d = grad_o_ *
                (w000_gd * lut000 + w100_gd * lut100 + w010_gd * lut010 + w110_gd * lut110 +
                 w001_gd * lut001 + w101_gd * lut101 + w011_gd * lut011 + w111_gd * lut111);
        atomicAdd(image_grad + index + height * width, grad_d * 1 / binsize);
        // b
        grad_d = grad_o_ *
                (w000_bd * lut000 + w100_bd * lut100 + w010_bd * lut010 + w110_bd * lut110 +
                 w001_bd * lut001 + w101_bd * lut101 + w011_bd * lut011 + w111_bd * lut111);
        atomicAdd(image_grad + index + height * width * 2, grad_d * 1 / binsize);         
    }

    }
    }

void TriLinearBackwardLaucher(const torch::Tensor &grad_output, 
    const torch::Tensor &lut, const torch::Tensor &input, 
    torch::Tensor grad_lut, torch::Tensor grad_image) {
    
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    int dim   = lut.size(2);
    int shift   = dim * dim * dim;
   
    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "trilinear_cuda_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_image_  = grad_image[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                TriLinearBackward<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, grad_out, data_lut, data_image, 
                    grad_lut_, grad_image_, 
                    dim, shift, binsize,
                    width, height, num_channels);
            }));
        AT_CUDA_CHECK(cudaGetLastError());
    }
}
