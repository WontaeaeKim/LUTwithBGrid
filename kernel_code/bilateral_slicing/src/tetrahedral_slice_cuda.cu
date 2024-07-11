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
__global__ void TetrahedralSliceForward(const int nthreads, 
                                const scalar_t* __restrict__ grid, 
                                const scalar_t* __restrict__ image, 
                                scalar_t* __restrict__ output, 
                                const int dim, 
                                const int shift, 
                                const scalar_t binsize, 
                                const int width, 
                                const int height,
								const int num_channels,
                                const int grid_per_ch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

        const int x_ = index % width;
	    const int y_ = index / width;

        const scalar_t x = x_ / (width -1);
	    const scalar_t y = y_ / (height -1);

	    const scalar_t r = image[index];
        const scalar_t g = image[index + width * height];
        const scalar_t b = image[index + width * height * 2];
        
        const int32_t x_id = clamp((int32_t)floor(x * (dim-1)),0, dim-2);
        const int32_t y_id = clamp((int32_t)floor(y * (dim-1)),0, dim-2);

	    const int32_t r_id = clamp((int32_t)floor(r * (dim-1)),0, dim-2);
	    const int32_t g_id = clamp((int32_t)floor(g * (dim-1)),0, dim-2);
	    const int32_t b_id = clamp((int32_t)floor(b * (dim-1)),0, dim-2);

        const scalar_t  x_d = (x - binsize * x_id) / binsize;
        const scalar_t  y_d = (y - binsize * y_id) / binsize;

        const scalar_t  r_d = (r - binsize * r_id) / binsize;
        const scalar_t  g_d = (g - binsize * g_id) / binsize;
        const scalar_t  b_d = (b - binsize * b_id) / binsize;

        const int id000 = (x_id    ) + (y_id    ) * dim;
        const int id111 = (x_id + 1) + (y_id + 1) * dim;
		
		const int id000_r = id000 + (r_id    ) * dim * dim; 
        const int id111_r = id111 + (r_id + 1) * dim * dim;
        int id00x_r = 0;
        int id0xx_r = 0;
				
		const int id000_g = id000 + (g_id    ) * dim * dim; 
        const int id111_g = id111 + (g_id + 1) * dim * dim;
        int id00x_g = 0;
        int id0xx_g = 0;
		
		const int id000_b = id000 + (b_id    ) * dim * dim;
        const int id111_b = id111 + (b_id + 1) * dim * dim;
        int id00x_b = 0;
        int id0xx_b = 0;
				
		scalar_t  w000_r = 0.0;
        scalar_t  w111_r = 0.0;
        scalar_t  w00x_r = 0.0;
        scalar_t  w0xx_r = 0.0;

        scalar_t  w000_g = 0.0;
        scalar_t  w111_g = 0.0;
        scalar_t  w00x_g = 0.0;
        scalar_t  w0xx_g = 0.0;

        scalar_t  w000_b = 0.0;
        scalar_t  w111_b = 0.0;
        scalar_t  w00x_b = 0.0;
        scalar_t  w0xx_b = 0.0;

        if (x_d > y_d){
            if (y_d > r_d) { //r_d > g_d > b_d
                id00x_r = (x_id + 1) + (y_id    ) * dim + r_id * dim * dim; //id100
                id0xx_r = (x_id + 1) + (y_id + 1) * dim + r_id * dim * dim; //id110

                w000_r = (1-x_d);
                w00x_r = (x_d-y_d);
                w0xx_r = (y_d-r_d);
                w111_r = r_d;
            }
            else if(y_d > r_d){ //r_d > b_d > g_d 
                id00x_r = (x_id + 1) + (y_id    ) * dim + (r_id    ) * dim * dim; //id100
                id0xx_r = (x_id + 1) + (y_id    ) * dim + (r_id + 1) * dim * dim; //id101

                w000_r = (1-x_d);
                w00x_r = (x_d-r_d);
                w0xx_r = (r_d-y_d);
                w111_r = y_d;
            }
            else{ //b_d > r_d > g_d
                id00x_r = (x_id    ) + (y_id    ) * dim + (r_id + 1) * dim * dim; //id001
                id0xx_r = (x_id + 1) + (y_id    ) * dim + (r_id + 1) * dim * dim; //id101

                w000_r = (1-r_d);
                w00x_r = (r_d-x_d);
                w0xx_r = (x_d-y_d);
                w111_r = y_d;
            }

            if (y_d > g_d) { //r_d > g_d > b_d
                id00x_g = (x_id + 1) + (y_id    ) * dim + (g_id    ) * dim * dim; //id100
                id0xx_g = (x_id + 1) + (y_id + 1) * dim + (g_id    ) * dim * dim; //id110

                w000_g = (1-x_d);
                w00x_g = (x_d-y_d);
                w0xx_g = (y_d-g_d);
                w111_g = g_d;
            }
            else if(y_d > g_d){ //r_d > b_d > g_d 
                id00x_g = (x_id + 1) + (y_id    ) * dim + (g_id    ) * dim * dim; //id100
                id0xx_g = (x_id + 1) + (y_id    ) * dim + (g_id + 1) * dim * dim; //id101

                w000_g = (1-x_d);
                w00x_g = (x_d-g_d);
                w0xx_g = (g_d-y_d);
                w111_g = y_d;
            }
            else{ //b_d > r_d > g_d
                id00x_g = (x_id    ) + (y_id    ) * dim + (g_id + 1) * dim * dim; //id001
                id0xx_g = (x_id + 1) + (y_id    ) * dim + (g_id + 1) * dim * dim; //id101

                w000_g = (1-g_d);
                w00x_g = (g_d-x_d);
                w0xx_g = (x_d-y_d);
                w111_g = y_d;
            }

            if (y_d > b_d) { //r_d > g_d > b_d
                id00x_b = (x_id + 1) + (y_id    ) * dim + (b_id    ) * dim * dim; //id100
                id0xx_b = (x_id + 1) + (y_id + 1) * dim + (b_id    ) * dim * dim; //id110

                w000_b = (1-x_d);
                w00x_b = (x_d-y_d);
                w0xx_b = (y_d-b_d);
                w111_b = b_d;
            }
            else if(y_d > b_d){ //r_d > b_d > g_d 
                id00x_b = (x_id + 1) + (y_id    ) * dim + (b_id    ) * dim * dim; //id100
                id0xx_b = (x_id + 1) + (y_id    ) * dim + (b_id + 1) * dim * dim; //id101

                w000_b = (1-x_d);
                w00x_b = (x_d-b_d);
                w0xx_b = (b_d-y_d);
                w111_b = y_d;
            }
            else{ //b_d > r_d > g_d
                id00x_b = (x_id    ) + (y_id    ) * dim + (b_id + 1) * dim * dim; //id001
                id0xx_b = (x_id + 1) + (y_id    ) * dim + (b_id + 1) * dim * dim; //id101

                w000_b = (1-b_d);
                w00x_b = (b_d-x_d);
                w0xx_b = (x_d-y_d);
                w111_b = y_d;
            }
                
        }
        else{
            if (r_d > y_d){ //b_d > g_d > r_d  
                id00x_r = (x_id    ) + (y_id    ) * dim + (r_id + 1) * dim * dim; //id001
                id0xx_r = (x_id    ) + (y_id + 1) * dim + (r_id + 1) * dim * dim; //id011

                w000_r = (1-r_d);
                w00x_r = (r_d-y_d);
                w0xx_r = (y_d-x_d);
                w111_r = x_d;
            }
            else if (r_d > x_d) { //g_d > b_d > r_d 
                id00x_r = (x_id    ) + (y_id + 1) * dim + (r_id    ) * dim * dim; //id010
                id0xx_r = (x_id    ) + (y_id + 1) * dim + (r_id + 1) * dim * dim; //id011

                w000_r = (1-y_d);
                w00x_r = (y_d-r_d);
                w0xx_r = (r_d-x_d);
                w111_r = x_d;
            }
            else { // g_d > r_d > b_d
                id00x_r = (x_id    ) + (y_id + 1) * dim + (r_id    ) * dim * dim; //id010
                id0xx_r = (x_id + 1) + (y_id + 1) * dim + (r_id    ) * dim * dim; //id110

                w000_r = (1-y_d);
                w00x_r = (y_d-x_d);
                w0xx_r = (x_d-r_d);
                w111_r = r_d;
            }

            if (g_d > y_d){ //b_d > g_d > r_d  
                id00x_g = (x_id    ) + (y_id    ) * dim + (g_id + 1) * dim * dim; //id001
                id0xx_g = (x_id    ) + (y_id + 1) * dim + (g_id + 1) * dim * dim; //id011

                w000_g = (1-g_d);
                w00x_g = (g_d-y_d);
                w0xx_g = (y_d-x_d);
                w111_g = x_d;
            }
            else if (g_d > x_d) { //g_d > b_d > r_d 
                id00x_g = (x_id    ) + (y_id + 1) * dim + (g_id    ) * dim * dim; //id010
                id0xx_g = (x_id    ) + (y_id + 1) * dim + (g_id + 1) * dim * dim; //id011

                w000_g = (1-y_d);
                w00x_g = (y_d-g_d);
                w0xx_g = (g_d-x_d);
                w111_g = x_d;
            }
            else { // g_d > r_d > b_d
                id00x_g = (x_id    ) + (y_id + 1) * dim + (g_id    ) * dim * dim; //id010
                id0xx_g = (x_id + 1) + (y_id + 1) * dim + (g_id    ) * dim * dim; //id110

                w000_g = (1-y_d);
                w00x_g = (y_d-x_d);
                w0xx_g = (x_d-g_d);
                w111_g = g_d;
            }

            if (b_d > y_d){ //b_d > g_d > r_d  
                id00x_b = (x_id    ) + (y_id    ) * dim + (b_id + 1) * dim * dim; //id001
                id0xx_b = (x_id    ) + (y_id + 1) * dim + (b_id + 1) * dim * dim; //id011

                w000_b = (1-b_d);
                w00x_b = (b_d-y_d);
                w0xx_b = (y_d-x_d);
                w111_b = x_d;
            }
            else if (b_d > x_d) { //g_d > b_d > r_d 
                id00x_b = (x_id    ) + (y_id + 1) * dim + (b_id    ) * dim * dim; //id010
                id0xx_b = (x_id    ) + (y_id + 1) * dim + (b_id + 1) * dim * dim; //id011

                w000_b = (1-y_d);
                w00x_b = (y_d-b_d);
                w0xx_b = (b_d-x_d);
                w111_b = x_d;
            }
            else { // g_d > r_d > b_d
                id00x_b = (x_id    ) + (y_id + 1) * dim + (b_id    ) * dim * dim; //id010
                id0xx_b = (x_id + 1) + (y_id + 1) * dim + (b_id    ) * dim * dim; //id110

                w000_b = (1-y_d);
                w00x_b = (y_d-x_d);
                w0xx_b = (x_d-b_d);
                w111_b = b_d;
            }
        }

        for(int i = 0; i < grid_per_ch; ++i){
            output[index + width * height * (i + (grid_per_ch) * 0)] = w000_r * grid[id000_r + shift * (i + grid_per_ch * 0)] + 
			                                                           w00x_r * grid[id00x_r + shift * (i + grid_per_ch * 0)] + 
																	   w0xx_r * grid[id0xx_r + shift * (i + grid_per_ch * 0)] + 
																	   w111_r * grid[id111_r + shift * (i + grid_per_ch * 0)];
																	 
			output[index + width * height * (i + (grid_per_ch) * 1)] = w000_g * grid[id000_g + shift * (i + grid_per_ch * 1)] + 
			                                                           w00x_g * grid[id00x_g + shift * (i + grid_per_ch * 1)] + 
																	   w0xx_g * grid[id0xx_g + shift * (i + grid_per_ch * 1)] + 
																	   w111_g * grid[id111_g + shift * (i + grid_per_ch * 1)];
																			
			output[index + width * height * (i + (grid_per_ch) * 2)] = w000_b * grid[id000_b + shift * (i + grid_per_ch * 2)] + 
			                                                           w00x_b * grid[id00x_b + shift * (i + grid_per_ch * 2)] + 
																	   w0xx_b * grid[id0xx_b + shift * (i + grid_per_ch * 2)] + 
																	   w111_b * grid[id111_b + shift * (i + grid_per_ch * 2)];
        }
		/*
		output[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 0)] = r;
		output[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 1)] = g;
		output[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 2)] = b;
        */
    }
}


void TetrahedralSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor output) {
    c10::cuda::CUDAGuard device_guard(input.device());
    
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = input.size(1);
	int grid_channels = grid.size(1);
    int dim   = grid.size(2);
    int shift   = dim * dim * dim;
	
	int grid_per_ch = grid_channels / num_channels;
   
    int num_kernels = height * width;
    
    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "tetrahedral_cuda_forward", ([&] {
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid = grid[elt].data_ptr<scalar_t>();
                scalar_t *data_output = output[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                TetrahedralSliceForward<<<GET_BLOCKS(num_kernels),
                                              THREADS_PER_BLOCK, 0,
                                              at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, data_grid, data_image, data_output,
                    dim, shift, binsize,
                    width, height, num_channels, grid_per_ch);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void TetrahedralSliceBackward(const int nthreads,
                                  const scalar_t* __restrict__  output_grad, 
                                  const scalar_t* __restrict__ grid, 
                                  const scalar_t* __restrict__ image,                                                                    
                                  scalar_t* __restrict__  grid_grad,
                                  scalar_t* __restrict__  image_grad, 
                                  const int dim, 
                                  const int shift, 
                                  const scalar_t binsize, 
                                  const int width, 
                                  const int height, 
                                  const int num_channels, 
                                  const int grid_per_ch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

        const int x_ = index % width;
	    const int y_ = index / width;

        const scalar_t x = x_ / (width -1);
	    const scalar_t y = y_ / (height -1);

	    const scalar_t r = image[index];
        const scalar_t g = image[index + width * height];
        const scalar_t b = image[index + width * height * 2];
        
        const int32_t x_id = clamp((int32_t)floor(x * (dim-1)),0, dim-2);
        const int32_t y_id = clamp((int32_t)floor(y * (dim-1)),0, dim-2);

	    const int32_t r_id = clamp((int32_t)floor(r * (dim-1)),0, dim-2);
	    const int32_t g_id = clamp((int32_t)floor(g * (dim-1)),0, dim-2);
	    const int32_t b_id = clamp((int32_t)floor(b * (dim-1)),0, dim-2);

        const scalar_t  x_d = (x - binsize * x_id) / binsize;
        const scalar_t  y_d = (y - binsize * y_id) / binsize;

        const scalar_t  r_d = (r - binsize * r_id) / binsize;
        const scalar_t  g_d = (g - binsize * g_id) / binsize;
        const scalar_t  b_d = (b - binsize * b_id) / binsize;

        const int id000 = (x_id    ) + (y_id    ) * dim;
        const int id111 = (x_id + 1) + (y_id + 1) * dim;
		
		const int id000_r = id000 + (r_id    ) * dim * dim; 
        const int id111_r = id111 + (r_id + 1) * dim * dim;
        int id00x_r = 0;
        int id0xx_r = 0;
				
		const int id000_g = id000 + (g_id    ) * dim * dim; 
        const int id111_g = id111 + (g_id + 1) * dim * dim;
        int id00x_g = 0;
        int id0xx_g = 0;
		
		const int id000_b = id000 + (b_id    ) * dim * dim;
        const int id111_b = id111 + (b_id + 1) * dim * dim;
        int id00x_b = 0;
        int id0xx_b = 0;
		
		scalar_t  w000_r = 0.0;
        scalar_t  w111_r = 0.0;
        scalar_t  w00x_r = 0.0;
        scalar_t  w0xx_r = 0.0;

        scalar_t  w000_g = 0.0;
        scalar_t  w111_g = 0.0;
        scalar_t  w00x_g = 0.0;
        scalar_t  w0xx_g = 0.0;

        scalar_t  w000_b = 0.0;
        scalar_t  w111_b = 0.0;
        scalar_t  w00x_b = 0.0;
        scalar_t  w0xx_b = 0.0;

        scalar_t  w_d[3][4] = {{-1.0, 1.0, 0.0, 0.0},
                               {0.0, -1.0, 1.0, 0.0},
                               {0.0, 0.0, -1.0, 1.0}};
        int  w_d_idx_r = 0;
        int  w_d_idx_g = 0;
        int  w_d_idx_b = 0;

        if (x_d > y_d){
            if (y_d > r_d) { //r_d > g_d > b_d
                id00x_r = (x_id + 1) + (y_id    ) * dim + (r_id    ) * dim * dim; //id100
                id0xx_r = (x_id + 1) + (y_id + 1) * dim + (r_id    ) * dim * dim; //id110

                w000_r = (1-x_d);
                w00x_r = (x_d-y_d);
                w0xx_r = (y_d-r_d);
                w111_r = r_d;

                w_d_idx_r = 2;
            }
            else if(y_d > r_d){ //r_d > b_d > g_d 
                id00x_r = (x_id + 1) + (y_id    ) * dim + (r_id    ) * dim * dim; //id100
                id0xx_r = (x_id + 1) + (y_id    ) * dim + (r_id + 1) * dim * dim; //id101

                w000_r = (1-x_d);
                w00x_r = (x_d-r_d);
                w0xx_r = (r_d-y_d);
                w111_r = y_d;

                w_d_idx_r = 1;
            }
            else{ //b_d > r_d > g_d
                id00x_r = (x_id    ) + (y_id    ) * dim + (r_id + 1) * dim * dim; //id001
                id0xx_r = (x_id + 1) + (y_id    ) * dim + (r_id + 1) * dim * dim; //id101

                w000_r = (1-r_d);
                w00x_r = (r_d-x_d);
                w0xx_r = (x_d-y_d);
                w111_r = y_d;

                w_d_idx_r = 0;
            }

            if (y_d > g_d) { //r_d > g_d > b_d
                id00x_g = (x_id + 1) + (y_id    ) * dim + (g_id    ) * dim * dim; //id100
                id0xx_g = (x_id + 1) + (y_id + 1) * dim + (g_id    ) * dim * dim; //id110

                w000_g = (1-x_d);
                w00x_g = (x_d-y_d);
                w0xx_g = (y_d-g_d);
                w111_g = g_d;

                w_d_idx_g = 2;
            }
            else if(y_d > g_d){ //r_d > b_d > g_d 
                id00x_g = (x_id + 1) + (y_id    ) * dim + (g_id    ) * dim * dim; //id100
                id0xx_g = (x_id + 1) + (y_id    ) * dim + (g_id + 1) * dim * dim; //id101

                w000_g = (1-x_d);
                w00x_g = (x_d-g_d);
                w0xx_g = (g_d-y_d);
                w111_g = y_d;

                w_d_idx_g = 1;
            }
            else{ //b_d > r_d > g_d
                id00x_g = (x_id    ) + (y_id    ) * dim + (g_id + 1) * dim * dim; //id001
                id0xx_g = (x_id + 1) + (y_id    ) * dim + (g_id + 1) * dim * dim; //id101

                w000_g = (1-g_d);
                w00x_g = (g_d-x_d);
                w0xx_g = (x_d-y_d);
                w111_g = y_d;

                w_d_idx_g = 0;
            }

            if (y_d > b_d) { //r_d > g_d > b_d
                id00x_b = (x_id + 1) + (y_id    ) * dim + (b_id    ) * dim * dim; //id100
                id0xx_b = (x_id + 1) + (y_id + 1) * dim + (b_id    ) * dim * dim; //id110

                w000_b = (1-x_d);
                w00x_b = (x_d-y_d);
                w0xx_b = (y_d-b_d);
                w111_b = b_d;

                w_d_idx_b = 2;
            }
            else if(y_d > b_d){ //r_d > b_d > g_d 
                id00x_b = (x_id + 1) + (y_id    ) * dim + (b_id    ) * dim * dim; //id100
                id0xx_b = (x_id + 1) + (y_id    ) * dim + (b_id + 1) * dim * dim; //id101

                w000_b = (1-x_d);
                w00x_b = (x_d-b_d);
                w0xx_b = (b_d-y_d);
                w111_b = y_d;

                w_d_idx_b = 1;
            }
            else{ //b_d > r_d > g_d
                id00x_b = (x_id    ) + (y_id    ) * dim + (b_id + 1) * dim * dim; //id001
                id0xx_b = (x_id + 1) + (y_id    ) * dim + (b_id + 1) * dim * dim; //id101

                w000_b = (1-b_d);
                w00x_b = (b_d-x_d);
                w0xx_b = (x_d-y_d);
                w111_b = y_d;

                w_d_idx_b = 0;
            }
                
        }
        else{
            if (r_d > y_d){ //b_d > g_d > r_d  
                id00x_r = (x_id    ) + (y_id    ) * dim + (r_id + 1) * dim * dim; //id001
                id0xx_r = (x_id    ) + (y_id + 1) * dim + (r_id + 1) * dim * dim; //id011

                w000_r = (1-r_d);
                w00x_r = (r_d-y_d);
                w0xx_r = (y_d-x_d);
                w111_r = x_d;

                w_d_idx_r = 0;
            }
            else if (r_d > x_d) { //g_d > b_d > r_d 
                id00x_r = (x_id    ) + (y_id + 1) * dim + (r_id    ) * dim * dim; //id010
                id0xx_r = (x_id    ) + (y_id + 1) * dim + (r_id + 1) * dim * dim; //id011

                w000_r = (1-y_d);
                w00x_r = (y_d-r_d);
                w0xx_r = (r_d-x_d);
                w111_r = x_d;

                w_d_idx_r = 1;
            }
            else { // g_d > r_d > b_d
                id00x_r = (x_id    ) + (y_id + 1) * dim + (r_id    ) * dim * dim; //id010
                id0xx_r = (x_id + 1) + (y_id + 1) * dim + (r_id    ) * dim * dim; //id110

                w000_r = (1-y_d);
                w00x_r = (y_d-x_d);
                w0xx_r = (x_d-r_d);
                w111_r = r_d;

                w_d_idx_r = 2;
            }

            if (g_d > y_d){ //b_d > g_d > r_d  
                id00x_g = (x_id    ) + (y_id    ) * dim + (g_id + 1) * dim * dim; //id001
                id0xx_g = (x_id    ) + (y_id + 1) * dim + (g_id + 1) * dim * dim; //id011

                w000_g = (1-g_d);
                w00x_g = (g_d-y_d);
                w0xx_g = (y_d-x_d);
                w111_g = x_d;

                w_d_idx_g = 0;
            }
            else if (g_d > x_d) { //g_d > b_d > r_d 
                id00x_g = (x_id    ) + (y_id + 1) * dim + (g_id    ) * dim * dim; //id010
                id0xx_g = (x_id    ) + (y_id + 1) * dim + (g_id + 1) * dim * dim; //id011

                w000_g = (1-y_d);
                w00x_g = (y_d-g_d);
                w0xx_g = (g_d-x_d);
                w111_g = x_d;

                w_d_idx_g = 1;
            }
            else { // g_d > r_d > b_d
                id00x_g = (x_id    ) + (y_id + 1) * dim + (g_id    ) * dim * dim; //id010
                id0xx_g = (x_id + 1) + (y_id + 1) * dim + (g_id    ) * dim * dim; //id110

                w000_g = (1-y_d);
                w00x_g = (y_d-x_d);
                w0xx_g = (x_d-g_d);
                w111_g = g_d;

                w_d_idx_g = 2;
            }

            if (b_d > y_d){ //b_d > g_d > r_d  
                id00x_b = (x_id    ) + (y_id    ) * dim + (b_id + 1) * dim * dim; //id001
                id0xx_b = (x_id    ) + (y_id + 1) * dim + (b_id + 1) * dim * dim; //id011

                w000_b = (1-b_d);
                w00x_b = (b_d-y_d);
                w0xx_b = (y_d-x_d);
                w111_b = x_d;

                w_d_idx_b = 0;
            }
            else if (b_d > x_d) { //g_d > b_d > r_d 
                id00x_b = (x_id    ) + (y_id + 1) * dim + (b_id    ) * dim * dim; //id010
                id0xx_b = (x_id    ) + (y_id + 1) * dim + (b_id + 1) * dim * dim; //id011

                w000_b = (1-y_d);
                w00x_b = (y_d-b_d);
                w0xx_b = (b_d-x_d);
                w111_b = x_d;

                w_d_idx_b = 1;
            }
            else { // g_d > r_d > b_d
                id00x_b = (x_id    ) + (y_id + 1) * dim + (b_id    ) * dim * dim; //id010
                id0xx_b = (x_id + 1) + (y_id + 1) * dim + (b_id    ) * dim * dim; //id110

                w000_b = (1-y_d);
                w00x_b = (y_d-x_d);
                w0xx_b = (x_d-b_d);
                w111_b = b_d;

                w_d_idx_b = 2;
            }
        }	

        scalar_t grad_o_r = 0;
        scalar_t grad_o_g = 0;
        scalar_t grad_o_b = 0;

		for(int i=0;i<grid_per_ch;++i)
		{
			grad_o_r = output_grad[index + width * height * (i + (grid_per_ch) * 0)];
			grad_o_g = output_grad[index + width * height * (i + (grid_per_ch) * 1)];
			grad_o_b = output_grad[index + width * height * (i + (grid_per_ch) * 2)];

			atomicAdd(grid_grad + id000_r + shift * (i + grid_per_ch * 0), grad_o_r * w000_r);
			atomicAdd(grid_grad + id00x_r + shift * (i + grid_per_ch * 0), grad_o_r * w00x_r);
			atomicAdd(grid_grad + id0xx_r + shift * (i + grid_per_ch * 0), grad_o_r * w0xx_r);
			atomicAdd(grid_grad + id111_r + shift * (i + grid_per_ch * 0), grad_o_r * w111_r);
			
			atomicAdd(grid_grad + id000_g + shift * (i + grid_per_ch * 1), grad_o_g * w000_g);
			atomicAdd(grid_grad + id00x_g + shift * (i + grid_per_ch * 1), grad_o_g * w00x_g);
			atomicAdd(grid_grad + id0xx_g + shift * (i + grid_per_ch * 1), grad_o_g * w0xx_g);
			atomicAdd(grid_grad + id111_g + shift * (i + grid_per_ch * 1), grad_o_g * w111_g);
			
			atomicAdd(grid_grad + id000_b + shift * (i + grid_per_ch * 2), grad_o_b * w000_b);
			atomicAdd(grid_grad + id00x_b + shift * (i + grid_per_ch * 2), grad_o_b * w00x_b);
			atomicAdd(grid_grad + id0xx_b + shift * (i + grid_per_ch * 2), grad_o_b * w0xx_b);
			atomicAdd(grid_grad + id111_b + shift * (i + grid_per_ch * 2), grad_o_b * w111_b);

			scalar_t grad_d = 0;
			// r
            scalar_t grid000 = grid[id000_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid00x = grid[id00x_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid0xx = grid[id0xx_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid111 = grid[id111_r + shift * (i + grid_per_ch * 0)];
			grad_d = grad_o_r *
					(w_d[w_d_idx_r][0] * grid000 + w_d[w_d_idx_r][1] * grid00x +
					 w_d[w_d_idx_r][2] * grid0xx + w_d[w_d_idx_r][3] * grid111);
			atomicAdd(image_grad + index, grad_d * 1 / binsize);
			// g
			grid000 = grid[id000_g + shift * (i + grid_per_ch * 1)];
			grid00x = grid[id00x_g + shift * (i + grid_per_ch * 1)];
			grid0xx = grid[id0xx_g + shift * (i + grid_per_ch * 1)];
			grid111 = grid[id111_g + shift * (i + grid_per_ch * 1)];
			grad_d = grad_o_g *
					(w_d[w_d_idx_g][0] * grid000 + w_d[w_d_idx_g][1] * grid00x +
					 w_d[w_d_idx_g][2] * grid0xx + w_d[w_d_idx_g][3] * grid111);
			atomicAdd(image_grad + index + height * width, grad_d * 1 / binsize);
			// b
			grid000 = grid[id000_b + shift * (i + grid_per_ch * 2)];
			grid00x = grid[id00x_b + shift * (i + grid_per_ch * 2)];
			grid0xx = grid[id0xx_b + shift * (i + grid_per_ch * 2)];
			grid111 = grid[id111_b + shift * (i + grid_per_ch * 2)];
			grad_d = grad_o_b *
					(w_d[w_d_idx_b][0] * grid000 + w_d[w_d_idx_b][1] * grid00x +
					 w_d[w_d_idx_b][2] * grid0xx + w_d[w_d_idx_b][3] * grid111);
			atomicAdd(image_grad + index + height * width * 2, grad_d * 1 / binsize);         
		}
        /*
        grad_o_r = output_grad[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 0)];
        grad_o_g = output_grad[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 1)];
        grad_o_b = output_grad[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 2)];
        atomicAdd(image_grad + index, grad_o_r * 1 / binsize);
        atomicAdd(image_grad + index + height * width, grad_o_g * 1 / binsize);
        atomicAdd(image_grad + index + height * width * 2, grad_o_b * 1 / binsize);
        */
    }
    }

void TetrahedralSliceBackwardLaucher(const torch::Tensor &grad_output, 
    const torch::Tensor &grid, const torch::Tensor &input, 
    torch::Tensor grad_grid, torch::Tensor grad_image) {
    
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = input.size(1);
	int grid_channels = grid.size(1);
    int dim   = grid.size(2);
    int shift   = dim * dim * dim;
	
	int grid_per_ch = grid_channels / num_channels;
   
    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "tetrahedral_cuda_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid = grid[elt].data_ptr<scalar_t>();
                scalar_t *grad_image_  = grad_image[elt].data_ptr<scalar_t>();
                scalar_t *grad_grid_ = grad_grid[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                TetrahedralSliceBackward<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, grad_out, data_grid, data_image, 
                    grad_grid_, grad_image_, 
                    dim, shift, binsize,
                    width, height, num_channels, grid_per_ch);
            }));
        AT_CUDA_CHECK(cudaGetLastError());
    }
}
