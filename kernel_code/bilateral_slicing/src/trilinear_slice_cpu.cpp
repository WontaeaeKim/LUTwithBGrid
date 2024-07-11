#include <torch/extension.h>

#include <ATen/ATen.h>

template <typename scalar_t>
inline constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template <typename scalar_t>
void TriLinearCPUSliceForward(const int nthreads, 
                                const scalar_t* grid, 
                                const scalar_t* image, 
                                scalar_t* output, 
                                const int dim, 
                                const int shift, 
                                const scalar_t binsize, 
                                const int width, 
                                const int height,
								const int num_channels,
                                const int grid_per_ch) {
        for (int index=0;index<nthreads;index++) {

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
        const int id100 = (x_id + 1) + (y_id    ) * dim;
        const int id010 = (x_id    ) + (y_id + 1) * dim;
        const int id110 = (x_id + 1) + (y_id + 1) * dim;
        const int id001 = (x_id    ) + (y_id    ) * dim;
        const int id101 = (x_id + 1) + (y_id    ) * dim;
        const int id011 = (x_id    ) + (y_id + 1) * dim;
        const int id111 = (x_id + 1) + (y_id + 1) * dim;
		
		const int id000_r = id000 + (r_id    ) * dim * dim; 
        const int id100_r = id100 + (r_id    ) * dim * dim;
        const int id010_r = id010 + (r_id    ) * dim * dim;
        const int id110_r = id110 + (r_id    ) * dim * dim;
        const int id001_r = id001 + (r_id + 1) * dim * dim;
        const int id101_r = id101 + (r_id + 1) * dim * dim;
        const int id011_r = id011 + (r_id + 1) * dim * dim;
        const int id111_r = id111 + (r_id + 1) * dim * dim;
				
		const int id000_g = id000 + (g_id    ) * dim * dim; 
        const int id100_g = id100 + (g_id    ) * dim * dim;
        const int id010_g = id010 + (g_id    ) * dim * dim;
        const int id110_g = id110 + (g_id    ) * dim * dim;
        const int id001_g = id001 + (g_id + 1) * dim * dim;
        const int id101_g = id101 + (g_id + 1) * dim * dim;
        const int id011_g = id011 + (g_id + 1) * dim * dim;
        const int id111_g = id111 + (g_id + 1) * dim * dim;
		
		const int id000_b = id000 + (b_id    ) * dim * dim; 
        const int id100_b = id100 + (b_id    ) * dim * dim;
        const int id010_b = id010 + (b_id    ) * dim * dim;
        const int id110_b = id110 + (b_id    ) * dim * dim;
        const int id001_b = id001 + (b_id + 1) * dim * dim;
        const int id101_b = id101 + (b_id + 1) * dim * dim;
        const int id011_b = id011 + (b_id + 1) * dim * dim;
        const int id111_b = id111 + (b_id + 1) * dim * dim;
		
        const scalar_t  w000 = (1 - x_d) * (1 - y_d);
        const scalar_t  w100 = (    x_d) * (1 - y_d);
        const scalar_t  w010 = (1 - x_d) * (    y_d);
        const scalar_t  w110 = (    x_d) * (    y_d);
        const scalar_t  w001 = (1 - x_d) * (1 - y_d);
        const scalar_t  w101 = (    x_d) * (1 - y_d);
        const scalar_t  w011 = (1 - x_d) * (    y_d);
        const scalar_t  w111 = (    x_d) * (    y_d);
		
		const scalar_t  w000_r = w000 * (1 - r_d);
        const scalar_t  w100_r = w100 * (1 - r_d);
        const scalar_t  w010_r = w010 * (1 - r_d);
        const scalar_t  w110_r = w110 * (1 - r_d);
        const scalar_t  w001_r = w001 * (    r_d);
        const scalar_t  w101_r = w101 * (    r_d);
        const scalar_t  w011_r = w011 * (    r_d);
        const scalar_t  w111_r = w111 * (    r_d);
		
		const scalar_t  w000_g = w000 * (1 - g_d);
        const scalar_t  w100_g = w100 * (1 - g_d);
        const scalar_t  w010_g = w010 * (1 - g_d);
        const scalar_t  w110_g = w110 * (1 - g_d);
        const scalar_t  w001_g = w001 * (    g_d);
        const scalar_t  w101_g = w101 * (    g_d);
        const scalar_t  w011_g = w011 * (    g_d);
        const scalar_t  w111_g = w111 * (    g_d);
		
		const scalar_t  w000_b = w000 * (1 - b_d);
        const scalar_t  w100_b = w100 * (1 - b_d);
        const scalar_t  w010_b = w010 * (1 - b_d);
        const scalar_t  w110_b = w110 * (1 - b_d);
        const scalar_t  w001_b = w001 * (    b_d);
        const scalar_t  w101_b = w101 * (    b_d);
        const scalar_t  w011_b = w011 * (    b_d);
        const scalar_t  w111_b = w111 * (    b_d);


        for(int i = 0; i < grid_per_ch; ++i){
            output[index + width * height * (i + (grid_per_ch) * 0)] = w000_r * grid[id000_r + shift * (i + grid_per_ch * 0)] + 
			                                                                w100_r * grid[id100_r + shift * (i + grid_per_ch * 0)] + 
																	        w010_r * grid[id010_r + shift * (i + grid_per_ch * 0)] + 
																			w110_r * grid[id110_r + shift * (i + grid_per_ch * 0)] + 
																	        w001_r * grid[id001_r + shift * (i + grid_per_ch * 0)] + 
																			w101_r * grid[id101_r + shift * (i + grid_per_ch * 0)] + 
																	        w011_r * grid[id011_r + shift * (i + grid_per_ch * 0)] + 
																			w111_r * grid[id111_r + shift * (i + grid_per_ch * 0)];
																	 
			output[index + width * height * (i + (grid_per_ch ) * 1)] = w000_g * grid[id000_g + shift * (i + grid_per_ch * 1)] + 
			                                                                w100_g * grid[id100_g + shift * (i + grid_per_ch * 1)] + 
																	        w010_g * grid[id010_g + shift * (i + grid_per_ch * 1)] + 
																			w110_g * grid[id110_g + shift * (i + grid_per_ch * 1)] + 
																	        w001_g * grid[id001_g + shift * (i + grid_per_ch * 1)] + 
																			w101_g * grid[id101_g + shift * (i + grid_per_ch * 1)] + 
																	        w011_g * grid[id011_g + shift * (i + grid_per_ch * 1)] + 
																			w111_g * grid[id111_g + shift * (i + grid_per_ch * 1)];
																			
			output[index + width * height * (i + (grid_per_ch ) * 2)] = w000_b * grid[id000_b + shift * (i + grid_per_ch * 2)] + 
			                                                                w100_b * grid[id100_b + shift * (i + grid_per_ch * 2)] + 
																	        w010_b * grid[id010_b + shift * (i + grid_per_ch * 2)] + 
																			w110_b * grid[id110_b + shift * (i + grid_per_ch * 2)] + 
																	        w001_b * grid[id001_b + shift * (i + grid_per_ch * 2)] + 
																			w101_b * grid[id101_b + shift * (i + grid_per_ch * 2)] + 
																	        w011_b * grid[id011_b + shift * (i + grid_per_ch * 2)] + 
																			w111_b * grid[id111_b + shift * (i + grid_per_ch * 2)];
        }
		/*
		output[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 0)] = r;
		output[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 1)] = g;
		output[index + width * height * (grid_per_ch + (grid_per_ch + 1) * 2)] = b;
        */
    }
}


void TriLinearCPUSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor output) {
    
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
            input.scalar_type(), "trilinear_cpu_forward", ([&] {
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid = grid[elt].data_ptr<scalar_t>();
                scalar_t *data_output = output[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                TriLinearCPUSliceForward(
                    num_kernels, data_grid, data_image, data_output,
                    dim, shift, binsize,
                    width, height, num_channels, grid_per_ch);
            }));
    }
}

template <typename scalar_t>
void TriLinearCPUSliceBackward(const int nthreads,
                                  const scalar_t* output_grad, 
                                  const scalar_t* grid, 
                                  const scalar_t* image,                                                                    
                                  scalar_t* grid_grad,
                                  scalar_t* image_grad, 
                                  const int dim, 
                                  const int shift, 
                                  const scalar_t binsize, 
                                  const int width, 
                                  const int height, 
                                  const int num_channels, 
                                  const int grid_per_ch) {
        for (int index=0;index<nthreads;index++) {

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
        const int id100 = (x_id + 1) + (y_id    ) * dim;
        const int id010 = (x_id    ) + (y_id + 1) * dim;
        const int id110 = (x_id + 1) + (y_id + 1) * dim;
        const int id001 = (x_id    ) + (y_id    ) * dim;
        const int id101 = (x_id + 1) + (y_id    ) * dim;
        const int id011 = (x_id    ) + (y_id + 1) * dim;
        const int id111 = (x_id + 1) + (y_id + 1) * dim;
		
		const int id000_r = id000 + (r_id    ) * dim * dim; 
        const int id100_r = id100 + (r_id    ) * dim * dim;
        const int id010_r = id010 + (r_id    ) * dim * dim;
        const int id110_r = id110 + (r_id    ) * dim * dim;
        const int id001_r = id001 + (r_id + 1) * dim * dim;
        const int id101_r = id101 + (r_id + 1) * dim * dim;
        const int id011_r = id011 + (r_id + 1) * dim * dim;
        const int id111_r = id111 + (r_id + 1) * dim * dim;
				
		const int id000_g = id000 + (g_id    ) * dim * dim; 
        const int id100_g = id100 + (g_id    ) * dim * dim;
        const int id010_g = id010 + (g_id    ) * dim * dim;
        const int id110_g = id110 + (g_id    ) * dim * dim;
        const int id001_g = id001 + (g_id + 1) * dim * dim;
        const int id101_g = id101 + (g_id + 1) * dim * dim;
        const int id011_g = id011 + (g_id + 1) * dim * dim;
        const int id111_g = id111 + (g_id + 1) * dim * dim;
		
		const int id000_b = id000 + (b_id    ) * dim * dim; 
        const int id100_b = id100 + (b_id    ) * dim * dim;
        const int id010_b = id010 + (b_id    ) * dim * dim;
        const int id110_b = id110 + (b_id    ) * dim * dim;
        const int id001_b = id001 + (b_id + 1) * dim * dim;
        const int id101_b = id101 + (b_id + 1) * dim * dim;
        const int id011_b = id011 + (b_id + 1) * dim * dim;
        const int id111_b = id111 + (b_id + 1) * dim * dim;
		
        const scalar_t  w000 = (1 - x_d) * (1 - y_d);
        const scalar_t  w100 = (    x_d) * (1 - y_d);
        const scalar_t  w010 = (1 - x_d) * (    y_d);
        const scalar_t  w110 = (    x_d) * (    y_d);
        const scalar_t  w001 = (1 - x_d) * (1 - y_d);
        const scalar_t  w101 = (    x_d) * (1 - y_d);
        const scalar_t  w011 = (1 - x_d) * (    y_d);
        const scalar_t  w111 = (    x_d) * (    y_d);

        const scalar_t  w000_r = w000 * (1 - r_d);
        const scalar_t  w100_r = w100 * (1 - r_d);
        const scalar_t  w010_r = w010 * (1 - r_d);
        const scalar_t  w110_r = w110 * (1 - r_d);
        const scalar_t  w001_r = w001 * (    r_d);
        const scalar_t  w101_r = w101 * (    r_d);
        const scalar_t  w011_r = w011 * (    r_d);
        const scalar_t  w111_r = w111 * (    r_d);
		
		const scalar_t  w000_g = w000 * (1 - g_d);
        const scalar_t  w100_g = w100 * (1 - g_d);
        const scalar_t  w010_g = w010 * (1 - g_d);
        const scalar_t  w110_g = w110 * (1 - g_d);
        const scalar_t  w001_g = w001 * (    g_d);
        const scalar_t  w101_g = w101 * (    g_d);
        const scalar_t  w011_g = w011 * (    g_d);
        const scalar_t  w111_g = w111 * (    g_d);
		
		const scalar_t  w000_b = w000 * (1 - b_d);
        const scalar_t  w100_b = w100 * (1 - b_d);
        const scalar_t  w010_b = w010 * (1 - b_d);
        const scalar_t  w110_b = w110 * (1 - b_d);
        const scalar_t  w001_b = w001 * (    b_d);
        const scalar_t  w101_b = w101 * (    b_d);
        const scalar_t  w011_b = w011 * (    b_d);
        const scalar_t  w111_b = w111 * (    b_d);		

		/* derivatives: w to rd, gd, bd */
		const scalar_t w000_zd = - w000;
		const scalar_t w100_zd = - w100;
		const scalar_t w010_zd = - w010;
		const scalar_t w110_zd = - w110;
		const scalar_t w001_zd =   w001;
		const scalar_t w101_zd =   w101;
		const scalar_t w011_zd =   w011;
		const scalar_t w111_zd =   w111;
		
        scalar_t grad_o_r = 0;
        scalar_t grad_o_g = 0;
        scalar_t grad_o_b = 0;

		for(int i=0;i<grid_per_ch;++i)
		{
			grad_o_r = output_grad[index + width * height * (i + (grid_per_ch) * 0)];
			grad_o_g = output_grad[index + width * height * (i + (grid_per_ch) * 1)];
			grad_o_b = output_grad[index + width * height * (i + (grid_per_ch) * 2)];

			grid_grad[id000_r + shift * (i + grid_per_ch * 0)] += grad_o_r * w000_r;
            grid_grad[id100_r + shift * (i + grid_per_ch * 0)] += grad_o_r * w100_r;
            grid_grad[id010_r + shift * (i + grid_per_ch * 0)] += grad_o_r * w010_r;
            grid_grad[id110_r + shift * (i + grid_per_ch * 0)] += grad_o_r * w110_r;
            grid_grad[id001_r + shift * (i + grid_per_ch * 0)] += grad_o_r * w001_r;
            grid_grad[id101_r + shift * (i + grid_per_ch * 0)] += grad_o_r * w101_r;
            grid_grad[id011_r + shift * (i + grid_per_ch * 0)] += grad_o_r * w011_r;
            grid_grad[id111_r + shift * (i + grid_per_ch * 0)] += grad_o_r * w111_r;
			
			grid_grad[id000_g + shift * (i + grid_per_ch * 1)] += grad_o_g * w000_g;
            grid_grad[id100_g + shift * (i + grid_per_ch * 1)] += grad_o_g * w100_g;
            grid_grad[id010_g + shift * (i + grid_per_ch * 1)] += grad_o_g * w010_g;
            grid_grad[id110_g + shift * (i + grid_per_ch * 1)] += grad_o_g * w110_g;
            grid_grad[id001_g + shift * (i + grid_per_ch * 1)] += grad_o_g * w001_g;
            grid_grad[id101_g + shift * (i + grid_per_ch * 1)] += grad_o_g * w101_g;
            grid_grad[id011_g + shift * (i + grid_per_ch * 1)] += grad_o_g * w011_g;
            grid_grad[id111_g + shift * (i + grid_per_ch * 1)] += grad_o_g * w111_g;
			
			grid_grad[id000_b + shift * (i + grid_per_ch * 2)] += grad_o_b * w000_b;
            grid_grad[id100_b + shift * (i + grid_per_ch * 2)] += grad_o_b * w100_b;
            grid_grad[id010_b + shift * (i + grid_per_ch * 2)] += grad_o_b * w010_b;
            grid_grad[id110_b + shift * (i + grid_per_ch * 2)] += grad_o_b * w110_b;
            grid_grad[id001_b + shift * (i + grid_per_ch * 2)] += grad_o_b * w001_b;
            grid_grad[id101_b + shift * (i + grid_per_ch * 2)] += grad_o_b * w101_b;
            grid_grad[id011_b + shift * (i + grid_per_ch * 2)] += grad_o_b * w011_b;
            grid_grad[id111_b + shift * (i + grid_per_ch * 2)] += grad_o_b * w111_b;

			scalar_t grad_d = 0;
			scalar_t grid000 = grid[id000_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid100 = grid[id100_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid010 = grid[id010_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid110 = grid[id110_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid001 = grid[id001_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid101 = grid[id101_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid011 = grid[id011_r + shift * (i + grid_per_ch * 0)];
			scalar_t grid111 = grid[id111_r + shift * (i + grid_per_ch * 0)];
			// r
			grad_d = grad_o_r *
					(w000_zd * grid000 + w100_zd * grid100 + w010_zd * grid010 + w110_zd * grid110 +
					 w001_zd * grid001 + w101_zd * grid101 + w011_zd * grid011 + w111_zd * grid111);
			image_grad[index] += grad_d * 1 / binsize;
			// g
			grid000 = grid[id000_g + shift * (i + grid_per_ch * 1)];
			grid100 = grid[id100_g + shift * (i + grid_per_ch * 1)];
			grid010 = grid[id010_g + shift * (i + grid_per_ch * 1)];
			grid110 = grid[id110_g + shift * (i + grid_per_ch * 1)];
			grid001 = grid[id001_g + shift * (i + grid_per_ch * 1)];
			grid101 = grid[id101_g + shift * (i + grid_per_ch * 1)];
			grid011 = grid[id011_g + shift * (i + grid_per_ch * 1)];
			grid111 = grid[id111_g + shift * (i + grid_per_ch * 1)];
			grad_d = grad_o_g *
					(w000_zd * grid000 + w100_zd * grid100 + w010_zd * grid010 + w110_zd * grid110 +
					 w001_zd * grid001 + w101_zd * grid101 + w011_zd * grid011 + w111_zd * grid111);
			image_grad[index + height * width] += grad_d * 1 / binsize;
			// b
			grid000 = grid[id000_b + shift * (i + grid_per_ch * 2)];
			grid100 = grid[id100_b + shift * (i + grid_per_ch * 2)];
			grid010 = grid[id010_b + shift * (i + grid_per_ch * 2)];
			grid110 = grid[id110_b + shift * (i + grid_per_ch * 2)];
			grid001 = grid[id001_b + shift * (i + grid_per_ch * 2)];
			grid101 = grid[id101_b + shift * (i + grid_per_ch * 2)];
			grid011 = grid[id011_b + shift * (i + grid_per_ch * 2)];
			grid111 = grid[id111_b + shift * (i + grid_per_ch * 2)];
			grad_d = grad_o_b *
					(w000_zd * grid000 + w100_zd * grid100 + w010_zd * grid010 + w110_zd * grid110 +
					 w001_zd * grid001 + w101_zd * grid101 + w011_zd * grid011 + w111_zd * grid111);
			image_grad[index + height * width * 2] += grad_d * 1 / binsize;         
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

void TriLinearCPUSliceBackwardLaucher(const torch::Tensor &grad_output, 
    const torch::Tensor &grid, const torch::Tensor &input, 
    torch::Tensor grad_grid, torch::Tensor grad_image) {

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
            input.scalar_type(), "trilinear_cpu_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid = grid[elt].data_ptr<scalar_t>();
                scalar_t *grad_image_  = grad_image[elt].data_ptr<scalar_t>();
                scalar_t *grad_grid_ = grad_grid[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                TriLinearCPUSliceBackward(
                    num_kernels, grad_out, data_grid, data_image, 
                    grad_grid_, grad_image_, 
                    dim, shift, binsize,
                    width, height, num_channels, grid_per_ch);
            }));

    }
}
