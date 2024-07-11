#include <torch/extension.h>

#include <ATen/ATen.h>


template <typename scalar_t>
inline constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template <typename scalar_t>
void TetrahedralCPUForward(const int nthreads, 
                            const scalar_t* lut, 
                            const scalar_t* image, 
                            scalar_t* output, 
                            const int dim, 
                            const int shift, 
                            const scalar_t binsize, 
                            const int width, 
                            const int height, 
                            const int num_channels) {
        for (int index=0;index<nthreads;index++) {

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
        const int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id00x = 0;
        int id0xx = 0;

        scalar_t  w000 = 0.0;
        scalar_t  w111 = 0.0;
        scalar_t  w00x = 0.0;
        scalar_t  w0xx = 0.0;

        if (r_d > g_d){
            if (g_d > b_d) { //r_d > g_d > b_d
                id00x = r_id + 1 + g_id * dim + b_id * dim * dim; //id100
                id0xx = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim; //id110

                w000 = (1-r_d);
                w00x = (r_d-g_d);
                w0xx = (g_d-b_d);
                w111 = b_d;
            }
            else if(r_d > b_d){ //r_d > b_d > g_d 
                id00x = r_id + 1 + g_id * dim + b_id * dim * dim; //id100
                id0xx = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim; //id101

                w000 = (1-r_d);
                w00x = (r_d-b_d);
                w0xx = (b_d-g_d);
                w111 = g_d;
            }
            else{ //b_d > r_d > g_d
                id00x = r_id + g_id * dim + (b_id + 1) * dim * dim; //id001
                id0xx = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim; //id101

                w000 = (1-b_d);
                w00x = (b_d-r_d);
                w0xx = (r_d-g_d);
                w111 = g_d;
            }
                
        }
        else{
            if (b_d > g_d){ //b_d > g_d > r_d  
                id00x = r_id + g_id * dim + (b_id + 1) * dim * dim; //id001
                id0xx = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim; //id011

                w000 = (1-b_d);
                w00x = (b_d-g_d);
                w0xx = (g_d-r_d);
                w111 = r_d;
            }
            else if (b_d > r_d) { //g_d > b_d > r_d 
                id00x = r_id + (g_id + 1) * dim + b_id * dim * dim; //id010
                id0xx = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim; //id011

                w000 = (1-g_d);
                w00x = (g_d-b_d);
                w0xx = (b_d-r_d);
                w111 = r_d;
            }
            else { // g_d > r_d > b_d
                id00x = r_id + (g_id + 1) * dim + b_id * dim * dim; //id010
                id0xx = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim; //id110

                w000 = (1-g_d);
                w00x = (g_d-r_d);
                w0xx = (r_d-b_d);
                w111 = b_d;
            }
        }


        for(int i =0; i<num_channels;++i){
            output[index + width * height * i] = w000 * lut[id000 + shift * i] + w00x * lut[id00x + shift * i] + 
                                                 w0xx * lut[id0xx + shift * i] + w111 * lut[id111 + shift * i];
        }

    }
}


void TetrahedralCPUForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, torch::Tensor output) {
    
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
            input.scalar_type(), "tetrahedral_cpu_forward", ([&] {
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *data_output = output[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                TetrahedralCPUForward(
                    num_kernels, data_lut, data_image, data_output,
                    dim, shift, binsize,
                    width, height, num_channels);
            }));

    }
}

template <typename scalar_t>
void TetrahedralCPUBackward(const int nthreads,
                        const scalar_t*  output_grad, 
                        const scalar_t* lut, 
                        const scalar_t*  image,                                                                    
                        scalar_t*  lut_grad,
                        scalar_t*  image_grad, 
                        const int dim, 
                        const int shift, 
                        const scalar_t binsize, 
                        const int width, 
                        const int height, 
                        const int num_channels) {
    for (int index=0;index<nthreads;index++) {

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
    const int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;
    int id00x = 0;
    int id0xx = 0;

    scalar_t  w000 = 0.0;
    scalar_t  w111 = 0.0;
    scalar_t  w00x = 0.0;
    scalar_t  w0xx = 0.0;

    scalar_t  w_d[3][4] = {{-1.0, 1.0, 0.0, 0.0},
                           {0.0, -1.0, 1.0, 0.0},
                           {0.0, 0.0, -1.0, 1.0}};
    int  w_d_idx[3] = {0, };

    if (r_d > g_d){
        if (g_d > b_d) { //r_d > g_d > b_d
            id00x = r_id + 1 + g_id * dim + b_id * dim * dim; //id100
            id0xx = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim; //id110

            w000 = (1-r_d);
            w00x = (r_d-g_d);
            w0xx = (g_d-b_d);
            w111 = b_d;

            w_d_idx[0] = 0; //r
            w_d_idx[1] = 1; //g
            w_d_idx[2] = 2; //b
        }
        else if(r_d > b_d){ //r_d > b_d > g_d 
            id00x = r_id + 1 + g_id * dim + b_id * dim * dim; //id100
            id0xx = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim; //id101

            w000 = (1-r_d);
            w00x = (r_d-b_d);
            w0xx = (b_d-g_d);
            w111 = g_d;
            
            w_d_idx[0] = 0;
            w_d_idx[1] = 2;
            w_d_idx[2] = 1;
        }
        else{ //b_d > r_d > g_d
            id00x = r_id + g_id * dim + (b_id + 1) * dim * dim; //id001
            id0xx = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim; //id101

            w000 = (1-b_d);
            w00x = (b_d-r_d);
            w0xx = (r_d-g_d);
            w111 = g_d;
            
            w_d_idx[0] = 1;
            w_d_idx[1] = 2;
            w_d_idx[2] = 0;
        }
            
    }
    else{
        if (b_d > g_d){ //b_d > g_d > r_d  
            id00x = r_id + g_id * dim + (b_id + 1) * dim * dim; //id001
            id0xx = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim; //id011

            w000 = (1-b_d);
            w00x = (b_d-g_d);
            w0xx = (g_d-r_d);
            w111 = r_d;

            w_d_idx[0] = 2;
            w_d_idx[1] = 1;
            w_d_idx[2] = 0;
        }
        else if (b_d > r_d) { //g_d > b_d > r_d 
            id00x = r_id + (g_id + 1) * dim + b_id * dim * dim; //id010
            id0xx = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim; //id011

            w000 = (1-g_d);
            w00x = (g_d-b_d);
            w0xx = (b_d-r_d);
            w111 = r_d;

            w_d_idx[0] = 2;
            w_d_idx[1] = 0;
            w_d_idx[2] = 1;
        }
        else { // g_d > r_d > b_d
            id00x = r_id + (g_id + 1) * dim + b_id * dim * dim; //id010
            id0xx = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim; //id110

            w000 = (1-g_d);
            w00x = (g_d-r_d);
            w0xx = (r_d-b_d);
            w111 = b_d;

            w_d_idx[0] = 1;
            w_d_idx[1] = 0;
            w_d_idx[2] = 2;
        }
    }

    for(int i=0;i<num_channels;++i)
    {
        scalar_t grad_o_ = output_grad[index + width * height * i];

        lut_grad[id000 + shift * i] += grad_o_ * w000;
        lut_grad[id00x + shift * i] += grad_o_ * w00x;
        lut_grad[id0xx + shift * i] += grad_o_ * w0xx;
        lut_grad[id111 + shift * i] += grad_o_ * w111;

        scalar_t grad_d = 0;
        const scalar_t lut000 = lut[id000 + shift * i];
        const scalar_t lut00x = lut[id00x + shift * i];
        const scalar_t lut0xx = lut[id0xx + shift * i];
        const scalar_t lut111 = lut[id111 + shift * i];
        // r
        grad_d = grad_o_ *
                (w_d[w_d_idx[0]][0] * lut000 + w_d[w_d_idx[0]][1] * lut00x +
                 w_d[w_d_idx[0]][2] * lut0xx + w_d[w_d_idx[0]][3] * lut111);
        image_grad[index] += grad_d * 1 / binsize;
        // g
        grad_d = grad_o_ *
                (w_d[w_d_idx[1]][0] * lut000 + w_d[w_d_idx[1]][1] * lut00x +
                 w_d[w_d_idx[1]][2] * lut0xx + w_d[w_d_idx[1]][3] * lut111);
        image_grad[index + height * width] += grad_d * 1 / binsize;
        // b
        grad_d = grad_o_ *
                (w_d[w_d_idx[2]][0] * lut000 + w_d[w_d_idx[2]][1] * lut00x +
                 w_d[w_d_idx[2]][2] * lut0xx + w_d[w_d_idx[2]][3] * lut111);
        image_grad[index + height * width * 2] += grad_d * 1 / binsize;         
    }

    }
    }

void TetrahedralCPUBackwardLaucher(const torch::Tensor &grad_output, 
    const torch::Tensor &lut, const torch::Tensor &input, 
    torch::Tensor grad_lut, torch::Tensor grad_image) {

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
            input.scalar_type(), "tetrahedral_cpu_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_image_  = grad_image[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                TetrahedralCPUBackward(
                    num_kernels, grad_out, data_lut, data_image, 
                    grad_lut_, grad_image_, 
                    dim, shift, binsize,
                    width, height, num_channels);
            }));
    }
}
