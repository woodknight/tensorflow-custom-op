#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "inner_product.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
namespace functor
{

typedef Eigen::GpuDevice GPUDevice;

// define the cuda kernel
template<typename T>
__global__ void InnerProductCudaKernel(int weight_height, int weight_width, const T *data, const T *weight, T *out)
{
    for(int y = blockIdx.x * blockDim.x + threadIdx.x; y < weight_height; y += blockDim.x * gridDim.x)
    {
        T sum = 0;
        for(int x = 0; x < weight_width; x++)
        {          
            int offset = y * weight_width + x;
            sum += weight[offset] * data[x];
        }
        out[y] = sum;
    }
}

// GPU specialization of the actual computation
// Define the GPU implementation that launches the CUDA kernel
template <typename T>
struct InnerProductFunctor<GPUDevice, T>
{
    void operator()(const GPUDevice &d, int weight_height, int weight_width, const T *data, const T *weight, T *out)
    {
        // launch the cuda kernel
        int block_count = 1024;
        int thread_per_block = 20;
        InnerProductCudaKernel<T><<<block_count, thread_per_block>>>(weight_height, weight_width, data, weight, out);
    }
};

// Explicitly instantiate functors for the types of OpKernels registered
template struct InnerProductFunctor<GPUDevice, int32>;
template struct InnerProductFunctor<GPUDevice, float>;
    
} // namespace functor

    
} // namespace tensorflow



#endif // GOOGLE_CUDA