#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "time_two.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
namespace functor
{

typedef Eigen::GpuDevice GPUDevice;

// Define the cuda kernel
template <typename T>
__global__ void TimeTwoCudaKernel(const int size, const T *in, T *out)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        out[i] = 2 * in[i];
}

// GPU specialization of the actual computation
// Define the GPU implementation that launches the CUDA kernel
template <typename T>
struct TimeTwoFunctor<GPUDevice, T>
{
    void operator()(const GPUDevice &d, int size, const T *in, T *out)
    {
        // launch the cuda kernel
        int block_count = 1024;
        int thread_per_block = 20;
        TimeTwoCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
    }
};

// Explicitly instantiate functors for the types of OpKernels registered
template struct TimeTwoFunctor<GPUDevice, float>;
template struct TimeTwoFunctor<GPUDevice, int32>;

    
} // namespace functor
    
} // namespace tensorflow


#endif  // GOOGLe_CUDA