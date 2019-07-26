#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "inner_product.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow
{

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor
{

template <typename T>
struct InnerProductFunctor<CPUDevice, T>
{
    // CPU specialization of the computation
    void operator()(const CPUDevice &d, int weight_height, int weight_width, const T *data, const T *weight, T *out)
    {
        for(int y = 0; y < weight_height; y++)
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
};

// OpKernel definition
// template parameter <t> is the datatype of the tensors
template <typename Device, typename T>
class InnerProductOp : public OpKernel
{
public:
    explicit InnerProductOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensors
        const Tensor &data_tensor = context->input(0);
        const Tensor &weight_tensor = context->input(1);

        // check the input tensor shapes
        const TensorShape &data_shape = data_tensor.shape();
        const TensorShape &weight_shape = weight_tensor.shape();

        CHECK_EQ(data_shape.dims(), 2);
        CHECK_EQ(data_shape.dim_size(1), 1);

        CHECK_EQ(weight_shape.dims(), 2);
        CHECK_EQ(weight_shape.dim_size(1), data_shape.dim_size(0));

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(weight_shape.dim_size(0));
        output_shape.AddDim(1);

        // Create the output tensor
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

        // Do the computation
        OP_REQUIRES(context, data_tensor.NumElements() <= tensorflow::kint32max, 
            errors::InvalidArgument("too many elements in input tensor"));
        OP_REQUIRES(context, weight_tensor.NumElements() <= tensorflow::kint32max, 
            errors::InvalidArgument("too many elements in input tensor"));

        InnerProductFunctor<Device, T>()(
            context->eigen_device<Device>(),
            weight_shape.dim_size(0),
            weight_shape.dim_size(1),
            data_tensor.flat<T>().data(),
            weight_tensor.flat<T>().data(),
            output_tensor->flat<T>().data()
        );
    }
};

// Register the CPU kernels
#define REGISTER_CPU(T) \
    REGISTER_KERNEL_BUILDER(    \
        Name("InnerProduct").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        InnerProductOp<CPUDevice, T>    \
    );

REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels
#define REGISTER_GPU(T) \
    REGISTER_KERNEL_BUILDER(    \
        Name("InnerProduct").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        InnerProductOp<GPUDevice, T>    \
    );

REGISTER_GPU(int32);
REGISTER_GPU(float);

} // namespace functor
} // namespace tensorflow


