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
struct InnerProductFunctor<CPUDevice, T>    // partial template specialization
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
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
    extern template struct InnerProductFunctor<GPUDevice, T>; \
    REGISTER_KERNEL_BUILDER(    \
        Name("InnerProduct").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        InnerProductOp<GPUDevice, T>    \
    );

REGISTER_GPU(float);
REGISTER_GPU(int32);

#endif

//============================ OP Gradients ==========================
template <typename T>
struct InnerProductGradFunctor<CPUDevice, T>
{
    void operator()(const CPUDevice &d, int weight_height, int weight_width, 
                    const T *data, const T *weight, const T *grad_output,
                    T *grad_data, T *grad_weight)
    {   
        // y = Wx + b
        // dC/dx = W^T (dC/dy)
        // dC/dW = (dC/dy) x^T

        // CPU specialization of the operation
        // grad_data
        for(int i = 0; i < weight_width; i++)
        {
            T sum = 0;
            for(int j = 0; j < weight_height; j++)
            {
                sum += weight[j * weight_width + i] * grad_output[j];
            }
            grad_data[i] = sum;
        }        

        // grad_weight
        for(int i = 0; i < weight_height; i++)
        {
            for(int j = 0; j < weight_width; j++)
            {
                grad_weight[i * weight_width + j] = grad_output[i] * data[j];
            }
        }
    }
};


// OpKernel definition
// template parameter <t> is the datatype of the tensors
template <typename Device, typename T>
class InnerProductGradOp : public OpKernel
{
public:
    explicit InnerProductGradOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // grab the input tensors
        const Tensor &data_tensor = context->input(0);
        const Tensor &weight_tensor = context->input(1);
        const Tensor &grad_output_tensor = context->input(2);

        // check the input tensor shapes
        const TensorShape &data_shape = data_tensor.shape();
        const TensorShape &weight_shape = weight_tensor.shape();
        const TensorShape &grad_output_shape = grad_output_tensor.shape();

        CHECK_EQ(data_shape.dims(), 2);
        CHECK_EQ(data_shape.dim_size(1), 1);

        CHECK_EQ(grad_output_shape.dims(), 2);
        CHECK_EQ(grad_output_shape.dim_size(1), 1);

        CHECK_EQ(weight_shape.dims(), 2);
        CHECK_EQ(weight_shape.dim_size(0), grad_output_shape.dim_size(0));
        CHECK_EQ(weight_shape.dim_size(1), data_shape.dim_size(0));

        // create the output tensor
        Tensor *grad_data_tensor = nullptr;
        Tensor *grad_weight_tensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, data_shape, &grad_data_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, weight_shape, &grad_weight_tensor));

        // Do the computation
        OP_REQUIRES(context, data_tensor.NumElements() <= tensorflow::kint32max, 
            errors::InvalidArgument("too many elements in input tensor"));
        OP_REQUIRES(context, weight_tensor.NumElements() <= tensorflow::kint32max, 
            errors::InvalidArgument("too many elements in input tensor"));

        InnerProductGradFunctor<Device, T>()(
            context->eigen_device<Device>(),
            weight_shape.dim_size(0),
            weight_shape.dim_size(1),
            data_tensor.flat<T>().data(),
            weight_tensor.flat<T>().data(),
            grad_output_tensor.flat<T>().data(),
            grad_data_tensor->flat<T>().data(),
            grad_weight_tensor->flat<T>().data()
        );
    }
};

// register the CPU kernel
#define REGISTER_CPU_GRAD(T) \
    REGISTER_KERNEL_BUILDER(    \
        Name("InnerProductGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        InnerProductGradOp<CPUDevice, T>    \
    );

REGISTER_CPU_GRAD(float);
REGISTER_CPU_GRAD(int32);
} // namespace functor

} // namespace tensorflow


