#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow
{
class InnerProductOp : public OpKernel
{
public:
    /// \brief Constructor.
    /// \param context
    explicit InnerProductOp(OpKernelConstruction *context) : OpKernel(context) {}

    /// \brief Compute the inner product.
    /// \param context
    void Compute(OpKernelContext *context) override
    {
        // some checks to be sure ...
        DCHECK_EQ(2, context->num_inputs());

        // get the input tensor
        const Tensor &input = context->input(0);

        // get the weight tensor
        const Tensor &weights = context->input(1);

        // check shapes of input and weights
        const TensorShape &input_shape = input.shape();
        const TensorShape &weights_shape = weights.shape();

        // check input is a standing vector
        DCHECK_EQ(input_shape.dims(), 2);
        DCHECK_EQ(input_shape.dim_size(1), 1);

        // check weights is matrix of correct size
        DCHECK_EQ(weights_shape.dims(), 2);
        DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(weights_shape.dim_size(0));
        output_shape.AddDim(1);

        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        // get the corresponding Eigen tensors for data access
        auto input_tensor = input.matrix<float>();
        auto weights_tensor = weights.matrix<float>();
        auto output_tensor = output->matrix<float>();

        for (int i = 0; i < output->shape().dim_size(0); i++)
        {
            output_tensor(i, 0) = 0;
            for (int j = 0; j < weights.shape().dim_size(1); j++)
            {
                output_tensor(i, 0) += weights_tensor(i, j) * input_tensor(j, 0);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("InnerProduct").Device(DEVICE_CPU), InnerProductOp);

class InnerProductGradOp : public OpKernel
{
public:
    /// \brief Constructor.
    /// \param context
    explicit InnerProductGradOp(OpKernelConstruction *context) : OpKernel(context) {}

    /// \brief Compute the inner product gradients.
    /// \param context
    void Compute(OpKernelContext *context) override
    {
        // output and grad is provided as input
        DCHECK_EQ(3, context->num_inputs());

        // get the gradient tensor
        const Tensor &grad = context->input(0);

        // get the original input tensor
        const Tensor &input = context->input(1);

        // get the weight tensor
        const Tensor &weights = context->input(2);

        // create input shape (inferred from the additional attribute `n`)
        TensorShape input_shape = input.shape();
        TensorShape weights_shape = weights.shape();

        DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
        DCHECK_EQ(weights_shape.dim_size(0), grad.shape().dim_size(0));

        // create output tensors
        Tensor *grad_input = NULL;
        Tensor *grad_weights = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));

        // get the Eigen tensors for data access
        auto grad_tensor = grad.matrix<float>();
        auto weights_tensor = weights.matrix<float>();
        auto input_tensor = input.matrix<float>();
        auto grad_input_tensor = grad_input->matrix<float>();
        auto grad_weights_tensor = grad_weights->matrix<float>();

        // doing it manually for ismplicity
        for (int i = 0; i < weights_shape.dim_size(0); i++)
        {
            grad_input_tensor(i, 0) = 0;
            for (int j = 0; j < grad.shape().dim_size(0); j++)
            {
                grad_input_tensor(i, 0) += grad_tensor(j, 0) * weights_tensor(j, i);
            }
        }

        for (int i = 0; i < weights_shape.dim_size(0); i++)
        {
            for (int j = 0; j < weights_shape.dim_size(1); j++)
            {
                grad_weights_tensor(i, j) = grad_tensor(i, 0) * input_tensor(j, 0);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("InnerProductGrad").Device(DEVICE_CPU), InnerProductGradOp);

} // namespace tensorflow
