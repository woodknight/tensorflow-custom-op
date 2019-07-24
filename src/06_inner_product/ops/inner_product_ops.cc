#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

namespace tensorflow
{
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("InnerProduct")
    .Input("input: float")
    .Input("weights: float")
    .Output("inner_product: float")
    .SetShapeFn([](InferenceContext *c) 
    {
        ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

        ShapeHandle weight_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));

        DimensionHandle output_rows = c->Dim(weight_shape, 0);
        DimensionHandle input_rows = c->Dim(input_shape, 0);
        DimensionHandle weight_cols = c->Dim(weight_shape, 1);
        DimensionHandle merged;
        TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));

        c->set_output(0, c->Matrix(output_rows, 1));
        return Status::OK();
    });

REGISTER_OP("InnerProductGrad")
    .Input("grad: float32")
    .Input("input: float32")
    .Input("weights: float32")
    .Output("grad_input: float32")
    .Output("grad_weights: float32");

} // namespace tensorflow