#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::shape_inference::DimensionHandle;

REGISTER_OP("InnerProduct")
    .Attr("T: {int32, float}")
    .Input("data: T")
    .Input("weight: T")
    .Output("out: T")
    .SetShapeFn(
        [](InferenceContext *c)
        {
            ShapeHandle data_shape, weight_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &data_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
            DimensionHandle output_rows = c->Dim(weight_shape, 0);

            c->set_output(0, c->Matrix(output_rows, 1));
            return Status::OK();
        }
    );

REGISTER_OP("InnerProductGrad")
    .Attr("T: {int32, float}")
    .Input("data: T")
    .Input("weight: T")
    .Input("grad_output: T")
    .Output("grad_data: T")
    .Output("grad_weight: T")
    .SetShapeFn(
        [](InferenceContext *c)
        {
            c->set_output(0, c->input(0));
            c->set_output(1, c->input(1));
            return Status::OK();
        }
    );

    
} // namespace tensorflow
