#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::shape_inference::DimensionHandle;

REGISTER_OP("BilateralSlice")
    .Input("in: float")
    .Input("guide: float")
    .Output("out: float")
    // .SetShapeFn(
    //     [](InferenceContext *c)
    //     {
    //         ShapeHandle grid;
    //         TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &grid));
    //         ShapeHandle guide;
    //         TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &guide));

    //         ShapeHandle output; // will be [guide[:] + grid[-1]]           
    //         TF_RETURN_IF_ERROR(c->Concatenate(guide, c->Vector(c->Dim(grid, -1)), &output));

    //         c->set_output(0, output);
    //         return Status::OK();   
    //     }
    // )
    .Doc(R"doc(Slices input in in the location defined by guide, to produce output.)doc");



REGISTER_OP("BilateralSliceGrad")
    .Input("in: float")
    .Input("guide: float")
    .Input("backprop: float")
    .Output("grid_grad: float")
    .Output("guide_grad: float")
    .SetShapeFn(
        [](InferenceContext *c)
        {
            c->set_output(0, c->input(0));
            c->set_output(1, c->input(1));
            return Status::OK();
        }
    );

REGISTER_OP("BilateralSliceApply")
    .Input("grid: float")
    .Input("guide: float")
    .Input("input: float")
    .Attr("has_offset: bool")
    .Output("out: float")
    .Doc(R"doc(Slices input in in the location defined by guide and apply it, to produce output.)doc");


REGISTER_OP("BilateralSliceApplyGrad")
    .Input("grid: float")
    .Input("guide: float")
    .Input("input: float")
    .Input("backprop: float")
    .Attr("has_offset: bool")
    .Output("grid_grad: float")
    .Output("guide_grad: float")
    .Output("input_grad: float")
    .SetShapeFn(
        [](InferenceContext *c)
        {
            c->set_output(0, c->input(0));
            c->set_output(1, c->input(1));
            c->set_output(2, c->input(2));
            return Status::OK();
        }
    );


} // namespace tensorflow