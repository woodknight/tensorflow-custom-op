#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{

using ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("TimeTwo")
    .Attr("T: {int32, float}")
    .Input("in: T")
    .Output("out: T")
    .SetShapeFn(
        [](InferenceContext *c)
        {
            c->set_output(0, c->input(1));
            return Status::OK();
        }
    );

} // namespace tensorflow