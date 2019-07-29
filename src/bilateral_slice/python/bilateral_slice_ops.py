import tensorflow as tf
from tensorflow.python.framework import ops

bilateral_slice_ops = tf.load_op_library('lib/bilateral_slice_ops.so')

# ops
bilateral_slice = bilateral_slice_ops.bilateral_slice
bilateral_slice_apply = bilateral_slice_ops.bilateral_slice_apply

#register gradients
@ops.RegisterGradient('BilateralSlice')
def _bilateral_slice_grad(op, grad):
    grid_tensor = op.inputs[0]
    guide_tensor = op.inputs[1]
    return bilateral_slice_ops.bilateral_slice_grad(grid_tensor, guide_tensor, grad)


@ops.RegisterGradient('BilateralSliceApply')
def _bilateral_slice_grad(op, grad):
    grid_tensor = op.inputs[0]
    guide_tensor = op.inputs[1]
    input_tensor = op.inputs[2]
    has_offset = op.get_attr('has_offset')
    return bilateral_slice_ops.bilateral_slice_apply_grad(
        grid_tensor, guide_tensor, input_tensor, grad, has_offset=has_offset)