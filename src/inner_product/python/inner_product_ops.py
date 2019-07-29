import tensorflow as tf
from tensorflow.python.framework import ops

inner_product_ops = tf.load_op_library('lib/inner_product_ops.so')

@ops.RegisterGradient('InnerProduct')
def _inner_product_grad(op, grad_output):
    data, weight = op.inputs
    grad_output_tensor = ops.convert_to_tensor(grad_output, name="grad_output")
    return inner_product_ops.inner_product_grad(data, weight, grad_output_tensor)

