import tensorflow as tf
from tensorflow.python.framework import ops

inner_product = tf.load_op_library('lib/inner_product_ops.so')

@ops.RegisterGradient("InnerProduct")
def _inner_product_grad(op, grad):
    return inner_product.inner_product_grad(grad, op.inputs[0], op.inputs[1])