import tensorflow as tf

zero_out_module = tf.load_op_library('lib/zero_out_op.so')
print(dir(zero_out_module))
with tf.Session(''):
    print(zero_out_module.zero_out([[1, 2], [3, 4]]).eval())
