from src.inner_product.python.inner_product_ops import inner_product_ops

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

class InnerProductOpTest(tf.test.TestCase):
    def testInnerProduct(self):
        for d in ["/cpu:0", "/gpu:0"]:
            with ops.device(d):
                with tf.Session():
                    for H in range(1, 10):
                        for W in range(1, 10):
                            weight = np.random.rand(H, W).astype(np.float32)
                            data = np.random.rand(W, 1).astype(np.float32)
                            m = inner_product_ops.inner_product(data, weight).eval()
                            self.assertAllClose(m, weight @ data)

    def testInnerProductGradient(self):
        for d in ["/cpu:0"]:
            with ops.device(d):
                with tf.Session() as sess:
                    x = tf.constant(np.asarray([1, 2]).astype(np.float32))
                    W = tf.placeholder(tf.float32, shape = (2, 2))
                    
                    Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
                    Wx_inner_product = inner_product_ops.inner_product(tf.reshape(x, [-1, 1]), W)

                    grad_W_tf = tf.gradients(Wx_tf, W)
                    grad_W_inner_product = tf.gradients(Wx_inner_product, W)

                    gradient_tf = sess.run(grad_W_tf, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float32)})
                    gradient_inner_product = sess.run(grad_W_inner_product, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float32)})

                    self.assertAllEqual(gradient_tf, gradient_inner_product)

if __name__ == "__main__":
    tf.test.main()