import numpy as np
import tensorflow as tf

from python.inner_product.inner_product_ops import *

class InnerProductTest(tf.test.TestCase):
    def testInnerProduct(self):
        with tf.Session():
            result = inner_product.inner_product([[1], [2]], [[1, 2], [3, 4]]).eval()
            self.assertEqual(result.shape[0], 2)
            self.assertEqual(result[0], 5)
            self.assertEqual(result[1], 11)

    def testInnerProductGradient(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, shape=(2))
            W = tf.constant(np.asarray([[1, 2], [3, 4]]).astype(np.float32))

            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_inner_product = inner_product.inner_product(tf.reshape(x, [-1, 1]), W)

            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_inner_product = tf.gradients(Wx_inner_product, x)

            gradient_tf = sess.run(grad_x_tf, feed_dict={x: np.asarray([1, 2]).astype(np.float32)})
            gradient_inner_product = sess.run(grad_x_inner_product, feed_dict={x:np.asarray([1,2]).astype(np.float32)})

            self.assertEqual(gradient_tf[0][0], gradient_inner_product[0][0])
            self.assertEqual(gradient_tf[0][1], gradient_inner_product[0][1])

if __name__ == "__main__":
    tf.test.main()