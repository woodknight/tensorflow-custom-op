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

if __name__ == "__main__":
    tf.test.main()