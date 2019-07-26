from src.inner_product.python.inner_product_ops import inner_product_ops

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

with tf.Session() as sess:
    with ops.device("/cpu:0"):
        # m = inner_product_ops.inner_product([[1.0], [2.0]], [[1.0, 2.0], [3.0, 4.0]]).eval()
        m = inner_product_ops.inner_product([[1], [2]], [[1, 2], [3, 4]]).eval()
    print(m)