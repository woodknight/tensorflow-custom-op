from src.time_two.python.time_two_ops import time_two_ops

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

class TimeTwoTest(tf.test.TestCase):
    def testTimeTwo(self):
        with self.test_session():
            with ops.device("/gpu:0"):
                self.assertAllEqual(
                    time_two_ops.time_two([[1, 2], [3, 4]]).eval(), np.array([[2, 4], [6, 8]])
                )

if __name__ == "__main__":
    tf.test.main()