import tensorflow as tf
import numpy as np
from utils.general_utils import test_all_close
# input: n x m tensor
# output: n x m tensor
def softmax(x):
    exp = tf.exp(x)
    sum = tf.reduce_sum(exp, axis=1)
    softmax = tf.divide(exp, sum)
    return softmax

# input: n x m tensor
# output: n x m tensor
def cross_entropy_loss(y, yhat):
    n = tf.to_float(tf.shape(y)[0])
    y = tf.to_float(y)
    log_yhat = tf.log(yhat)
    prod = tf.multiply(y, log_yhat)
    neg_CE = tf.reduce_sum(prod, axis=1)
    neg_CE_batch = tf.reduce_sum(neg_CE, axis=0)
    neg_CE_batch = tf.divide(neg_CE_batch, n)
    CE = tf.multiply(tf.constant(-1.0), neg_CE_batch)
    return CE

def test_softmax_basic():
    """
        Some simple tests of softmax to get you started.
        Warning: these are not exhaustive.
        """

    test1 = softmax(tf.constant(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32))
    with tf.Session() as sess:
        test1 = sess.run(test1)
    test_all_close("Softmax test 1", test1, np.array([[0.26894142, 0.73105858],
                                                      [0.26894142, 0.73105858]]))

    test2 = softmax(tf.constant(np.array([[-1001, -1002]]), dtype=tf.float32))
    with tf.Session() as sess:
        test2 = sess.run(test2)
    test_all_close("Softmax test 2", test2, np.array([[0.73105858, 0.26894142]]))
    print("Basic (non-exhaustive) softmax tests pass\n")

def test_cross_entropy_loss_basic():
    """
        Some simple tests of cross_entropy_loss to get you started.
        Warning: these are not exhaustive.
        """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

    test1 = cross_entropy_loss(
        tf.constant(y, dtype=tf.int32),
        tf.constant(yhat, dtype=tf.float32))
    with tf.Session() as sess:
        test1 = sess.run(test1)
    expected = -1*np.log(.5)
    test_all_close("Cross-entropy test 1", test1, expected)

    print("Basic (non-exhaustive) cross-entropy tests pass")

if __name__ == "__main__":
    test_softmax_basic()
    test_cross_entropy_loss_basic()