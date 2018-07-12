# encoding:utf-8
import tensorflow as tf

with tf.Session() as sess:
    v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print tf.clip_by_value(v, 2.5, 4.5).eval()

    v = tf.constant([1.0, 2.0, 3.0]).eval()
    print tf.log(v)

    v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    print (v1 * v2).eval()
    print tf.matmul(v1, v2).eval()