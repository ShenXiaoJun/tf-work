# encoding:utf-8
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量重命名将原来变量v的滑动平均直接赋值给v
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "/home/shenxj/tf-work/model/p113-model.ckpt")
    print sess.run(v)