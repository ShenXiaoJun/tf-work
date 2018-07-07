#encoding:utf-8
import tensorflow as tf
#tf.constan是一个计算，这个计算的结果为一个张量，保存在变量a中
a = tf.constant([1.0,2.0], name="a")
b = tf.constant([2.0,3.0], name="b")

result = tf.add(a, b, name="add")
print result