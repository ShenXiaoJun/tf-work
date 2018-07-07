#encoding:utf-8
import tensorflow as tf

import time
import datetime


a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

g = tf.Graph()
#指定计算运行的设备
with g.device('/gpu:0'):
    start = datetime.datetime.now()
    result = a + b
    end = datetime.datetime.now()
    print (end - start).microseconds

with g.device('/cpu'):
    start = datetime.datetime.now()
    result = a + b
    end = datetime.datetime.now()
    print (end - start).microseconds

with g.device('/cpu:0'):
    start = datetime.datetime.now()
    result = a + b
    end = datetime.datetime.now()
    print (end - start).microseconds