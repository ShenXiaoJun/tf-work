# encoding:utf-8
import tensorflow as tf

# 定义变量相加的计算
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()
# 通过export_meta_graph函数导出Tensorflow计算图的元图，并保存为json格式
saver.export_meta_graph("/home/shenxj/tf-work/model/p117-model.ckpt.meda.json",
                        as_text=True)
