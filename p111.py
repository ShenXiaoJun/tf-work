# encoding:utf-8
import tensorflow as tf

# 声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0,shape=[1],name="v1"))
v2 = tf.Variable(tf.constant(2.0,shape=[1],name="v2"))
result = v1 + v2

init_op = tf.initialize_all_variables()
# 声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 将模型保存到./model/p111-model.ckpt文件
    saver.save(sess, "/home/shenxj/tf-work/model/p111-model.ckpt")