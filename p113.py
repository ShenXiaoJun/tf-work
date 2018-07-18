# encoding:utf-8
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有声明滑动平均模型时只有一个变量v，所以下面的语句只会输出"v:0"
for variables in tf.all_variables():
    print variables.name

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.all_variables())
# 在声明滑动平均模型之后，Tensorflow会自动生成一个影子变量
# v/ExponentialMoving Average.于是下面的语句会输出
# "v:0"和"v/ExponentialMovingAverage:0"
for variables in tf.all_variables():
    print variables.name

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    sess.run(tf.assign(v,10))
    sess.run(maintain_average_op)
    # 保存时，Tensorflow会将v:0和v/ExponentialMovingAverage:0两个变量都存下来
    saver.save(sess,"/home/shenxj/tf-work/model/p113-model.ckpt")
    print sess.run([v, ema.average(v)])