# encoding:utf-8
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)

# 通过使用variable_to_restore函数可以直接生成上面代码中提供的字典
# {"v/ExponentialMovingAverage": v}
# 以下代码会输出
# {'v/ExponentialMovingAverage': <tensorflow.python.ops.variables.Variable
# object at xxxxxxx>}
# 其中后面的Variable类就代表了变量v
print ema.variables_to_restore()

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "/home/shenxj/tf-work/model/p113-model.ckpt")
    print sess.run(v)