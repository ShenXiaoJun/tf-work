# encoding:utf-8
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播的函数
import p151_mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "/home/shenxj/tf-work/model/"
MODEL_NAME = "p150_model.ckpt"

def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(
        tf.float32,[
            BATCH_SIZE,                         # 第一维表示一个batch中样例的个数
            p151_mnist_inference.IMAGE_SIZE,    # 第二维和第三维表示图片的尺寸。
            p151_mnist_inference.IMAGE_SIZE,
            p151_mnist_inference.NUM_CHANNELS],   # 第四维表示图片的深度，对于RBG
                                                # 格式的图片，深度为3
        name='x-input'
    )
    y_ = tf.placeholder(
        tf.float32,[None,p151_mnist_inference.OUTPUT_NODE],name='y-input'
    )

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用p151_mnist_inference.py中定义的前向传播过程
    y = p151_mnist_inference.inference(
        input_ternsor=x,train=False,regualrizer=regularizer)
    global_step = tf.Variable(0,trainable=False)

    # 和5.2.1小节样例中类似地定义损失函数，学习率，滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step
    )
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y,labels=tf.argmax(y_,1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久化类.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独
        # 立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 类似地将输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入
            # sess.run()过程
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          p151_mnist_inference.IMAGE_SIZE,
                                          p151_mnist_inference.IMAGE_SIZE,
                                          p151_mnist_inference.NUM_CHANNELS))
            #_, loss_value, step = sess.run([train_op,loss,global_step],
            #                                feed_dict={x:xs, y_:ys})
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})
            # 每一千轮保存一次模型。
            if i % 1000 == 0:
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函
                # 数大小。通过损失函数的大小可以大概了解训练的情况。在验证数据集上的
                # 正确信息会有一个单独的程序来生成。
                print ("after %d training step(s), loss on trainging "
                       "batch is %g." % (step,loss_value))
                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个被
                # 保存模型的文件名末尾加上训练的轮数，比如"p150_model.ckpt-1000"表示
                # 训练1000轮之后得到的模型
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                    global_step=global_step
                )

def main(argv=None):
    mnist = input_data.read_data_sets("/home/shenxj/tf-work/mnist",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()