# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
INPUT_NODE = 784        # 输入层的节点数。对于MNIST数据集，这个就是等于图片的像素
OUTPUT_NODE = 10        # 输出层的节点数。这个等于类别的数目.因为在MNIST数据集中
                        # 需要区分的是0~9这10个数字，所以这里输出层的节点数为10

# 配置神经网络的参数
LAYER1_NODE = 500       # 隐藏层节点数。这里使用一个隐藏层的网络结构作为样例
                        # 这个隐藏层有500个节点
BATCH_SIZE = 100        # 一个训练batch中的训练数据个数。数字越小时，训练过程越接近
                        # 随机梯度下降；数字越大时，训练越接近梯度下降

def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope('layer1',reuse=reuse):
        # 根据传进来的reuse来判断是创建新变量还是使用已经创建好的。在第一次构造网
        # 络时需要创建新的变量，以后每次调用这个函数都直接使用reuse=True就不需
        # 要每次将变量传进来了。
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
            initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)

    # 类似的定义第二层神经网络的变量和前向传播过程
    with tf.variable_scope("layer2",reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",[OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1,weights)+biases)
    # 返回最后的前向传播结果
    return layer2
