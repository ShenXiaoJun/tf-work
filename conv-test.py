# encoding:utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

SrcImgRaw = tf.gfile.FastGFile("/home/shenxj/tf-work/datasets/cat.jpg", 'r').read()
TagImgRaw = tf.gfile.FastGFile("/home/shenxj/tf-work/datasets/cat.jpg", 'r').read()

with tf.Session() as sess:
    SrcImg = tf.image.decode_jpeg(SrcImgRaw)
    SrcImg = tf.image.convert_image_dtype(SrcImg, dtype=tf.float32)
    SrcImg = tf.image.resize_images(SrcImg, [128, 128], method=0)
    plt.imshow(SrcImg.eval())
    plt.show()

    TagImg = tf.image.decode_jpeg(TagImgRaw)
    TagImg = tf.image.convert_image_dtype(TagImg, dtype=tf.float32)
    TagImg = tf.image.resize_images(TagImg, [128, 128], method=0)
    plt.imshow(TagImg.eval())
    plt.show()

    CONV_1_9_SIZE = 3
    CONV_1_9_DEEP = 16
    SRC_IMG_CHANNELS = 3
    conv_1_9_w = tf.get_variable(
        "conv_1_9_w", [CONV_1_9_SIZE, CONV_1_9_SIZE, SRC_IMG_CHANNELS, CONV_1_9_DEEP],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv_1_9_b = tf.get_variable(
        "conv_1_9_b", [CONV_1_9_DEEP], initializer=tf.constant_initializer(0.0))
    # 使用变长为3,深度为16的过滤器，过滤器移动的步长为1,且使用全0填充
    Convolution1 = tf.nn.conv2d(SrcImg, conv_1_9_w, strides=[1, 1, 1, 1], padding='SAME')
    ReLU1 = tf.nn.relu(tf.nn.bias_add(Convolution1, conv_1_9_b))

    Convolution9 = tf.nn.conv2d(TagImg, conv_1_9_w, strides=[1, 1, 1, 1], padding='SAME')
    ReLU9 = tf.nn.relu(tf.nn.bias_add(Convolution9, conv_1_9_b))

    CONV2_SIZE = 3
    CONV2_DEEP = 32
    conv2_w = tf.get_variable(
        "conv2_w", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv2_b = tf.get_variable(
        "conv2_b", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
    # 使用变长为3,深度为32的过滤器，过滤器移动的步长为2,且使用全0填充
    Convolution2 = tf.nn.conv2d(ReLU1, conv2_w, strides=[1, 2, 2, 1], padding='SAME')
    ReLU1 = tf.nn.relu(tf.nn.bias_add(Convolution2, conv2_b))

    Convolution10 = tf.nn.conv2d(ReLU9, conv2_w, strides=[1, 2, 2, 1], padding='SAME')
    ReLU9 = tf.nn.relu(tf.nn.bias_add(Convolution9, conv2_b))

    CONV2_SIZE = 3
    CONV2_DEEP = 32
    conv2_w = tf.get_variable(
        "conv2_w", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv2_b = tf.get_variable(
        "conv2_b", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
    # 使用变长为3,深度为32的过滤器，过滤器移动的步长为2,且使用全0填充
    Convolution2 = tf.nn.conv2d(ReLU1, conv2_w, strides=[1, 2, 2, 1], padding='SAME')
    ReLU1 = tf.nn.relu(tf.nn.bias_add(Convolution2, conv2_b))

    Convolution10 = tf.nn.conv2d(ReLU9, conv2_w, strides=[1, 2, 2, 1], padding='SAME')
    ReLU9 = tf.nn.relu(tf.nn.bias_add(Convolution9, conv2_b))