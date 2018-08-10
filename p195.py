# encoding:utf-8
import tensorflow as tf
import p182.py as p182
# 创建文件列表，并通过文件列表创建输入文件队列。在调用输入数据处理流程前，需要
# 统一所有原始数据的格式并将它们存储到TFRcord文件中。下面给出的文件列表应该包含所
# 有提供训练数据的TFRcord文件
files = tf.train.match_filenames_once("/home/shenxj/tf-work/datasets/file_pattern-*")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 使用类似7.1节中结婚嫂的方法解析TFRecord文件里的数据。这里假设image中存储的是图像
# 的原始数据，label为该样例所对应的标签。height,width和channels给出了图像的维度。
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'weigth': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    }
)
image, label = features['image'], features['label']
height, width = features['height'], features['wigth']
channels = features['channels']

# 从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])
# 定义神经网络输入层图片的大小。
image_size = 299
# preprocess_for_train为7.2.2小节中介绍的图像预处理程序
distorted_image = p182.preprocess_for_train

