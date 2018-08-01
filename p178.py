# encoding:utf-8
# matplotlib.pyplot是一个python的画图工具。在这一节中将使用这个工具来可视
# 化经过TensorFlow处理的图像
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("/home/shenxj/tf-work/datasets/cat.jpg", 'r').read()

with tf.Session() as sess:
    # 将图像使用jpeg的格式解码从而得到图像对应的三维矩阵。TensorFlow还提供了
    # tf.image.decode_png函数对png格式的图像进行解码。解码之后的结果为一个
    # 张量，在使用它的取值之前需要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data)

    print img_data.eval()

    # 使用pyplot工具可视化得到的图像
    plt.imshow(img_data.eval())
    plt.show()

    # 将数据的类型转化成实数方便下面的样例程序对图像进行处理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 将表示一张图像的三维矩阵重新按照jpeg格式编码存入文件中。打开这张图像，
    # 可以得到和原始图像一样的图像
    # encoded_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.GFile("/home/shenxj/tf-work/datasets/cat-fix.jpg", "wb") as f:
    #     f.write(encoded_image.eval())

    # 以一定概率上下翻转图像
    flipped = tf.image.random_flip_up_down(img_data)
    plt.imshow(flipped.eval())
    plt.show()
    # 以一定概率左右翻转图像
    flipped = tf.image.random_flip_left_right(img_data)
    plt.imshow(flipped.eval())
    plt.show()
