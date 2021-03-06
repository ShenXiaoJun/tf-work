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

    # 通过tf.image.resize_image_with_crop_or_pad函数调整图像的大小。这个函数的
    # 第一个参数为原始图像，后面两个参数是调整后的目标图像大小。如果原始图像的尺寸大于目标
    # 图像，那么这个函数会自动截取原始图像中居中的部分。如果目标图像
    # 大于原始图像，这个函数会自动在原始图像的四周填充全0背景。因为原
    # 始图像的大小为1797x2673，所以下面的第一个命令会自动剪裁，而第二个命令为自动填充
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    plt.imshow(croped.eval())
    plt.show()
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    plt.imshow(padded.eval())
    plt.show()

    # 通过tf.image.central)crop函数可以按比例裁剪图像。这个函数的第一个参数为原始图
    # 像，第二个为调整比例，这个比例需要是一个（0,1]的实数
    central_cropped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_cropped.eval())
    plt.show()