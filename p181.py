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
    #img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 将图像缩小一些，这样可视化能让标注框更加清楚
    img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    # tf.image.draw_bounding_boxes函数要求图像矩阵中的数字为实数，所以需要先将
    # 图像矩阵转化为实数类型。tf.image.draw_bounding_boxes函数图像的输入是一个
    # batch的数据，也就是多张图像组成的四维矩阵，所以需要将解码之后的图像矩阵加一维。
    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data, tf.float32), 0
    )
    # 给出每一张图像的所有标注框。一个标注框有四个数字，分别代表[ymin, xmin, ymax, xmax],
    # 注意这里给出的数字都是图像的相对位置,比如在180x267的图像中
    # [0.35, 0.47, 0.5, 0.56]代表了从(63, 125)到(90, 150)的图像
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)

    result = tf.squeeze(result, 0)
    plt.imshow(result.eval())
    plt.show()

####

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    # sample_distorted_bounding_box要求输入图片必须是实数类型。
    image_float = tf.image.convert_image_dtype(img_data, tf.float32)

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image_float), bounding_boxes=boxes, min_object_covered=0.4)

    # 截取后的图片
    distorted_image = tf.slice(image_float, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()

    # 在原图上用标注框画出截取的范围。由于原图的分辨率较大（2673x1797)，生成的标注框
    # 在Jupyter Notebook上通常因边框过细而无法分辨，这里为了演示方便先缩小分辨率。
    image_small = tf.image.resize_images(image_float, [180, 267], method=0)
    batchced_img = tf.expand_dims(image_small, 0)
    image_with_box = tf.image.draw_bounding_boxes(batchced_img, bbox_for_draw)
    print(bbox_for_draw.eval())
    plt.imshow(image_with_box[0].eval())
    plt.show()