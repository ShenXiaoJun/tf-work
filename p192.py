# encoding:utf-8
import tensorflow as tf

# 使用tf.train.match_filenames_once函数获取文件列表
files = tf.train.match_filenames_once(
    "/home/shenxj/tf-work/datasets/data.tfrecords-*")

# 通过tf.train.string_input_producer函数创建输入队列，输入队列中的文件列表为
# tf.train.macth_filenames_once函数获取的文件列表。这里将shuffle参数设为False
# 来避免随机打乱读文件的顺序。但一般在解决真实问题时，会将shuffle参数设为True。
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 读取并解析一个样本
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    }
)

with tf.Session() as sess:
    # 虽然在本段程序中没有声明任何变量，但使用tf.train.match_filename_once函数时需
    # 要初始化一些变量。
    #tf.initialize_all_variables().run()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print sess.run(files)

    # 声明tf.train.Coordinator类来协同不同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 多次执行获取数据的操作
    for i in range(6):
        print sess.run([features['i'], features['j']])
    coord.request_stop()
    coord.join(threads)
##############################
example, label = features['i'], features['j']

# 一个batch中样例的个数
batch_size = 3
# 组合样例的队列中最多可以存储的样例个数。这个队列如果太大，那么需要占用很多内存资源；
# 如果太小，那么出队操作可能会因为没有数据而被阻碍（block），从而导致训练效率降低。一般
# 来说这个队列的大小会和每一个batch大大小相关，下面一行代码给出了设置队列大小的一种
# 方式
capacity = 1000 + 3 * batch_size

# 使用tf.train.btach函数来组合样例。[example, label]参数给出了需要组合的元素，
# 一般example和label分别代表训练样本和这个样本对应的正确标签。batch_size参数给出
# 了每个batch中样例的个数。capacity给出了队列的最大容量。当队列长度等于容量时，
# TensorFlow将暂停入队操作，而只是等待元素出队。当元素个数小于容量时，TensorFlow
# 将自动重新启动入队操作
example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size, capacity=capacity
)

with tf.Session() as sess:
    #tf.initialize_all_variables().run()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 获取并打印组合之后的样例。在真实问题中，这个输出一般会作为神经网络的输入
    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run(
            [example_batch, label_batch]
        )
        print cur_example_batch, cur_label_batch

    coord.request_stop()
    coord.join(threads)
##############################
# 和tf.train.batch的样例代码一样产生example和label
example, label = features['i'], features['j']

batch_size = 3
capacity = 1000 + 3 * batch_size

# 使用tf.train.shuffle_batch函数来组合样例。tf.train.shuffle_batch函数
# 的参数大部分和tf.train.batch函数相似，但是min_after_dequeue参数是
# tf.train.shuffle_batch函数特有的。min_after_queue参数限制了出队时队列中元
# 素的最少个数。当队列中元素太少时，随机打乱样例顺序的作用就不大了。所以
# tf.train.shuffle_batch函数提供了限制出队时最少元素的个数来保证随机打乱顺序的
# 作用。当出队函数被调用但是队列中元素不够时，出队操作将等待更多的元素入队才会完成。
# 如果min_after_dequeue参数被设定，capacity也应该相应调整来满足性能需求
example_batch, label_batch = tf.train.shuffle_batch(
    [example, label], batch_size=batch_size,
    capacity=capacity, min_after_dequeue=30
)

with tf.Session() as sess:
    #tf.initialize_all_variables().run()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run(
            [example_batch, label_batch]
        )
        print cur_example_batch, cur_label_batch

    coord.request_stop()
    coord.join(threads)