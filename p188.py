# encoding:utf-8
import tensorflow as tf

# 声明一个先进先出的队列，队列中最多100个元素，类型为实数
queue = tf.FIFOQueue(100, "float")
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 使用tf.train.QueueRunner来创建多个线程运行队列的入队操作
# tf.train.QueueRunner的第一个参数给出了被操作的队列,[enqueue_op] * 5
# 表示了需要启动5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 将定义过的QueueRunner加入TensorFlow计算图上指定的集合。
# tf.train.add_queue_runner函数没有指定集合，
# 则加入默认集合tf.GraphKeys.QUEUE_RUNNERS.下面的函数就是刚刚定义的
# qr 加入默认的tf.GraphKeys.QUEUE_RUNNERS集合。
tf.train.add_queue_runner(qr)
# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用tf.train.Coordinator来协同启动的线程。
    coord = tf.train.Coordinator()
    # 使用tf.train.QueueRunner时，需要明确调用tf.train.start_queue_runners
    # 来启动所有线程。否则因为没有线程运行入队操作。当调用出队操作时，程序会一直等待入
    # 队操作被运行。tf.train.start_queue_runners函数会默认启动
    # tf.GraphKeys.QUEUE_RUNNERS集合中所有的QueueRunner。因为这个函数只支持启
    # 动指定集合中的QueueRunner,所以一般来说tf.train.add_queue_runner函数和
    # tf.train.start_queue_runners函数会指定同一个集合
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取队列中的取值
    for _ in range(3): print sess.run(out_tensor)[0]

    # 使用tf.train.Coordinator来停止所有的线程
    coord.request_stop()
    coord.join(threads)