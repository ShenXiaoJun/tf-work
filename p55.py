# encoding:utf-8
import tensorflow as tf

# 声明w1,w2 两个变量。这里还通过 seed参数设定了随即种子，这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。注意这里x是一个1*2的矩阵
x = tf.constant([[0.7, 0.9]])

# 通过3.4.2小节描述的前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 与3.4.2中的计算不同，这里不能直接通过sess.run(y)来获得y的取值
# 因为w1和w2都还没有运行初始化过程。下面的两行分别初始化了w1和w2两个变量
sess.run(w1.initializer)    # 初始化w1
sess.run(w2.initializer)    # 初始化w2
# 输出
print(sess.run(y))
sess.close()