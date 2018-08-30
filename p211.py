# encoding:utf-8
import tensorflow as tf
# 定义LSTM结构
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

# 使用DropoutWrapper类来实现dropout功能。该类通过两个参数来控制dropout的概率，
# 一个参数为input_keep_prob,它可以用来控制输出的dropout概率：另一个为
# output_keep_prob,它可以用来控制输出的dropout概率。
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.5)

# 在使用了dropout的基础上定义
stacked = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * number_of_layers)

# 和8.3.1小节中深层循环网络样例程序类似，运行前向传播过程