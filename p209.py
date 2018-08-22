# encoding:utf-8
import tensorflow as tf

# 定义一个基本的LSTM结构作为循环体的基础结构。深层循环神经网络也支持使用其他的循环体结构
lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)

# 通过MultiRNNCell类实现深层循环神经网络中每一时刻的前向传播过程。其中
# number_of_layers表示了有多少层，也就是图8-16中从xt到ht需要经过多少个LSTM结构
stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)

# 和经典的循环神经网络一样，可以通过zero_state函数来获取初始状态
state = stacked_lstm.zero_state(batch_size,tf.float32)

# 可2.8节中给出的代码一样，计算每一时刻的前向传播结果
loss = 0.0
for i in range(num_steps):

    if i > 0: tf.get_variable_scope().reuse_variables()

    stacked_lstm_output, state = stacked_lstm(current_input, state)

    final_output = fully_connected(stacked_lstm_output)

    loss += calc_loss(final_output, expected_output)