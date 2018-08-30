# encoding:utf-8
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

# 存放原始数据的路径
DATA_PATH = "/home/shenxj/tf-work/ptb/simple-examples/data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
# 读取数据原始数据
print len(train_data)
print train_data[:100]