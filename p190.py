# encoding:utf-8
import tensorflow as tf
# 创建TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 模拟海量数据情况下将数据写入不同的文件。num_shards定义了总共写入多少个文件
# instance_per_shared定义了每个文件中有多少个数据
num_shareds = 2
instance_per_shared = 2
for i in range(num_shareds):
    # 将数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分。其中m表
    # 示了数据总共被存在了多少个文件中，n表示当前文件的编号。式样的方式既方便了通过正
    # 则表达式获取文件列表，又在文件名中加入了更多的信息
    filename = ('/home/shenxj/tf-work/datasets/data.tfrecords-%.5d-of-%.5d' %
                (i, num_shareds))
    writer = tf.python_io.TFRecordWriter(filename)
    # 将数据封装Example结构并写入TFRecord文件
    for j in range((instance_per_shared)):
        # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)
        }))
        writer.write(example.SerializeToString())
    writer.close()