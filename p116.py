# encoding:utf-8
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "/home/shenxj/tf-work/model/p115-combined_model.pb"
    # 读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图加载到当前的图中。return_elements=["add:0"]给出了返回
    # 的张量的名称。在保存的时候给出的是结算节点的名称，所以为"add"。在加载
    # 的时候给出的是张量的名称，所以是add:0
    result = tf.import_graph_def(graph_def,return_elements=["add:0"])
    print sess.run(result)
