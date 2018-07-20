# encoding:utf-8
import tensorflow as tf
# tf.train.NewCheckpointReader可以读取checkpoint文件中保存的所有变量
reader = tf.train.NewCheckpointReader("/home/shenxj/tf-work/model/p113-model.ckpt")

# 获取所有变量列表。这个是一个从变量名到变量维度的字典
all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
    # variable_name为变量名称，all_variables[variable_name]为变量的维度
    print variable_name, all_variables[variable_name]

# 获取名称为v1的变量的取值
print "Value for variable v1 is ", reader.get_tensor("v1")