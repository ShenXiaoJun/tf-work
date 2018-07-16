# encoding:utf-8
import tensorflow as tf

v1 = tf.get_variable("v",[1])
# 输出v:0."v"为变量的名称,":0"表示这个变量是生成变量这个运算的第一个结果
print v1.name

with tf.get_variable_scope("foo"):
    v2 = tf.get_variable("v",[1])
    # 输出foo/v:0。在tf.variable_scope中创建的变量，名称前面会加入命名空间的名称，
    # 并通过/来分隔命名空间的名称和变量的名称
    print v2.name

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v",[1])
        # 输出foo/bar/v:0。命名空间可以嵌套，同时变量的名称也会加入所有命名空间的名称作为前缀
        print v3.name

    v4 = tf.get_variable("v1", [1])
    # 输出foo/v1:0。当命名空间推出之后，变量名称也就不会再被加入其前缀了。
    print v4.name

# 创建一个名称为空的命名空间，并设置reuse=True
with tf.variable_scope("",reuse=True):
    # 可以直接通过带命名空名称的变量名来获取其他变量命名空间下的变量。比如这
    # 里通过指定名称foo/bar/v来获取在命名空间foo/bar/v中创建的变量。
    v5 = tf.get_variable("foo/bar/v", [1])
    # 输出True
    print v5 == v3

    v6 = tf.get_variable("foo/v1",[1])
    print v6 == v4