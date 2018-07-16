# encoding:utf-8
import tensorflow as tf
with tf.variable_scope("root"):
    # 可以根据tf.get_variable-scope().reuse函数来获取当前上下文管理其中reuse参
    # 数的取值
    print tf.get_variable_scope().reuse   # 输出false，即最外层reuse是false
    # 新建一个嵌套的上下文管理器,并指定reuse为True
    with tf.variable_scope("foo",reuse=True):
        # 输出True
        print tf.get_variable_scope().reuse
        # 新建一个嵌套的上下文管理器但不指定reuse，这时reuse的取值会和外面一层保持一致。
        with tf.variable_scope("bar"):
            # 输出True
            print tf.get_variable_scope().reuse
    # 输出False。退出reuse设置为True的上下文之后reuse的值又回到了False
    print tf.get_variable_scope().reuse