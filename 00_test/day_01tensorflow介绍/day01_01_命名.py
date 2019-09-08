import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf


# # 实现一个加法运算
# con_a = tf.constant(3.0, name="a_a")
# con_b = tf.constant(4.0, name="b_b")
#
# sum_c = tf.add(con_a, con_b)
#
# print("打印con_a：\n", con_a)
# print("打印con_b：\n", con_b)
# print("打印sum_c：\n", sum_c)
#
# # 运行会话并打印设备信息
# # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
# #                                         log_device_placement=True)) as sess:
# with tf.Session() as sess:
#     sum_ = sess.run(sum_c)
#
#     # 返回filewriter,写入事件文件到指定目录(最好用绝对路径)，以提供给tensorboard使用
#     # tf.summary.FileWriter('./tmp/summary/test/', graph=sess.graph)
#     print('求和', sum_)
#
#     # 会话的图属性
#     print("属性", sess.graph)



# 定义占位符
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# sum_ab = tf.add(a, b)
# print(sum_ab)
#
# # 开启会话
# with tf.Session() as sess:
#     # sum_s = sess.run(sum_ab)
#     sum_s = sess.run(sum_ab, feed_dict={a: 3.0, b: 4.0})
#     print(sum_s)


# a = tf.constant(5.0)
# b = tf.constant(6.0)
# c = a * b
#
# # 创建会话
# sess = tf.Session()
#
# # 计算C的值
# print(sess.run(c))
# print(c.eval(session=sess))
# # 使用tf.operation.eval()也可运行operation，但需要在会话中运行



# tf.Tensor.eval()
# 功能：当默认的会话被指定之后可以通过其计算一个张量的取值
# a=tf.constant([1.0,2.0],name="a")
# b=tf.constant([2.0,3.0],name="b")
# c=tf.add(a,b,name="sum")
#
# print(c)
#
# sess=tf.Session()
# with sess.as_default():
#     print(c.eval())
#     print(a.eval())


# 张量的阶
# tensor1 = tf.constant(4.0)
# tensor2 = tf.constant([1, 2, 3, 4])
# linear_squares = tf.constant([[4], [9], [16], [25]], dtype=tf.int32)
#
# print(tensor1.shape)
# print(tensor2.shape)
# print(linear_squares.shape)


# 变量的operation
a = tf.Variable(initial_value=30.0)
b = tf.Variable(initial_value=40.0)
sum = tf.add(a, b)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 手动运行init_op
    sess.run(init_op)
    print(sess.run(sum))