import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# a = tf.constant(11.0)
# b = tf.constant(22.0)
# c = tf.add(a, b)
#
# # 获取默认图
# default_g = tf.get_default_graph()
# print('获取当前加法运算的图', default_g)
#
# # 打印所有操作， 张量默认图
# print(a.graph)
# print(b.graph)
# print(c.graph)
#
# # 创建另外一张图
# new_g = tf.Graph()
# with new_g.as_default():
#     new_a = tf.constant(11.0)
#     new_b = tf.constant(22.0)
#     new_c = tf.add(new_a, new_b)
#
# print(new_a.graph)
# print(new_b.graph)
# print(new_c.graph)
#
# # print(c)  #Tensor("Add:0", shape=(), dtype=float32)
#
# with tf.Session() as sess1:
#     sum_ = sess1.run(c)
#     print(sum_)
#
# with tf.Session(graph=new_g) as sess2:
#     sum_new = sess2.run(new_c)
#     print(sum_new)


# tensorboart

a = tf.constant(11.0, name="con_a")
b = tf.constant(22.0, name="con_b")
c = tf.add(a, b, name="sum_c")

# 获取默认图
# default_g = tf.get_default_graph()
# print('获取当前加法运算的图', default_g)

# 打印所有操作， 张量默认图
# print(a.graph)
# print(b.graph)
# print(c.graph)

# print(c)  #Tensor("Add:0", shape=(), dtype=float32)
print(a)
print(b)
print(c)


with tf.Session() as sess:
    print(sess.graph)

    # 写入到events文件当中
    file_writer = tf.summary.FileWriter('./tmp/', graph=sess.graph)

    sum_ = sess.run(c)
    print(sum_)



