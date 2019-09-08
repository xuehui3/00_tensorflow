import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def tensorflow_demo():
    """
    通过简单案例来了解tensorflow的基础结构
    :return: None
    """
    # 一、原生python实现加法运算
    a = 10
    b = 20
    c = a + b
    print("原生Python实现加法运算方法1：\n", c)
    def add(a, b):
        return a + b
    sum = add(a, b)
    print("原生python实现加法运算方法2：\n", sum)

    # 二、tensorflow实现加法运算
    a_t = tf.constant(10)
    b_t = tf.constant(20)
    # 不提倡直接运用这种符号运算符进行计算
    # 更常用tensorflow提供的函数进行计算
    # c_t = a_t + b_t
    c_t = tf.add(a_t, b_t)
    print("tensorflow实现加法运算：\n", c_t)
    # 如何让计算结果出现？
    # 开启会话
    with tf.Session() as sess:
        sum_t = sess.run(c_t)
        print("在sess当中的sum_t:\n", sum_t)

    return None


def graph_demo():
    # 图的演示
    a_t = tf.constant(10)
    b_t = tf.constant(20)
    # 不提倡直接运用这种符号运算符进行计算
    # 更常用tensorflow提供的函数进行计算
    # c_t = a_t + b_t
    c_t = tf.add(a_t, b_t)
    print("tensorflow实现加法运算：\n", c_t)

    # 获取默认图
    default_g = tf.get_default_graph()
    print("获取默认图：\n", default_g)

    # 数据的图属性
    print("a_t的graph:\n", a_t.graph)
    print("b_t的graph:\n", b_t.graph)
    # 操作的图属性
    print("c_t的graph:\n", c_t.graph)

    # 开启会话
    with tf.Session() as sess:
        sum_t = sess.run(c_t)
        print("在sess当中的sum_t:\n", sum_t)
        # 会话的图属性
        print("会话的图属性：\n", sess.graph)

    return None


if __name__ == '__main__':
    tensorflow_demo()
    graph_demo()