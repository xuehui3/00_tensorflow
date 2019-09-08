# import tensorflow as tf
# import os

# def linear_regression():
#     """
#     自实现线性回归
#     :return: None
#     """
#     # 1）准备好数据集：y = 0.8x + 0.7 100个样本
#     # 特征值X, 目标值y_true
#     X = tf.random_normal(shape=(100, 1), mean=2, stddev=2)
#     # y_true [100, 1]
#     # 矩阵运算 X（100， 1）* （1, 1）= y_true(100, 1)
#     y_true = tf.matmul(X, [[0.8]]) + 0.7
#     # 2）建立线性模型：
#     # y = W·X + b，目标：求出权重W和偏置b
#     # 3）随机初始化W1和b1
#     weights = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)))
#     bias = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)))
#     y_predict = tf.matmul(X, weights) + bias
#     # 4）确定损失函数（预测值与真实值之间的误差）-均方误差
#     error = tf.reduce_mean(tf.square(y_predict - y_true))
#     # 5）梯度下降优化损失：需要指定学习率（超参数）
#     # W2 = W1 - 学习率*(方向)
#     # b2 = b1 - 学习率*(方向)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
#
#     # 初始化变量
#     init = tf.global_variables_initializer()
#     # 开启会话进行训练
#     with tf.Session() as sess:
#         # 运行初始化变量Op
#         sess.run(init)
#         print("随机初始化的权重为%f， 偏置为%f" % (weights.eval(), bias.eval()))
#         # 训练模型
#         for i in range(200):
#             sess.run(optimizer)
#             print("第%d步的误差为%f，权重为%f， 偏置为%f" % (i, error.eval(), weights.eval(), bias.eval()))
#
#     return None

# if __name__ == '__main__':
    # linear_regression()

# 利用tensorflow实现线性回归的训练
# -导包
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

'''
步骤分析：
1-准备好数据集：y=2x + 3
2-建立线性模型
    随机初始化w1和b1
    y=wx + b 目标：求出权重w和偏置b
3-确定损失函数（预测值与真实值之间的误差）--均方误差mse
4-梯度下降优化损失: 需要指定学习率（超参数）

'''

# 命令行参数
tf.app.flags.DEFINE_integer("max_step", 400, "train step number")
# 定义获取命令行参数
FLAGS = tf.app.flags.FLAGS

def linear_regression():

    with tf.variable_scope("origin_scope"):
        # 1 - 准备好数据集：y = 2x + 3      特征值x，目标值y_true
        x = tf.random_normal(shape=(100, 1), mean=0, stddev=1)

        y_true = tf.matmul(x, [[2.0]]) + 3.0
        # y_true = tf.matmul([[2.0]], x) + 3.0  # 矩阵乘法
    with tf.variable_scope("linear_model"):
        # 2 - 建立线性模型
        #     随机初始化w1和b1
        weights = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)), name="w")
        # weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="weights", trainable=False)
        # trainable=False 权重不参与训练
        bias = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)), name="b")

        #     y = wx + b目标：求出权重w和偏置b
        # y_predict = tf.matmul(weights, x) + bias
        y_predict = tf.matmul(x, weights) + bias
        # y_predict_1 = tf.matmul(x, weight) + bias
    with tf.variable_scope("loss"):
        # 3-确定损失函数（预测值与真实值之间的误差）--均方误差mse
        mse = tf.reduce_mean(tf.square(y_predict - y_true))
        # mse_1 = tf.reduce_mean(tf.square(y_predict_1 - y_true))
        # print(type(mse), mse)
    with tf.variable_scope('GD_optimizer'):
        # 4 - 梯度下降优化损失: 需要指定学习率（超参数）
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=0.01).minimize(mse)
        # optimizer_1 = tf.train.ProximalGradientDescentOptimizer(learning_rate=0.01).minimize(mse_1)

        # ---近端的梯度下降优化器
        # optimizer: (n)-优化程序；最优控制器
        # proximal： (adj). 近端的；近源的；（牙齿）近侧的
        # descent :  (n) 下降

    # 收集变量
    tf.summary.scalar("error", mse)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", bias)

    # 合并变量
    merge = tf.summary.merge_all()


    # 初始化变量值
    init_op = tf.global_variables_initializer()


    saver = tf.train.Saver()
    # 开启会话 - 进行训练
    with tf.Session() as sess:
        # 运行初始化变量init_op

        sess.run(init_op)
        print("随机初始化的权重为%f， 偏置为%f" % (weights.eval(), bias.eval()))
        # 1）创建事件文件
        file_writer = tf.summary.FileWriter(logdir="./tmp_00/summary/", graph=sess.graph)
        # -----attention 要切换到当前文件目录下执行 tensorboard --logdir="./tmp_00/summary/"
        # 训练模型
        for i in range(FLAGS.max_step):
            # sess.run(optimizer_1)
            sess.run(optimizer)
            print("第%d步的误差为%f，权重为%f， 偏置为%f" % (i, mse.eval(), weights.eval(), bias.eval()))

            # 运行合并变量op
            summary = sess.run(merge)
            file_writer.add_summary(summary, i)

            # checkpoint检查点文件格式
            saver.save(sess, "./tmp_00/ckpt/linearregression")


        # while True:
        #     # sess.run(optimizer)
        #     if (mse.eval()) < 0.01:
        #         break
        #     else:
        #         sess.run(optimizer)
        #     print("误差为%f，权重为%f， 偏置为%f" % (mse.eval(), weights.eval(), bias.eval()))
        # -----如何查看梯度下降到第几步停止的？=-----------------------------------------------------------------

    return None


if __name__ == '__main__':
    linear_regression()
















