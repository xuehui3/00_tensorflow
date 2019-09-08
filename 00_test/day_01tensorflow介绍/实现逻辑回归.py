from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# 使用数据：制作二分类数据集
X, Y = make_classification(n_samples=500, n_features=5, n_classes=2)
# print(X.shape, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


'''
步骤设计： 分别构建算法的不同模块
'''

# 初始化参数
def initialize_with_zeros(shape):
    '''
    创建一个形状为（shape，1）的w参数和b=0
    :param shape:
    :return: w， b
    '''
    w = np.zeros((shape, 1))
    b = 0

    return w, b

def basic_sigmoid(x):
    '''
    计算sigmoid函数
    :param x:
    :return:
    '''
    s = 1/(1 + np.exp(-x))
    return s


'''
计算成本函数及其梯度
w (n,1).T * x (n, m)
y: (1, n)
'''


def propagate(w, b, X, Y):
    """
    参数：w,b,X,Y：网络参数和数据
    Return:
    损失cost、参数W的梯度dw、参数b的梯度db
    """
    m = X.shape[1]
    # print(m)
    # w (n,1), x (n, m)
    A = basic_sigmoid(np.dot(w.T, X) + b)
    # 计算损失
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dz = A - Y
    dw = 1 / m * np.dot(X, dz.T)
    db = 1 / m * np.sum(dz)

    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost

'''
使用优化算法（梯度下降）
实现优化函数. 全局的参数随着w,b对损失J进行优化改变. 对参数进行梯度下降公式计算，指定学习率和步长。
循环：
计算当前损失
计算当前梯度
更新参数（梯度下降）

'''
def optimize(w, b, X, Y, num_iterations, learning_rate):
    """
    参数：
    w:权重,b:偏置,X特征,Y目标值,num_iterations总迭代次数,learning_rate学习率
    Returns:
    params:更新后的参数字典
    grads:梯度
    costs:损失结果
    """

    costs = []

    for i in range(num_iterations):

        # 梯度更新计算函数
        grads, cost = propagate(w, b, X, Y)

        # 取出两个部分参数的梯度
        dw = grads['dw']
        db = grads['db']

        # 按照梯度下降公式去计算
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if i % 100 == 0:
            print("损失结果 %i: %f" % (i, cost))
            print(b)

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# 预测函数（不用实现）
# 利用得出的参数来进行测试得出准确率

def predict(w, b, X):
    '''
    利用训练好的参数预测
    return：预测结果
    '''

    m = X.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计算结果
    A = basic_sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1

    return y_prediction


# 整体逻辑
# 模型训练

def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.0001):
    """
    """

    # 修改数据形状
    x_train = x_train.reshape(-1, x_train.shape[0])
    x_test = x_test.reshape(-1, x_test.shape[0])
    y_train = y_train.reshape(1, y_train.shape[0])
    y_test = y_test.reshape(1, y_test.shape[0])
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # 1、初始化参数
    w, b = initialize_with_zeros(x_train.shape[0])

    # 2、梯度下降
    # params:更新后的网络参数
    # grads:最后一次梯度
    # costs:每次更新的损失列表
    params, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate)

    # 获取训练的参数
    # 预测结果
    w = params['w']
    b = params['b']
    y_prediction_train = predict(w, b, x_train)
    y_prediction_test = predict(w, b, x_test)

    # 打印准确率
    print("训练集准确率: {} ".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("测试集准确率: {} ".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return None


model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.0001)