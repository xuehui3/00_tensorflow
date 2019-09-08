import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.python.keras.datasets import cifar100
from tensorflow.python import keras
import tensorflow as tf


'''
第一层
卷积：[None, 32, 32, 3]———>[None, 32, 32, 32]
权重数量：[5, 5, 3 ,32]
偏置数量：[32]
激活：[None, 32, 32, 32]———>[None, 32, 32, 32]
池化：[None, 32, 32, 32]———>[None, 16, 16, 32]
第二层
卷积：[None, 16, 16, 32]———>[None, 16, 16, 64]
权重数量：[5, 5, 32 ,64]
偏置数量：[64]
激活：[None, 16, 16, 64]———>[None, 16, 16, 64]
池化：[None, 16, 16, 64]———>[None, 8, 8, 64]
全连接层
[None, 8, 8, 64]——>[None, 8 8 64]
[None, 8 8 64] x [8 8 64, 1024] = [None, 1024]
[None,1024] x [1024, 100]——>[None, 100]
权重数量：[8 8 64, 1024] + [1024, 100]，由分类别数而定
偏置数量：[1024] + [100]，由分类别数而定
'''

# - 读取数据集
# - 编写两层卷积层+两层神经网络层
# - 编译  训练  评估

class CNNMinst(object):
    # 编写两层卷积层 + 两层神经网络层
    model = keras.models.Sequential([
        # 卷积层1： 32个 5*5*3 的filter  步长strides=1 padding="same"
        keras.layers.Conv2D(32, kernel_size=5, strides=1, padding="same",
                            data_format="channels_last", activation=tf.nn.relu),   # --[None,32,32,32]
        # 池化层 ： 2*2 窗口， 步长strides=2
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"),           # --[None,16,16,32]
        # 第二层：
        # 卷积层2： 64个 5*5*32的 filter 步长为strides padding="same"
        keras.layers.Conv2D(64, kernel_size=5, strides=1, padding="same"),        # --[None,16,16,64]
        # 池化层2    2*2 窗口， 步长strides=2
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"),           # --[None,8,8,64]

        # --[None,8,8,64] >>[None,8*8*64] 展开成特征行
        # keras.layers.Flatten(input_shape=),
        keras.layers.Flatten(),

        # 全连接层神经网络
        # 1024 个神经网络
        keras.layers.Dense(1024, activation=tf.nn.relu),
        # 100 个神经网络
        keras.layers.Dense(100, activation=tf.nn.softmax)

    ])

    def __init__(self):

        # 获取训练测试数据
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        # print(self.x_train.shape)
        # print(self.x_test.shape)

        # 进行数据归一化
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def compile(self):
        # 编译器
        CNNMinst.model.compile(optimizer=keras.optimizers.Adam(),
                               loss=keras.losses.sparse_categorical_crossentropy,
                               metrics=["accuracy"])
        return None


    def fit(self):
        # 训练
        CNNMinst.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return None

    def evaluate(self):
        test_loss, test_acc = CNNMinst.model.evaluate(self.x_test, self.y_test)
        print(test_loss, test_acc)
        # epochs=1, batch_size=32
        # 50000/50000 [==============================] - 186s 4ms/step - loss: 3.3812 - acc: 0.2026
        # epochs=1, batch_size=128
        # 50000/50000 [==============================] - 143s 3ms/step - loss: 3.4126 - acc: 0.1973
        # epochs=5, batch_size=128
        # loss: 1.0692 - acc: 0.7037
        # epochs = 5, batch_size = 256
        # loss: 1.4372 - acc: 0.6099
        # epochs = 5, batch_size = 512
        # loss: 1.8700 - acc: 0.5111
        return None


if __name__ == '__main__':

    cnn = CNNMinst()

    cnn.compile()

    cnn.fit()

    print(cnn.model.summary())  # 输出模型各层的参数状况


