import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.python import keras
from tensorflow.python.keras.datasets import fashion_mnist
import tensorflow as tf


'''
70000 张灰度图像，涵盖 10 个类别。以下图像显示了单件服饰在较低分辨率（28x28 像素）下的效果：
'''

# 构建双层神经网络去进行时装模型训练与预测
#   -1、读取数据集
# # #   - 2、建立神经网络模型
# # #   - 3、编译模型优化器、损失、准确率
# # #   - 4、进行fit训练
# # #   - 评估模型测试效果


class SingleNN(object):

    # 建立神经网络模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # 将输入数据的形状进行修改成神经网络要求的数据形状
        keras.layers.Dense(128, activation=tf.nn.relu),  # 定义隐藏层，128个神经元的网络层
        keras.layers.Dense(10, activation=tf.nn.softmax)  # 10个类别的分类问题，输出神经元个数必须跟总类别数量相同
    ])

    def __init__(self):

        # 返回两个元组
        # x_train: (60000, 784), y_train:(60000, 1)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()

        # 进行数据的归一化
        # self.x_train = self.x_train / 255.0
        # self.x_test = self.x_test / 255.0

    def singlenn_compile(self):
        """
        编译模型优化器、损失、准确率
        :return:
        """
        # 优化器
        # 损失函数
        SingleNN.model.compile(optimizer=keras.optimizers.Adam(),
                               loss=keras.losses.sparse_categorical_crossentropy,
                               metrics=['accuracy'])

        return None

    def singlenn_fit(self):
        """
        进行fit训练
        :return:
        """
        # # fit当中添加回调函数，记录训练模型过程
        # modelcheck = keras.callbacks.ModelCheckpoint(
        #     filepath='./ckpt/singlenn_{epoch:02d}-{val_loss:.2f}.h5',
        #     monitor='val_loss',  # 保存损失还是准确率
        #     save_best_only=True,
        #     save_weights_only=True,
        #     mode='auto',
        #     period=1
        # )
        # 调用tensorboard回调函数
        board = keras.callbacks.TensorBoard(log_dir="./graph/", write_graph=True)

        # 训练样本的特征值和目标值
        SingleNN.model.fit(self.x_train, self.y_train, epochs=5,
                           batch_size=128, callbacks=[board])

        return None

    def single_evalute(self):

        # 评估模型测试效果
        test_loss, test_acc = SingleNN.model.evaluate(self.x_test, self.y_test)

        print(test_loss, test_acc)

        return None

    def single_predict(self):
        """
        预测结果
        :return:
        """
        # 首先加载模型
        # if os.path.exists("./ckpt/checkpoint"):
        SingleNN.model.load_weights("./ckpt/SingleNN.h5")

        predictions = SingleNN.model.predict(self.x_test)

        return predictions


if __name__ == '__main__':
    snn = SingleNN()

    snn.singlenn_compile()

    snn.singlenn_fit()

    snn.single_evalute()

    # SingleNN.model.save_weights("./ckpt/SingleNN.h5")
    # 进行模型预测
    # predictions = snn.single_predict()
    # # [10000, 10]
    # print(predictions)
    # print(np.argmax(predictions, axis=1))
















