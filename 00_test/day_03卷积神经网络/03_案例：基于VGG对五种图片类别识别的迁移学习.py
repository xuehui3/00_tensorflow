'''
思路和步骤：
    读取本地的图片数据以及类别
        keras.preprocessing.image import ImageDataGenerator提供了读取转换功能
    模型的结构修改（添加我们自定的分类层）
    freeze掉原始的VGG模型
    编译以及训练和保存模型方式
    输入数据进行预测
'''

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import VGG16


class TransferModel(object):

    def __init__(self):
        # 定义训练和测试图片的变换方法 标准化以及数据增强
        self.train_generator = ImageDataGenerator(rescale=1.0/255.0)
        self.test_generator = ImageDataGenerator(rescale=1.0/255.0)

        # 指定训练数据和测试数据的目录
        self.train_dir = "./data/train"
        self.test_dir = "./data/test"

        # 定义图片相关的网络参数
        self.image_size = (224, 224)
        self.batch_size = 32

        # 定义迁移学习的基类模型
        # 不包含VGG当中3个全连接层的模型加载，并且加载了参数
        self.base_model = VGG16(weights='imagenet', include_top=False)

    def get_local_data(self):
        '''
        读取本地的图片数据以及类别
        :return: 训练数据以及测试数据迭代器
        '''
        # 使用flow_from_derectory
        train_gen = self.train_generator.flow_from_directory(self.train_dir,
                                                             target_size=self.image_size,
                                                             batch_size=self.batch_size,
                                                             class_mode="binary",
                                                             shuffle=True)  # 打乱顺序训练

        test_gen = self.test_generator.flow_from_directory(self.test_dir,
                                                           target_size=self.image_size,
                                                           batch_size=self.batch_size,
                                                           class_mode="binary",
                                                           shuffle=True)
        return train_gen, test_gen


    def refine_base_model(self):
        '''
        微调VGG结构 5blocks后面 + 全局平均池化 （减少迁移学习的参数数量） + 两个全连接层
        :return:
        '''
        # 1-获取原notop模型输出
        x = self.base_model.outputs[0]

        # 2-在输出后面增加我们结构
        x = keras.layers.GlobalAveragePooling2D()(x)
        # 3-定义新的迁移模型
        x = keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        y_predict = keras.layers.Dense(5, activation=tf.nn.softmax)

        # model 定义新模型
        # VGG模型的输入 输出y_predict
        transfer_model = keras.models.Model(inputs=self.base_model.inputs, outputs=y_predict)

        return transfer_model


    def freeze_model(self):
        '''
        冻结VGG模型（5个blocks）
        冻结VGG的多少， 根据你的数据量
        :return:
        '''
        # self.base_model.layers 获取所有层， 返回层的列表
        for layer in self.base_model.layers:
            layer.trainable = False

    def compile(self, model):
        # 编译模型
        modle.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=["accuracy"])
        return None

    def fit_generator(self, model, train_gen, test_gen):

        # 训练模型 model.fit_generator() 不是选择fit()
        modelckpt = keras.callbacks.ModelCheckpoint("./cpkt/transfer_{epoch:02d}-{val_acc:.2f}.h5",
                                                    monitor="val_acc",
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    mode="auto",
                                                    period=1)

        modle.transfer_model.fit_generator(train_gen, epochs=3, validation_data=test_gen, callbacks=[])

        return None



if __name__ == '__main__':

    tm = TransferModel()

    train_gen, test_gen = tm.get_local_data()

    # print(tm.base_model.summary())

    modle = tm.refine_base_model()

    # print(modle)
    tm.freeze_model()

    tm.compile(modle)

    tm.fit_generator(modle, train_gen, test_gen)






    # print(train_gen)  # <keras_preprocessing.image.directory_iterator.DirectoryIterator object at 0x7f7c0e53b6d8>
    # for data in train_gen:
    #     # print(data)
    #     '''
    #     (array([[[[0.46274513, 0.42352945, 0.4666667 ],
    #      [0.86274517, 0.8235295 , 0.86666673],
    #      [0.8745099 , 0.8352942 , 0.87843144],
    #      ...,
    #      [0.2627451 , 0.3254902 , 0.18039216],
    #      [0.16862746, 0.2392157 , 0.08235294],
    #      [0.27058825, 0.34117648, 0.1764706 ]],
    #
    #     [[0.8000001 , 0.7607844 , 0.80392164],
    #      [0.854902  , 0.81568635, 0.8588236 ],
    #      [0.8705883 , 0.8313726 , 0.8745099 ],
    #      ...,
    #      [0.21568629, 0.2784314 , 0.13725491],
    #      [0.18431373, 0.25490198, 0.10588236],
    #      [0.18039216, 0.2509804 , 0.09411766]],
    #
    #
    #     [[0.6156863 , 0.53333336, 0.52156866],
    #      [0.59607846, 0.5137255 , 0.50980395],
    #      [0.6039216 , 0.5176471 , 0.5254902 ],
    #      ...,
    #      [0.5568628 , 0.40000004, 0.3921569 ],
    #      [0.59607846, 0.44705886, 0.4431373 ],
    #      [0.65882355, 0.50980395, 0.5058824 ]]]], dtype=float32), array([2., 4., 4., 2., 1., 1., 1., 0., 4., 2., 0., 0., 4., 2., 3., 4., 2.,
    #    4., 2., 1., 4., 3., 0., 4., 2., 2., 2., 1., 1., 4., 2., 0.],
    #     dtype=float32))
    #     (array([[[[0.23137257, 0.38431376, 0.1764706 ],
    #      [0.227451  , 0.3803922 , 0.17254902],
    #      [0.23137257, 0.38431376, 0.1764706 ],
    #      ...,
    #      [0.27450982, 0.3647059 , 0.18431373],
    #      [0.27058825, 0.3529412 , 0.18431373],
    #      [0.27058825, 0.3529412 , 0.18431373]],
    #     '''
    #     # print(data[0].shape, data[1].shape)
    #     '''
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (32, 224, 224, 3) (32,)
    #     (16, 224, 224, 3) (16,)
    #
    #     '''
    #     # print(data[0][0].shape)
    #     '''
    #     (224, 224, 3)
    #     (224, 224, 3)
    #     (224, 224, 3)
    #     (224, 224, 3)
    #     (224, 224, 3)
    #     (224, 224, 3)
    #     (224, 224, 3)
    #     (224, 224, 3)
    #     (224, 224, 3)
    #
    #     '''






































