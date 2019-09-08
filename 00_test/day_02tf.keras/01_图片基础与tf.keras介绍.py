from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Input
import tensorflow as tf

def main():

    # image = load_img("./bus/300.jpg", target_size=(300, 300))
    # print(image)
    # # 输入到tensorflow 做处理
    # image= img_to_array(image)
    #
    # print(image.shape)
    # print(image)

    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    # print(x_train.shape)
    # print(y_train.shape)

    # tf.keras.Sequential构建类似管道的模型
    '''
    Flatten:将输入数据进行形状改变展开
    Dense:添加一层神经元
        Dense(units,activation=None,**kwargs)
            units:神经元个数
            activation：激活函数,参考tf.nn.relu,tf.nn.softmax,tf.nn.sigmoid,tf.nn.tanh
            **kwargs:输入上层输入的形状，input_shape=()
    :return:
    '''
    # sequential adj, 连续的，相继的，有顺序的
    model_first = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation=tf.nn.relu),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])
    print(model_first)

    # 通过Model建立模型
    data = Input(shape=(784, ))
    print(data)
    out = Dense(64)(data)
    print(out)
    model_sec = Model(inputs=data, outputs=out)
    print(model_sec)

    print('获取模型结构列表', model_first.layers, model_sec.layers)

    print(model_first.inputs, model_first.outputs)

    # 模型结构参数
    print(model_first.summary())


if __name__ == '__main__':
    main()














