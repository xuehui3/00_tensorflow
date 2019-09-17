'''
VGG16 和 VGG19 的区别
    训练数据和模型不一样

    网络深度不一样：
        VGG16包含了16个隐藏层（13个卷积层和3个全连接层）
        VGG19包含了19个隐藏层（16个卷积层和3个全连接层）
'''
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


def predict():

    model = VGG16()
    print(model.summary())
    # 预测一张图片类别
    # 加载图片并输入到模型当中 （224， 24）是VGG的输入要求
    image = load_img("./flower.jpg", target_size=(224, 224))
    image = img_to_array(image)
    # print(image)
    print(image.shape)

    # 输入卷积中 需要四维结构
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    print(image.shape)


if __name__ == '__main__':
    predict()


