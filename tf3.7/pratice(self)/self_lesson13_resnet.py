import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

"""
需要两个类：basic_block和res_net
res_net包含预处理层、前向运算、全连接层
-预处理层：conv-bn-relu-Maxpool

-前向运算指res_block（4个）
-每个res_block包含几个basic_block
-basic_block：conv1-bn1-ru —> conv2-bn2 -->identity = downsample(inputs)  

-全连接层： avg - fc=layers.Dense

"""

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x


    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # identity一定要用原来的inputs
        identity = self.downsample(inputs)
        out = layers.add([x, identity])
        out = tf.nn.relu(out)

        return out


class ResNet(keras.Model):
    def __init__(self, lay_dims, classes_num=100):
        super(ResNet, self).__init__()

        # -预处理层：conv-bn-relu-Maxpool
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        # -前向运算
        self.layer1 = self.BuildResBlock(64, lay_dims[0])
        self.layer2 = self.BuildResBlock(128, lay_dims[1], stride=2)
        self.layer3 = self.BuildResBlock(256, lay_dims[2], stride=2)
        self.layer4 = self.BuildResBlock(512, lay_dims[3], stride=2)

        # -全连接层： avg - fc=layers.Dense
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(classes_num)

    def call(self, inputs, training=None):
        out = self.stem(inputs)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out)

        return out

    def BuildResBlock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks

def resnet18():
    return ResNet([2, 2, 2, 2])