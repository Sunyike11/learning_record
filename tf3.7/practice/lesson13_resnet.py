import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

class BasicBlock(layers.Layer):

    #  stride=1 默认不做采样
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        # relu层没有参数，可以使用两次
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        #  self.downsample 用于创建一个用于下采样的层序列。如果 stride 不等于 1，就需要添加一个卷积层来进行下采样。
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x


    def call(self, inputs, training=None):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output

class ResNet(keras.Model):

    def __init__(self, layer_dims, num_classes=100):  # [2, 2, 2, 2]: 4个res_block, 每个res_block包含两个basic_block; 预设100类
        super(ResNet, self).__init__()

        # 预处理层
        #  strides参数是一个包含两个值的元组，分别表示在水平方向和垂直方向上的步长。
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2),strides=(1, 1),padding='same')])

        # 前向运行过程
        self.layer1 = self.build_resblock(64, layer_dims[0])
        # h,w 维度减小
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # 全连接层
        # output: [b, 512, h, w] 不确定
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)


    def call(self, inputs, training=None):

        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b, c] 不需要reshape
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        #  may down sample
        #  这个下采样操作只在残差块的第一个 BasicBlock 层中执行，而后续的 BasicBlock 层通常不包含下采样。
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

def resnet18():

    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])