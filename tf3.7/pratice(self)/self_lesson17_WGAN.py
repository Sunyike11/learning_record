import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
class Generator: 
    [b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => [b, 64, 64, 3]
class Discriminator:
    [b, 64, 64, 3] => [b, 1]
"""

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = layers.Dense(3*3*512)
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        x = tf.tanh(x)

        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(64, 3, 3, 'valid')
        self.conv2 = layers.Conv2D(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = self.flatten(x)
        logits = self.fc(x)

        return logits

