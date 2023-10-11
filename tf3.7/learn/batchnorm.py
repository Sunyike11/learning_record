import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

# net = layers.BatchNormalization()
# x = tf.random.normal([2, 3])
# out = net(x)

net = layers.BatchNormalization(axis=3)
x = tf.random.normal([2, 4, 4, 3], mean=1, stddev=0.5)
# out = net(x, training=True)

#  net.trainable_variables: gamma beta
#  net.variables: gamma beta moving_mean moving_variance
# print(net.trainable_variables)
# print(net.variables)

for i in range(100):
    out = net(x, training=True)
print(net.variables)