
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

out = tf.random.uniform([4, 10])
# print(out)
y = tf.range(4)
y = tf.one_hot(y, depth=10)
# print(y)

loss = tf.keras.losses.mse(y, out)
# print(loss)

# 计算平均值，创建为标量
loss = tf.reduce_mean(loss)
# print(loss)


# vector
net = layers.Dense(10)
net.build((4, 8))
print(net.kernel)
# 偏差向量
print(net.bias)



