
import os
# 数据准备
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
# 防止tensorflow打印很多信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 训练数据，验证数据   x手写数字图像，y对应的标签
(x, y), (x_val, y_val) = datasets.mnist.load_data()
# print('datasets:', xs.shape, ys.shape)

# 转换成tensorflow张量，像素缩放到0-1
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
#  标签转换为向量
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
#  训练集划分小批次，每个批次200个样本
train_dataset = train_dataset.batch(200)
# db = tf.data.Dataset.from_tensor_slices((xs, ys))

# for step, (x, y) in enumerate(db):
#     print(step, x.shape, y, y.shape)


# 模型准备（3层 784-512-256-10）
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):

        #  前向运算
        with tf.GradientTape() as tape:
            # [b, 28, 28] -> [b, 784]
            # -1这里的作用是自动计算第一个维度的大小
            x = tf.reshape(x, (-1, 28*28))

            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # model.trainable_variables 表示获取模型中所有可训练参数的列表
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, "loss:", loss.numpy())

# epoch：训练周期：将整个数据集训练一次的过程
# step：一个训练周期内，模型参数的一次更新

# 执行30个训练周期
def train():
    for epoch in range(30):
        train_epoch(epoch)

if __name__ =='__main__':
    train()