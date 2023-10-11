import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, datasets, preprocessing, optimizers

"""
主要的问题是在 MyRNN 类的 call 方法中，你正在尝试在循环中使用 SimpleRNNCell 处理时间步
但是在 TensorFlow 2.x 中，要求在 @tf.function 内部的循环中不包含 Keras 符号张量。

"""

tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')


total_words = 10000
max_review_len = 80
embedding_len = 100
batchsz = 128
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
# x:[b, 80] y:0-差评；1-好评
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# y_train = tf.cast(y_train, dtype=tf.float32)
# y_test = tf.cast(y_test, dtype=tf.float32)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# 如果数据样本数不能整除批次大小 (batch size)，是否要丢弃剩余的不足一批的样本。
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(x_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)



class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        #  Embedding 层，用于将整数序列映射为密集的词嵌入；将词汇中的每个词转换为连续的向量表示。
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        # [b,80,100], h_dim:64
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        # 二分类
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x = self.fc(out1)
        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64
    epochs = 4
    model = MyRNN(units)
    # experimental_run_tf_function参数用于控制是否在模型的训练过程中启用TF Function
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'], experimental_run_tf_function=False)
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)

if __name__ == '__main__':
    main()
