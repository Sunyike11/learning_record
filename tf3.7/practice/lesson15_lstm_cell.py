import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, datasets, preprocessing, optimizers

tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')


total_words = 10000
max_review_len = 80
embedding_len = 100
batchsz = 128
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(x_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)

class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        self.state0 = [tf.zeros([batchsz, units]), tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units]), tf.zeros([batchsz, units])]

        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.rnn_cell0 = layers.LSTMCell(units, dropout=0.5)
        self.rnn_cell1 = layers.LSTMCell(units, dropout=0.5)
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
    import time
    t0 = time.time()
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'], experimental_run_tf_function=False)
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)
    t1 = time.time()
    print('time cost:', t1-t0)

if __name__ == '__main__':
    main()
