import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


batchsz = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

idx = tf.range(60000)
idx = tf.random.shuffle(idx)
x_train, y_train = tf.gather(x, idx[:50000]), tf.gather(y, idx[:50000])
x_val, y_val = tf.gather(x, idx[-10000:]), tf.gather(y, idx[-10000:])
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.map(preprocess).shuffle(50000).batch(batchsz)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)
