import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential

# cell = layers.SimpleRNNCell(3)
# cell.build(input_shape=(None,4))
# print(cell.trainable_variables)

# x = tf.random.normal([4, 80, 100])
# xt0 = x[:,0,:]
# cell = layers.SimpleRNNCell(64)
# out, ht1 = cell(xt0, [tf.zeros([4, 64])])
# print(out.shape, ht1[0].shape)
# print(id(out), id(ht1[0]))


x = tf.random.normal([4, 80, 100])
xt0 = x[:,0,:]
cell = layers.SimpleRNNCell(64)
cell2 = layers.SimpleRNNCell(64)
state0 = [tf.zeros([4, 64])]
state1 = [tf.zeros([4, 64])]

out0, state0 = cell(xt0, state0)
out2, state1 = cell2(out0, state1)

print(out2.shape)

# for word in tf.unstack(x, axis=1):
#     out0, state0 = cell(xt0, state0)
#     out2, state1 = cell2(out0, state1)


# rnn = Sequential([
#     layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
#     layers.SimpleRNN(units, dropout=0.5, unroll=True)
# ])
#
# x = rnn(x)
