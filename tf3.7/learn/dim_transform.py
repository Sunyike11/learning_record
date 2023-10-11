import tensorflow as tf

a = tf.random.normal([4, 28, 28, 3])

# reshape
# print(tf.reshape(a, [4, -1, 3]).shape)

# transpose
print(tf.transpose(a).shape)
print(tf.transpose(a, perm=[1,0,3,2]).shape)

# expand_dim squeeze_dim
b = tf.random.normal([4,35,8])

# print(tf.expand_dims(b, axis=0).shape)
# print(tf.expand_dims(b, axis=1).shape)
# print(tf.expand_dims(b, axis=-1).shape)

# print(tf.squeeze(tf.zeros([1,2,1,1,3])).shape)
# print(tf.squeeze(tf.zeros([1,2,1,1,3]),axis=2).shape)
