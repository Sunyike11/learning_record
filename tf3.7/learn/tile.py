import tensorflow as tf

a = tf.reshape(tf.range(9), [3,3])
# print(a)
# print(tf.pad(a,[[1,1],[0,1]]))

#  image padding
b = tf.random.normal([4,28,28,3])
c = tf.pad(b, [[0,0],[2,2],[2,2],[0,0]])

print(tf.tile(a,[1,3]))