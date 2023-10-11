import tensorflow as tf

x = tf.random.normal([1,784])
w = tf.random.normal([784,2])
b = tf.zeros([2])

logits = x @ w + b
loss = tf.losses.categorical_crossentropy([0,1], logits, from_logits=True)
print(loss)