import tensorflow as tf

# sigmoid_grad
# a = tf.linspace(-10., 10., 10)
# with tf.GradientTape() as tape:
#     tape.watch(a)
#     y = tf.sigmoid(a)
# grads = tape.gradient(y, [a])
# print('x:', a.numpy())
# print('y:', y.numpy())
# print('grad:', grads[0].numpy())


# mse_grad
# x = tf.random.normal([3,4])
# w = tf.random.normal([4,2])
# b = tf.ones([2])
#
# y = tf.constant([1,0,1])
#
# with tf.GradientTape() as tape:
#     tape.watch([w, b])
#     logits = tf.sigmoid(x@w+b)
#     loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y,depth=2), logits))
# grads = tape.gradient(loss, [w, b])
# print('-----------mse_grad-------------')
# print('w grad:', grads[0], 'b grad:', grads[1])


# crossentropy gradient
# x = tf.random.normal([2, 4])
# w = tf.random.normal([4, 3])
# b = tf.zeros([3])
# y = tf.constant([2, 0])
#
# with tf.GradientTape() as tape:
#     tape.watch([w, b])
#     logits = x @ w + b
#     loss = tf.reduce_mean(tf.losses.categorical_crossentropy(
#         tf.one_hot(y, depth=3), logits, from_logits=True
#     ))
#
# grads = tape.gradient(loss, [w, b])
# print('---------crossentropy gradient------------')
# print('w grad:', grads[0], 'b grad:', grads[1])


#  single_output_perceptron
# x = tf.random.normal([1, 3])
# w = tf.ones([3, 1])
# b = tf.ones([1])
# y = tf.constant([1])
#
# with tf.GradientTape() as tape:
#     tape.watch([w, b])
#     prob = tf.sigmoid(x@w+b)
#     loss = tf.reduce_mean(tf.losses.MSE(y, prob))
# grads = tape.gradient(loss, [w,b])
# print('---------single_output_perceptron------------')
# print('w grad:', grads[0], 'b grad:', grads[1])



# multi_output_perceptron
# x = tf.random.normal([2,4])
# w = tf.random.normal([4,3])
# b = tf.ones([3])
#
# y = tf.constant([2,0])
#
# with tf.GradientTape() as tape:
#     tape.watch([w, b])
#     prob = tf.nn.softmax(x@w+b, axis=1)
#     loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y,depth=3), prob))
# grads = tape.gradient(loss, [w, b])
# print('-----------multi_output_perceptron-------------')
# print('w grad:', grads[0], 'b grad:', grads[1])


# chain_rule
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1, b1, w2, b2])
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2
dy2_dy1 = tape.gradient(y2, [y1])[0]
dy1_dw1 = tape.gradient(y1, [w1])[0]
dy2_dw1 = tape.gradient(y2, [w1])[0]

print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)
