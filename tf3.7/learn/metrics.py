import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(60000).batch(128)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(128)

db_iter = iter(db)
sample = next(db_iter)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=[None, 28*28])

optimizer = optimizers.Adam(lr=0.01)

#
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()


def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28*28])
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

                #
                loss_meter.update_state(loss)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', loss_meter.result().numpy())
                loss_meter.reset_states()

        # test
        # total_correct, total_sum = 0, 0
        acc_meter.reset_states()
        for step, (x, y) in enumerate(db_test):
            x = tf.reshape(x, [-1, 28*28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            acc_meter.update_state(y, pred)
            # correct = tf.equal(pred, y)
            # correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

        #     total_correct += int(correct)
        #     total_sum += x.shape[0]
        #
        # acc = total_correct / total_sum
        print('acc:', acc_meter.result().numpy())

if __name__ == '__main__':
    main()