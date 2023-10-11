import tensorflow as tf
from tensorflow.keras import optimizers, datasets
from self_lesson13_resnet import resnet18

tf.random.set_seed(2345)

def preprocess(x,y):
    x = tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(10000).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)

def main():

    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = optimizers.Adam(lr=1e-3)

    for epoch in range(50):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, 'loss:', float(loss))

        #test
        total_correct, total_num = 0, 0
        for x,y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print('acc:', acc)





if __name__ == '__main__':
    main()