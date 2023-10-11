import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential

#  在机器学习中，随机初始化神经网络的权重是一种常见的做法，种子的选择可以确保不同运行下权重的初始化是相同的，以便进行比较和调试。
tf.random.set_seed(2345)

#  堆叠卷积层（Convolutional Layers）的作用：可以学习到越来越复杂和抽象的特征
#  最大池化层（MaxPooling Layers）的作用：减小特征图的空间维度，从而降低网络的计算负担。...保留了图中最显著的特征

# 5 units of conv + max pooling
conv_layers = [
    # unit 1 包含两个卷积层
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

]

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
#  因为y的shape是(64, 1), 所以需要squeeze变成(64,)
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

def main():

    #  [b, 32, 32, 3] -> [b, 1, 1, 512]
    #  [b, 1, 1, 512] reshape ->[b, 512]
    #  conv_layers已经是list
    conv_net = Sequential(conv_layers)
    # x = tf.random.normal([4, 32, 32, 3])
    # out = conv_net(x)
    # print(out.shape)

#  全连接层的主要作用是将卷积神经网络的前面层（通常是卷积层和池化层）提取到的特征映射转换为一个一维的向量，然后通过神经元的连接来进行分类或回归任务。
#  全连接层的输出通常与类别数量相匹配，并且通过softmax激活函数来得出各个类别的概率分布。
    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),

    ])
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])

    #  优化器
    optimizer = optimizers.Adam(lr=1e-4)

    #  "+" 类似列表的拼接
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(50):

        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                #  [b, 32, 32, 3] -> [b, 1, 1, 512]
                out = conv_net(x)
                #  [b, 1, 1, 512] reshape ->[b, 512]
                out = tf.reshape(out, [-1, 512])
                #  [b, 512] -> [b, 100]
                logits = fc_net(out)

                # 100分类 depth=100; [b] -> [b, 100]
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        # test
        total_num, total_correct = 0, 0
        for x, y in test_db:

            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)

if __name__ == '__main__':
    main()