import  tensorflow as tf

a = tf.ones([4,28,28,3])
# norm: 2范数
print(tf.norm(a))

b = tf.ones([2,2])
print(tf.norm(b,ord=2,axis=1))
print(tf.norm(b,ord=1))
print(tf.norm(b,ord=1,axis=0))

a = tf.random.normal([4,10])
print("min(a):",tf.reduce_min(a), "max(a):", tf.reduce_max(a), "mean(a):", tf.reduce_mean(a))

# 最大值位置
print(tf.argmax(a))
print(tf.argmin(a).shape)


# 比较
a = tf.constant([1,2,3,4,4])
b = tf.range(5)
print(tf.equal(a, b))
res = tf.equal(a, b)
# axis 参数设置为 1，则会对每一行进行求和 ; 如果将 axis 参数设置为 0，则会对每列进行求和
print(tf.reduce_sum(tf.cast(res, dtype=tf.int32)))

# 重复的
# 用tf.gather(unique,idx)还原
a = tf.constant([4,2,2,4,3])
print(tf.unique(a))
