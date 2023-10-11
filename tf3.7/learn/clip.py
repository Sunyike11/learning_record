import tensorflow as tf

# a = tf.constant([0,1,2,3,4,5,6,7,8,9])
#
# print(tf.maximum(a,2))
# print(tf.minimum(a,8))
#
# print(tf.clip_by_value(a,2,8))

# a = a-5
# # maximum(a,0)
# print(tf.nn.relu(a))

#  等比例放缩
# a = tf.random.normal([2,2],mean = 10)
# print(a)
# print('norm a:', tf.norm(a))
# aa = tf.clip_by_norm(a,15)
# print(aa)
# print('norm aa:', tf.norm(aa))

# gradient clipping
# grads, _ = tf.clip_by_global_norm(grads, 15)


# where
# a = tf.random.normal([3,3])
# print(a)
# mask = a>0
# print(mask)
#
# print(tf.boolean_mask(a,mask))
#
# indices = tf.where(mask)
# print(indices, tf.gather_nd(a, indices))
#
# A = tf.ones([3,3])
# B = tf.zeros([3,3])
# print(tf.where(mask, A, B))

# scatter_nd 全0的底板上更新
indices = tf.constant([[4],[3],[1],[7]])
updates = tf.constant([9,10,11,12])
shape = tf.constant([8])

print(tf.scatter_nd(indices, updates, shape))


