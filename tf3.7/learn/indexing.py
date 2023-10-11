import tensorflow as tf

a = tf.ones([1, 5, 5, 3])
# a[0][0]->shape=(5,3)
# print(a[0][0])

b = tf.random.normal([4, 28, 28, 3])
# (28, 28, 3)
# (28, 3)
# print(b[1].shape)
# print(b[1, 2].shape)

# [start:end)
# c = tf.range(10)
# print(c[-1:])

# b[0].shape
# print(b[0,:,:,:].shape)
# print(b[:,:,:,0].shape)

#  start:end:step
# print(b[:,0:28:2,0:28:2,:].shape)

# b[0,:,:,:].shape == b[0,...].shape
# print(b[0,...].shape)


#  selective Indexing
# (2, 28, 28, 3)
# (4, 5, 28, 3)
# print(tf.gather(b, axis=0, indices=[2,3]).shape)
# print(tf.gather(b, axis=1, indices=[3,2,7,9,16]).shape)

#  两个维度同时操作
# (28, 3)
# ()
# (1,)
# print(tf.gather_nd(b, [0,1]).shape)
# print(tf.gather_nd(b, [0,1,1,1]).shape)
# print(tf.gather_nd(b, [[1,1,1,1]]).shape)

# (2, 28, 3)
# print(tf.gather_nd(b, [[0,0],[1,1]]).shape)


print(tf.boolean_mask(b, mask=[True,True,False,False]).shape)
print(tf.boolean_mask(b, mask=[True,True,False], axis=3).shape)
c = tf.ones([2,3,5])
print(tf.boolean_mask(c, mask=[[True,False,False],[False,True,False]]))
