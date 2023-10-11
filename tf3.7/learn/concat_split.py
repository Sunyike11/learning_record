import tensorflow as tf

a = tf.ones([4,35,8])
b = tf.ones([4,35,8])

c = tf.concat([a,b],axis=0)
print(c.shape)


# stack: create new dim
# (2, 4, 35, 8)
d = tf.stack([a,b],axis=0)
print(d.shape)

aa,bb = tf.unstack(d,axis=0)
print(aa.shape, bb.shape)

print("unstack:", len(tf.unstack(d,axis=3)))

res = tf.split(d,axis=3, num_or_size_splits=2)
print("split:", len(res))
res = tf.split(d,axis=3, num_or_size_splits=[2,1,5])
print(res[1].shape, res[2].shape)

