import numpy as np
import tensorflow as tf


a = tf.reshape(tf.range(12), (2,3,2))
b = tf.reshape(tf.range(12), (2,2,3))
c = tf.matmul(a,b)
print c


print 'np'
a = np.random.normal(size=(2, 5, 3))
b = np.random.normal(size=(3, 4))
print a
print b
print np.matmul(a,b)


print 'tf'
x = tf.placeholder(tf.float32, shape=(None, 5, 3 ))
print x
w = tf.random_normal((3,4))
print w
ww = tf.tile(tf.expand_dims(w, 0), (tf.shape(x)[0], 1, 1))
print ww
out = tf.matmul(x,ww)
print out