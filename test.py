import tensorflow as tf
import keras
import numpy as np

g = tf.Graph()
with g.as_default():

	# x = tf.get_variable('x', shape=(3, 10, 5))
	x = tf.constant([[1,2,3]], shape=(1,3))
	w = np.ones(shape=(3,5))
	w[-1,-1] = 3
	print w
	y = tf.layers.dense(x, 5, kernel_initializer=tf.constant_initializer(w), trainable=True)

	with tf.Session(graph = g) as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run((x,y))