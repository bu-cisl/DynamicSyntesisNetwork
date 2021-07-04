import tensorflow as tf

def gat_net(holo):

	##############################################################################
	conv1 = tf.layers.conv2d(inputs=holo, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	pool_flat = tf.layers.flatten(inputs=pool2)
	output = tf.layers.dense(inputs=pool_flat, units=3)
	##############################################################################

	return output
