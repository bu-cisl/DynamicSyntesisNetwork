import tensorflow as tf
from Layers import convolution_3d, deconvolution_3d


def convolution_block(x, n_channels, num_convolutions=1):

	mn1, var1 = tf.nn.moments(x, axes=[0, 1, 2, 3])
	beta1 = tf.Variable(tf.constant(0.0, shape=[1]), name='beta', trainable=True)
	gamma1 = tf.Variable(tf.constant(1.0, shape=[1]), name='gamme', trainable=True)
	x1 = tf.nn.batch_normalization(x, mn1, var1, beta1, gamma1, 1e-3)

	for i in range(num_convolutions):
		with tf.variable_scope('conv_' + str(i+1)):
			x2 = tf.nn.relu(convolution_3d(x1, [3, 3, 3, n_channels, 1], [1, 1, 1, 1, 1]))
			# batch normalization
			mn, var = tf.nn.moments(x2, axes=[0,1,2,3])
			beta = tf.Variable(tf.constant(0.0, shape=[1]),name='beta',trainable=True)
			gamma = tf.Variable(tf.constant(1.0,shape=[1]),name='gamme',trainable=True)
			x2 = tf.nn.batch_normalization(x2,mn,var,beta,gamma,1e-3)
	return x2, x1

def convolution_block_2(layer_input, fine_grained_features, n_channels, num_convolutions):

	x = tf.concat((layer_input, fine_grained_features), axis=-1)

	with tf.variable_scope('conv_' + str(1)):
		x = convolution_3d(x, [3, 3, 3, n_channels * 2, n_channels], [1, 1, 1, 1, 1])

	for i in range(1, num_convolutions):
		with tf.variable_scope('conv_' + str(i+1)):
			x = tf.nn.relu(convolution_3d(x, [3, 3, 3, n_channels, n_channels], [1, 1, 1, 1, 1]))
			# batch normalization
			mn, var = tf.nn.moments(x, axes=[0,1,2,3])
			beta = tf.Variable(tf.constant(0.0, shape=[n_channels]),name='beta',trainable=True)
			gamma = tf.Variable(tf.constant(1.0,shape=[n_channels]),name='gamme',trainable=True)
			x = tf.nn.batch_normalization(x,mn,var,beta,gamma,1e-3)
	return x + layer_input


def down_convolution(layer_input, in_channels):
	with tf.variable_scope('down_convolution'):
		
		return tf.nn.relu(convolution_3d(layer_input, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1]))


def up_convolution(layer_input, output_shape, in_channels):
	with tf.variable_scope('up_convolution'):
		return tf.nn.relu(deconvolution_3d(layer_input, [2, 2, 2, in_channels // 2, in_channels],
												 output_shape, [1, 2, 2, 2, 1]))

#############################################################################################
def xavier_uniform_dist_conv3d(shape):
	with tf.variable_scope('xavier_glorot_initializer'):
		denominator = tf.cast((tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:])), tf.float32)
		lim = tf.sqrt(6. / denominator)
		return tf.random_uniform(shape, minval=-lim, maxval=lim)
		
def weight_variable2(shape):
	"""weight_variable generates a weight variable of a given shape."""
	shape_ = tf.cast(shape, tf.float32)
	denominator = tf.cast((tf.reduce_prod(shape_[:2]) * tf.reduce_sum(shape_[2:])), tf.float32)
	lim = tf.sqrt(4. / denominator)
	return tf.random_uniform(shape, minval=-lim, maxval=lim)

def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=1.0)
	return tf.Variable(initial, name='weights')

def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(1.0, shape=shape)
	return tf.Variable(initial, name='biases')

def bias_variable2(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, name='biases')

def convolution_fcn1(layer_input):
	
	# Fully connected layer - after passing throught the network, our 128x128x100 volume
	# is down to 8x8x7x256 feature maps -- this layer maps them to 1024 features.
	W_fc1 = weight_variable([8*8*7*3, 1024])
	b_fc1 = bias_variable([1024])

	W_fc2 = weight_variable([1024, 1024])
	b_fc2 = bias_variable([1024])

	W_fc3 = weight_variable([1024, 1024])
	b_fc3 = bias_variable([1024])

	l1 = tf.nn.relu(tf.matmul(layer_input, W_fc1) + b_fc1)
	l2 = tf.nn.relu(tf.matmul(l1, W_fc2) + b_fc2)
	l3 = tf.nn.relu(tf.matmul(l2, W_fc3) + b_fc3)
	return l1,l2,l3

def convolution_fcn2(layer_input):
	
	# Fully connected layer - after passing through the network, our 128x128x100 volume
	# is down to 8x8x7x256 feature maps -- this layer maps them to 1024 features.
	W_fc2 = weight_variable([1024, 3])
	b_fc2 = bias_variable([3])

	return tf.matmul(layer_input, W_fc2) + b_fc2
#############################################################################################


def gat_net(m2c5, m3c5, m4c5, input_channels, output_channels=1, n_channels=16):

	with tf.variable_scope('level0_m2'):
		m2c5_1ch, m2c5_bn = convolution_block(m2c5, n_channels * 16)
	with tf.variable_scope('level0_m3'):
		m3c5_1ch, m3c5_bn = convolution_block(m3c5, n_channels * 16)
	with tf.variable_scope('level0_m4'):
		m4c5_1ch, m4c5_bn = convolution_block(m4c5, n_channels * 16)

	with tf.variable_scope('fcn_layer_1'):
		fcn_in = tf.concat((m2c5_1ch, m3c5_1ch, m4c5_1ch), axis=-1)
		fcn_in_flat = tf.reshape(fcn_in, [-1, 8*8*7*3])
		l1,l2,l3 = convolution_fcn1(fcn_in_flat) #1024 out channels

	with tf.variable_scope('fcn_layer_2'):
		gat_wts = convolution_fcn2(l3) #3 out channels

	return gat_wts, m2c5_1ch, m3c5_1ch, m4c5_1ch, l1,l2,l3, fcn_in, m2c5_bn, m3c5_bn, m4c5_bn
