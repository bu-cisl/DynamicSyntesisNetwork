import tensorflow as tf
from Layers3 import convolution_3d, deconvolution_3d


def convolution_block(layer_input, alpha, n_channels, num_convolutions):
	x = layer_input
	for i in range(num_convolutions):
		with tf.variable_scope('conv_' + str(i+1)):
			x = tf.nn.relu(convolution_3d(x, alpha, [3, 3, 3, n_channels, n_channels], [1, 1, 1, 1, 1]))
			# batch normalization
			mn, var = tf.nn.moments(x, axes=[0,1,2,3])

			beta1 = tf.Variable(tf.constant(0.0, shape=[n_channels]),name='beta1',trainable=True)
			beta2 = tf.Variable(tf.constant(0.0, shape=[n_channels]),name='beta2',trainable=True)
			beta3 = tf.Variable(tf.constant(0.0, shape=[n_channels]),name='beta3',trainable=True)

			gamma1 = tf.Variable(tf.constant(1.0,shape=[n_channels]),name='gamme1',trainable=True)
			gamma2 = tf.Variable(tf.constant(1.0,shape=[n_channels]),name='gamme2',trainable=True)
			gamma3 = tf.Variable(tf.constant(1.0,shape=[n_channels]),name='gamme3',trainable=True)

			beta = beta1*alpha[0,0] + beta2*alpha[0,1] + beta3*alpha[0,2]
			gamma = gamma1*alpha[0,0] + gamma2*alpha[0,1] + gamma3*alpha[0,2]
			x = tf.nn.batch_normalization(x,mn,var,beta,gamma,1e-3)
	return x + layer_input


def convolution_block_2(layer_input, fine_grained_features, alpha, n_channels, num_convolutions):

	x = tf.concat((layer_input, fine_grained_features), axis=-1)

	with tf.variable_scope('conv_' + str(1)):
		x = convolution_3d(x, alpha, [3, 3, 3, n_channels * 2, n_channels], [1, 1, 1, 1, 1])

	for i in range(1, num_convolutions):
		with tf.variable_scope('conv_' + str(i+1)):
			x = tf.nn.relu(convolution_3d(x, alpha, [3, 3, 3, n_channels, n_channels], [1, 1, 1, 1, 1]))
			# batch normalization
			mn, var = tf.nn.moments(x, axes=[0,1,2,3])

			beta1 = tf.Variable(tf.constant(0.0, shape=[n_channels]),name='beta1',trainable=True)
			beta2 = tf.Variable(tf.constant(0.0, shape=[n_channels]),name='beta2',trainable=True)
			beta3 = tf.Variable(tf.constant(0.0, shape=[n_channels]),name='beta3',trainable=True)

			gamma1 = tf.Variable(tf.constant(1.0,shape=[n_channels]),name='gamme1',trainable=True)
			gamma2 = tf.Variable(tf.constant(1.0,shape=[n_channels]),name='gamme2',trainable=True)
			gamma3 = tf.Variable(tf.constant(1.0,shape=[n_channels]),name='gamme3',trainable=True)

			beta = beta1*alpha[0,0] + beta2*alpha[0,1] + beta3*alpha[0,2]
			gamma = gamma1*alpha[0,0] + gamma2*alpha[0,1] + gamma3*alpha[0,2]
			x = tf.nn.batch_normalization(x,mn,var,beta,gamma,1e-3)
	return x + layer_input


def down_convolution(layer_input, alpha, in_channels):
	with tf.variable_scope('down_convolution'):
		
		return tf.nn.relu(convolution_3d(layer_input, alpha, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1]))


def up_convolution(layer_input, alpha, output_shape, in_channels):
	with tf.variable_scope('up_convolution'):
		return tf.nn.relu(deconvolution_3d(layer_input, alpha, [2, 2, 2, in_channels // 2, in_channels],
												 output_shape, [1, 2, 2, 2, 1]))


def v_net(c5, c4, c3, c2, c1, alpha, input_channels, output_channels=1, n_channels=16):

	with tf.variable_scope('contracting_path'):
		# if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
		with tf.variable_scope('level_5'):
			c52 = up_convolution(c5, alpha, tf.shape(c4), n_channels * 16)

	with tf.variable_scope('expanding_path'):

		with tf.variable_scope('level_4'):
			e4 = convolution_block_2(c52, c4, alpha, n_channels * 8, 3)
			e42 = up_convolution(e4, alpha, tf.shape(c3), n_channels * 8)

		with tf.variable_scope('level_3'):
			e3 = convolution_block_2(e42, c3, alpha, n_channels * 4, 3)
			e32 = up_convolution(e3, alpha, tf.shape(c2), n_channels * 4)

		with tf.variable_scope('level_2'):
			e2 = convolution_block_2(e32, c2, alpha, n_channels * 2, 2)
			e22 = up_convolution(e2, alpha, tf.shape(c1), n_channels * 2)

		with tf.variable_scope('level_1'):
			e1 = convolution_block_2(e22, c1, alpha, n_channels, 1)
			with tf.variable_scope('output_layer'):
				logits = convolution_3d(e1, alpha, [1, 1, 1, n_channels, output_channels], [1, 1, 1, 1, 1])

	return logits
