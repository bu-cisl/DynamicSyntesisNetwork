""" Implementation of Dynamic Synthesis Network
Author: Waleed Tahir
Email: waleedt@bu.edu
Publication: W.Tahir, L. Tian, "Adaptive 3D descattering with a dynamic synthesis network", arXiv 2107.00484
Publication Link: https://arxiv.org/abs/2107.00484
Last Updated: July 04, 2021
"""

import tensorflow as tf
import numpy as np
from vnet_3D_encoder import vnet_enc
from vnet_3D_synthesized_decoder import v_net
from gatnet import gat_net
from utils import get_all_variables_from_top_scope
from data_io import batch_generator, Data, save_3d_tiff
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


### Main program begins here:
downsample_step = 4
input_channels = 1
output_channels = 1
batch_size = 1
n_channels = 16
nx = 128
ny = 128
nz = 100


# Load data 
t = time.time()
################################################
################################################
# Load data
test_list = [[],[],[],[],[]]
conc_n = 5
data_n = 10
print('reading data')
for conc in range(conc_n):
	for obj in range(data_n):
			fn = './test_data/conc_%d/obj_%d'%(conc+1,obj+1)
			test_list[conc].append(Data(fn, is_test=True))
################################################
################################################
elapsed = time.time() - t
print('time to read data is %f'%elapsed)



############################################################################################################
# Create the model
############################################################################################################
x_holo = tf.placeholder(dtype=tf.float32, shape=(None, int(nx/downsample_step), int(ny/downsample_step), 1), name="x_holo")
x = tf.placeholder(dtype=tf.float32, shape=(None, nx, ny, nz, input_channels), name="x")
y_ = tf.placeholder(dtype=tf.float32, shape=(None, nx, ny, nz, output_channels), name="y_")

# DSN Encoders
with tf.variable_scope('model2_enc') as topscope2:
	m2c5, m2c4, m2c3, m2c2, m2c1 = vnet_enc(x, input_channels, output_channels, n_channels)
	m2_enc_vars = get_all_variables_from_top_scope(tf, topscope2)

with tf.variable_scope('model3_enc') as topscope3:
	m3c5, m3c4, m3c3, m3c2, m3c1 = vnet_enc(x, input_channels, output_channels, n_channels)
	m3_enc_vars = get_all_variables_from_top_scope(tf, topscope3)

with tf.variable_scope('model4_enc') as topscope4:
	m4c5, m4c4, m4c3, m4c2, m4c1 = vnet_enc(x, input_channels, output_channels, n_channels)
	m4_enc_vars = get_all_variables_from_top_scope(tf, topscope4)

# Gating network (GTN)
with tf.variable_scope('model_g') as topscope_g:
	# Gating
	gat_alpha = gat_net(x_holo)
	alpha = tf.reshape(gat_alpha,[batch_size,3])
	alpha_s = tf.nn.softmax(alpha)
	gat_net_vars = get_all_variables_from_top_scope(tf, topscope_g)

# DSN Decoder
with tf.variable_scope('model_e') as topscope_s:
	c5 = alpha_s[0,0]*m2c5 + alpha_s[0,1]*m3c5 + alpha_s[0,2]*m4c5
	c4 = alpha_s[0,0]*m2c4 + alpha_s[0,1]*m3c4 + alpha_s[0,2]*m4c4
	c3 = alpha_s[0,0]*m2c3 + alpha_s[0,1]*m3c3 + alpha_s[0,2]*m4c3
	c2 = alpha_s[0,0]*m2c2 + alpha_s[0,1]*m3c2 + alpha_s[0,2]*m4c2
	c1 = alpha_s[0,0]*m2c1 + alpha_s[0,1]*m3c1 + alpha_s[0,2]*m4c1
	y_conv = v_net(c5, c4, c3, c2, c1, alpha_s, input_channels, output_channels, n_channels)
	vars_all_e = get_all_variables_from_top_scope(tf, topscope_s)
	y_pred = tf.nn.sigmoid(y_conv, name="y_conv_pred")

# Model variable names
m2_dec_vars = []
m3_dec_vars = []
m4_dec_vars = []
for i in range(0,len(vars_all_e),3):
	m2_dec_vars.append(vars_all_e[i])
	m3_dec_vars.append(vars_all_e[i+1])
	m4_dec_vars.append(vars_all_e[i+2])


############################################################################################################

# Loss: binary cross entropy
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(y_,[batch_size*nx*ny*nz,output_channels]), logits=tf.reshape(y_conv,[batch_size*nx*ny*nz,output_channels]))
cross_entropy = tf.reduce_mean(cross_entropy, name="cross_entropy")
weight_decay = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
total_loss = cross_entropy + weight_decay


# Optimizer: Adam
optimizer = tf.train.AdamOptimizer(1e-5)
grads_and_vars = optimizer.compute_gradients(total_loss, var_list=gat_net_vars+vars_all_e+m2_enc_vars+m3_enc_vars+m4_enc_vars)
train_step = optimizer.apply_gradients(grads_and_vars, name='train_step')

# Computation of slicewise dice score
gt_ = tf.placeholder(dtype=tf.float32, shape=(1024, 1024, 100), name="gt_")
pred_ = tf.placeholder(dtype=tf.float32, shape=(1024, 1024, 100), name="pred_")

# Launch TF session
tr_loss = []
te_loss = []
saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer()) 

	saver.restore(sess, './chkpt/my-model-1')

	
	for conc in range(conc_n):
		for obj in range(data_n):
			# Test on test-data
			print('[Testing] [conc: %d] [obj: %d]' % (conc + 1, obj + 1))
			testmouse = test_list[conc][obj]

			y_te_cube_pred = np.empty(testmouse.shape, dtype='float32')

			for test_batch in batch_generator([testmouse], batch_size):
				feed_dict = {
					x_holo: test_batch.get_holos_in_batch()[:, ::downsample_step, ::downsample_step, :],
					x: test_batch.get_original_images_in_batch(),
				}

				y_pred_p = sess.run((y_pred), feed_dict)

				# put data into full FoV volume for testmouse
				for patch_i in range(batch_size):
					y_te_cube_pred[test_batch.get_index_list()[patch_i]] = y_pred_p[patch_i, ..., 0]

			# Make folders to save diagnostics
			resdir = 'test_results/conc_%d/obj_%d' % (conc + 1, obj + 1)
			if not os.path.exists(resdir):
				os.makedirs(resdir)
			save_3d_tiff(resdir, {'predicted': y_te_cube_pred})


