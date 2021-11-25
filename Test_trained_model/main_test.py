# V-net for biological image segmentation

import tensorflow as tf
import numpy as np
from Vnet_3D_Encoder import vnet_enc
from Vnet_3D_dynamic_dec import v_net
from gatnet import gat_net
from utils import get_all_variables_from_top_scope
from slicewise_metrics import dice_opt_gpu, slicewise_01
import matplotlib.pyplot as plt
import os
import time
###
from data_io import batch_generator, Mouse, save_3d_tiff
###

###
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Main program begins here:
step = 4 #downsample x
input_channels = 1
output_channels = 1
batch_size = 1
n_channels = 16 #for default vnet
nx = 128
ny = 128
nz = 100
n_tr_patches = 980
n_te_patches = 64

ntrain_cube = 20
ntest_cube = 4
ntrain_subcube = 200
train_epochs = 500
save_every_x_epoch = 1
train_iters = train_epochs*(ntrain_cube*ntrain_subcube)

# Load data 
t = time.time()
################################################
# Load data
test_list = [[[],[],[],[],[]], [[],[],[],[],[]]]
data_type = ['sim']
conc_n = 1
data_n = 1
print('reading data')
for k in range(len(data_type)):
	for conc in range(conc_n):
		for obj in range(data_n):
				fn = './test_data/conc_%d/obj_%d'%(conc+1,obj+1)
				test_list[k][conc].append(Mouse(fn, is_test=True, is_exp=bool(k)))
################################################
################################################
elapsed = time.time() - t
print('time to read data is %f'%elapsed)



############################################################################################################
# Create the model
############################################################################################################
x_holo = tf.placeholder(dtype=tf.float32, shape=(None, int(nx/step), int(ny/step), 1), name="x_holo")
x = tf.placeholder(dtype=tf.float32, shape=(None, nx, ny, nz, input_channels), name="x")
y_ = tf.placeholder(dtype=tf.float32, shape=(None, nx, ny, nz, output_channels), name="y_")

# Pretrained Encoders
with tf.variable_scope('model2_enc') as topscope2:
	m2c5, m2c4, m2c3, m2c2, m2c1 = vnet_enc(x, input_channels, output_channels, n_channels)
	m2_enc_vars = get_all_variables_from_top_scope(tf, topscope2)

with tf.variable_scope('model3_enc') as topscope3:
	m3c5, m3c4, m3c3, m3c2, m3c1 = vnet_enc(x, input_channels, output_channels, n_channels)
	m3_enc_vars = get_all_variables_from_top_scope(tf, topscope3)

with tf.variable_scope('model4_enc') as topscope4:
	m4c5, m4c4, m4c3, m4c2, m4c1 = vnet_enc(x, input_channels, output_channels, n_channels)
	m4_enc_vars = get_all_variables_from_top_scope(tf, topscope4)

# Gating network
with tf.variable_scope('model_g') as topscope_g:
	# Gating
	gat_alpha = gat_net(x_holo)
	alpha = tf.reshape(gat_alpha,[batch_size,3])
	alpha_s = tf.nn.softmax(alpha)
	gat_net_vars = get_all_variables_from_top_scope(tf, topscope_g)

# Synthesis network
with tf.variable_scope('model_e') as topscope_s:
	c5 = alpha_s[0,0]*m2c5 + alpha_s[0,1]*m3c5 + alpha_s[0,2]*m4c5
	c4 = alpha_s[0,0]*m2c4 + alpha_s[0,1]*m3c4 + alpha_s[0,2]*m4c4
	c3 = alpha_s[0,0]*m2c3 + alpha_s[0,1]*m3c3 + alpha_s[0,2]*m4c3
	c2 = alpha_s[0,0]*m2c2 + alpha_s[0,1]*m3c2 + alpha_s[0,2]*m4c2
	c1 = alpha_s[0,0]*m2c1 + alpha_s[0,1]*m3c1 + alpha_s[0,2]*m4c1
	y_conv = v_net(c5, c4, c3, c2, c1, alpha_s, input_channels, output_channels, n_channels)
	vars_all_e = get_all_variables_from_top_scope(tf, topscope_s)
	y_pred = tf.nn.sigmoid(y_conv, name="y_conv_pred")

# model variable names
m2_dec_vars = []
m3_dec_vars = []
m4_dec_vars = []
for i in range(0,len(vars_all_e),3):
	m2_dec_vars.append(vars_all_e[i])
	m3_dec_vars.append(vars_all_e[i+1])
	m4_dec_vars.append(vars_all_e[i+2])

############################################################################################################

# Loss: cross entropy
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(y_,[batch_size*nx*ny*nz,output_channels]), logits=tf.reshape(y_conv,[batch_size*nx*ny*nz,output_channels]))
cross_entropy = tf.reduce_mean(cross_entropy, name="cross_entropy")
print(cross_entropy)

# Optimizer: Adam
optimizer = tf.train.AdamOptimizer(1e-5)
grads_and_vars = optimizer.compute_gradients(cross_entropy, var_list=gat_net_vars+vars_all_e+m2_enc_vars+m3_enc_vars+m4_enc_vars)
train_step = optimizer.apply_gradients(grads_and_vars, name='train_step')

# Computation of slicewise dice score
gt_ = tf.placeholder(dtype=tf.float32, shape=(1024, 1024, 100), name="gt_")
pred_ = tf.placeholder(dtype=tf.float32, shape=(1024, 1024, 100), name="pred_")
dice_, seg_, thresh_ = dice_opt_gpu(gt_, pred_)

# Accuracy scores
epoch = 0
tr_loss = []
te_loss = []
te_loss_tmp = np.zeros((int(n_te_patches/batch_size)))
pr_te_tmp = np.zeros((int(n_te_patches/batch_size)))
saver = tf.train.Saver(max_to_keep=100)

saver2e = tf.train.Saver(var_list=m2_enc_vars)
saver3e = tf.train.Saver(var_list=m3_enc_vars)
saver4e = tf.train.Saver(var_list=m4_enc_vars)

saver2d = tf.train.Saver(var_list=m2_dec_vars)
saver3d = tf.train.Saver(var_list=m3_dec_vars)
saver4d = tf.train.Saver(var_list=m4_dec_vars)

dnn_conc =5

with tf.Session() as sess: # laun ch the model in an interactive session
	sess.run(tf.global_variables_initializer()) # create an operation to initialize the variables we created
	sess.run(tf.local_variables_initializer()) # needed to compute tp, tn, fp, fn

	saver.restore(sess, './chkpt/my-model-1')

	for k in range(len(data_type)):
		for conc in range(conc_n):
			for obj in range(data_n):
				# Test on test-data
				print('[Testing] [DNN: %d] [%s] [conc: %d] [obj: %d]' % (dnn_conc + 1, data_type[k], conc + 1, obj + 1))
				testmouse = test_list[k][conc][obj]

				y_te_cube_pred = np.empty(testmouse.shape, dtype='float32')

				for test_batch in batch_generator([testmouse], batch_size):
					feed_dict = {
						x_holo: test_batch.get_holos_in_batch()[:, ::step, ::step, :],
						x: test_batch.get_original_images_in_batch(),
					}

					y_pred_p = sess.run((y_pred), feed_dict)

					# put data into full FoV volume for testmouse
					for patch_i in range(batch_size):
						y_te_cube_pred[test_batch.get_index_list()[patch_i]] = y_pred_p[patch_i, ..., 0]

				# Make folders to save diagnostics
				resdir = 'test_results/dnn_conc_%d/%s/conc_%d/obj_%d' % (dnn_conc + 1, data_type[k], conc + 1, obj + 1)
				if not os.path.exists(resdir):
					os.makedirs(resdir)
				save_3d_tiff(resdir, {'predicted': y_te_cube_pred})
