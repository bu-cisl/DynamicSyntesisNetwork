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
import tensorflow.contrib.slim as slim
from tensorflow.python.layers import base

###

###
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Main program begins here:
step = 4  # downsample x
input_channels = 1
output_channels = 1
batch_size = 1
n_channels = 16  # for default vnet
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
train_iters = train_epochs * (ntrain_cube * ntrain_subcube)

# Load data
# t = time.time()
################################################
################################################
# TRAIN_MOUSE_INDEX = range(5, 75)
# MOUSE_N = 75
# all_mouse_idx = range(0, MOUSE_N)
# tr_mouse_idx = TRAIN_MOUSE_INDEX
# te_mouse_idx = [e for e in all_mouse_idx if e not in TRAIN_MOUSE_INDEX]

# Load data
# train_list = []
# test_list = []
# for mouse_i in range(MOUSE_N):
#     if mouse_i in tr_mouse_idx:
#         train_list.append(Mouse(mouse_i, is_test=False))
#     else:
#         test_list.append(Mouse(mouse_i, is_test=True))
################################################
################################################
# elapsed = time.time() - t
# print('time to read data is %f' % elapsed)

############################################################################################################
# Create the model
############################################################################################################
x_holo = tf.placeholder(dtype=tf.float32, shape=(None, int(nx / step), int(ny / step), 1), name="x_holo")
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
    alpha = tf.reshape(gat_alpha, [batch_size, 3])
    alpha_s = tf.nn.softmax(alpha)
    gat_net_vars = get_all_variables_from_top_scope(tf, topscope_g)

# Synthesis network
with tf.variable_scope('model_e') as topscope_s:
    c5 = alpha_s[0, 0] * m2c5 + alpha_s[0, 1] * m3c5 + alpha_s[0, 2] * m4c5
    c4 = alpha_s[0, 0] * m2c4 + alpha_s[0, 1] * m3c4 + alpha_s[0, 2] * m4c4
    c3 = alpha_s[0, 0] * m2c3 + alpha_s[0, 1] * m3c3 + alpha_s[0, 2] * m4c3
    c2 = alpha_s[0, 0] * m2c2 + alpha_s[0, 1] * m3c2 + alpha_s[0, 2] * m4c2
    c1 = alpha_s[0, 0] * m2c1 + alpha_s[0, 1] * m3c1 + alpha_s[0, 2] * m4c1
    y_conv = v_net(c5, c4, c3, c2, c1, alpha_s, input_channels, output_channels, n_channels)
    vars_all_e = get_all_variables_from_top_scope(tf, topscope_s)
    y_pred = tf.nn.sigmoid(y_conv, name="y_conv_pred")

# model variable names
# m2_dec_vars = []
# m3_dec_vars = []
# m4_dec_vars = []
# for i in range(0, len(vars_all_e), 3):
#     m2_dec_vars.append(vars_all_e[i])
#     m3_dec_vars.append(vars_all_e[i + 1])
#     m4_dec_vars.append(vars_all_e[i + 2])

# # Print expert network variables
# [print(v.name) for v in m2_dec_vars]
# print('--------------------')
# [print(v.name) for v in m3_dec_vars]
# print('--------------------')
# [print(v.name) for v in m4_dec_vars]
# print('--------------------')

# # Print expert network variables
# [print(v.name) for v in m2_enc_vars]
# print('--------------------')
# [print(v.name) for v in m3_enc_vars]
# print('--------------------')
# [print(v.name) for v in m4_enc_vars]
# print('--------------------')

# # Print trainable variables
# [print(v.name) for v in m_vars_train]
# print('--------------')
# first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model_s")
# [print(v.name) for v in first_train_vars]


############################################################################################################

# Loss: cross entropy
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.reshape(y_, [batch_size * nx * ny * nz, output_channels]),
    logits=tf.reshape(y_conv, [batch_size * nx * ny * nz, output_channels]))
cross_entropy = tf.reduce_mean(cross_entropy, name="cross_entropy")
weight_decay = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
total_loss = cross_entropy + weight_decay

# Optimizer: Adam
# train_vars = gat_net_vars + vars_all_e
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=gat_net_vars, name="train_step") #train_step is an operation
optimizer = tf.train.AdamOptimizer(1e-5)
grads_and_vars = optimizer.compute_gradients(total_loss,
                                             var_list=gat_net_vars + vars_all_e + m2_enc_vars + m3_enc_vars + m4_enc_vars)
train_step = optimizer.apply_gradients(grads_and_vars, name='train_step')

# Computation of slicewise dice score
# gt_ = tf.placeholder(dtype=tf.float32, shape=(1024, 1024, 100), name="gt_")
# pred_ = tf.placeholder(dtype=tf.float32, shape=(1024, 1024, 100), name="pred_")
# dice_, seg_, thresh_ = dice_opt_gpu(gt_, pred_)

# # Accuracy scores
# epoch = 0
# tr_loss = []
# te_loss = []
# te_loss_tmp = np.zeros((int(n_te_patches / batch_size)))
# pr_te_tmp = np.zeros((int(n_te_patches / batch_size)))
# saver = tf.train.Saver(max_to_keep=100)
#
# saver2e = tf.train.Saver(var_list=m2_enc_vars)
# saver3e = tf.train.Saver(var_list=m3_enc_vars)
# saver4e = tf.train.Saver(var_list=m4_enc_vars)
#
# saver2d = tf.train.Saver(var_list=m2_dec_vars)
# saver3d = tf.train.Saver(var_list=m3_dec_vars)
# saver4d = tf.train.Saver(var_list=m4_dec_vars)

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()
#
# with tf.Session() as sess:  # laun ch the model in an interactive session
#     sess.run(tf.global_variables_initializer())  # create an operation to initialize the variables we created
#     sess.run(tf.local_variables_initializer())  # needed to compute tp, tn, fp, fn
#
#     saver2d.restore(sess, './trained_chkpts_dec/dnn_conc_2/my-model')
#     saver3d.restore(sess, './trained_chkpts_dec/dnn_conc_3/my-model')
#     saver4d.restore(sess, './trained_chkpts_dec/dnn_conc_4/my-model')
#
#     saver2e.restore(sess, './trained_chkpts_enc/dnn_conc_2/my-model')
#     saver3e.restore(sess, './trained_chkpts_enc/dnn_conc_3/my-model')
#     saver4e.restore(sess, './trained_chkpts_enc/dnn_conc_4/my-model')
#
#     for epoch_i in range(1, train_epochs):  # 10,000 iterations
#
#         ############### Training for one epoch ###############
#         ntrstep = 1
#         for train_batch in batch_generator(train_list, batch_size):  # (no. of traiing steps =  no. of raining patches)
#             ## Training step
#             t = time.time()
#             feed_dict = {
#                 x_holo: train_batch.get_holos_in_batch()[:, ::step, ::step, :],
#                 x: train_batch.get_original_images_in_batch(),
#                 y_: train_batch.get_ground_truths_in_batch()
#             }
#             train_step.run(feed_dict)  # we run train_step feeding in the batches data to replace the placeholders
#             loss_ce, loss_wt = sess.run((cross_entropy, weight_decay), feed_dict)
#             tr_loss.append(loss_ce)
#             ntrstep = ntrstep + 1
#
#             # Printing diagnostics
#             elapsed = time.time() - t
#             print('[epoch: %d] [step: %d] [iter_time: %f] [loss_ce: %1.10f] [loss_wt: %1.10f]' % (
#             epoch_i, ntrstep, elapsed, loss_ce, loss_wt))
#
#             ############### Testing after one epoch ###############
#             if (ntrstep % 10000) == 0:
#                 # Test on test-data
#                 print('Testing...')
#                 for testmouse in test_list:
#                     y_te_cube_pred = np.empty(testmouse.shape, dtype='float32')
#
#                     for test_batch in batch_generator([testmouse], batch_size):
#                         feed_dict = {
#                             x_holo: test_batch.get_holos_in_batch()[:, ::step, ::step, :],
#                             x: test_batch.get_original_images_in_batch(),
#                             y_: test_batch.get_ground_truths_in_batch(),
#                         }
#
#                         y_pred_p, loss_p = sess.run((y_pred, cross_entropy), feed_dict)
#                         print('[Testing loss: %f]' % loss_p)
#
#                         # put data into full FoV volume for testmouse
#                         for patch_i in range(batch_size):
#                             y_te_cube_pred[test_batch.get_index_list()[patch_i]] = y_pred_p[patch_i, ..., 0]
#
#                     te_loss.append(loss_p)
#                     x_te_cube = testmouse._o_images[0]
#                     y_te_cube = testmouse._g_images[0]
#
#                     # Make folders to save diagnostics
#                     test_result_dir = 'test_result/epoch%d_step%d/obj%d' % (epoch_i, ntrstep, testmouse.mousenum)
#                     if not os.path.exists(test_result_dir):
#                         os.makedirs(test_result_dir)
#
#                     # save DNN prediction
#                     save_3d_tiff(test_result_dir, {'predicted': y_te_cube_pred})
#
#                     # save slicewise dice index
#                     y_te_cube_pred_slicewise_01 = slicewise_01(y_te_cube_pred)
#                     feed_dict = {
#                         gt_: y_te_cube,
#                         pred_: y_te_cube_pred_slicewise_01,
#                     }
#                     slicewise_dice, seg_opt_dice = sess.run((dice_, seg_), feed_dict)
#                     seg_opt_dice = np.swapaxes(seg_opt_dice, 0, 2)
#                     seg_opt_dice = np.swapaxes(seg_opt_dice, 0, 1)
#                     for jj in range(100):
#                         slice_ = seg_opt_dice[:, :, jj]
#                         slice_gt = y_te_cube[:, :, jj]
#                         if np.all((slice_gt == 0)):
#                             seg_opt_dice[:, :, jj] = slice_ * 0
#                             slicewise_dice[jj] = 1
#
#                     save_3d_tiff(test_result_dir, {'seg_opt_dice': seg_opt_dice})
#                     slicewise_dice = np.flip(slicewise_dice)
#                     plt.plot(slicewise_dice)
#                     plt.ylim(0, 1)
#                     plt.xlabel('Slice number')
#                     plt.ylabel('Dice coefficient index')
#                     fn = '%s/slicewise_dice.png' % (test_result_dir)
#                     plt.savefig(fn)
#                     plt.close()
#
#                 # Save Loss value
#                 losses_fn = 'test_result/loss_tr_te'
#                 np.savez(losses_fn, np.asarray(tr_loss, dtype=np.float32), np.asarray(te_loss, dtype=np.float32))
#
#         # Save model
#         saver.save(sess, "./chkpt/my-model", global_step=epoch_i)
#
#
