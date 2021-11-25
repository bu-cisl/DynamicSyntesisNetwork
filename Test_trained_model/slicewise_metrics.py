
import numpy as np
import tensorflow as tf


def slicewise_01(cube):
	for i in range(100):
		slice = cube[:, :, i]
		if slice.min() != slice.max():
			slice = (slice - slice.min()) / (slice.max() - slice.min())
		else:
			slice = slice * 0
		cube[:, :, i] = slice
	return cube


########################################################################################################################
# DICE - CPU
########################################################################################################################

# slicewise dice-coefficient
def dice(gt_cube, pred_cube):
	dice_all = []
	for j in range(100):
		gt = gt_cube[:, :, j]
		gt[gt>0] = 1
		seg = pred_cube[:, :, j]
		seg[seg > 0.5] = 1
		if seg.min() != seg.max():
			dice_now = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
		else:
			dice_now = 0.5
		dice_all.append(dice_now)
	return np.flip(dice_all,0)

# slicewise dice-coefficient 2
def dice_opt(gt_cube, pred_cube):
	dice_all = []
	# over each slice
	for j in range(100):
		gt = gt_cube[:, :, j]
		seg = pred_cube[:, :, j]

		# over each threshold
		dice_oneS_allT = []
		for t in range(0, 255):
			seg[seg > (t*(1/255))] = 1
			if seg.min() != seg.max():
				dice_now = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
			else:
				dice_now = 0.5
			dice_oneS_allT.append(dice_now)

		# find max dice
		dice_all.append(max(dice_oneS_allT))
	return np.flip(dice_all,0)


########################################################################################################################
# DICE - GPU
########################################################################################################################
# Thresh 0.5
def cond_fwd(ind_z, gt_cube, pred_cube, opt_thresh, dice):
	return ind_z < 100

def body_fwd(ind_z, gt_cube, pred_cube, opt_thresh, dice):
	gt =  gt_cube[:, :,ind_z]
	pred = pred_cube[:, :, ind_z]
	seg = tf.cast(tf.greater_equal(pred, 0.5), tf.float32)
	opt_thresh = opt_thresh.write(ind_z, 0.5)
	dice_now = tf.reduce_sum(seg * gt) * 2.0 / (tf.reduce_sum(seg) + tf.reduce_sum(gt))
	dice = dice.write(ind_z, dice_now)
	return ind_z + 1, gt_cube, pred_cube, opt_thresh, dice

def dice_gpu(gt_cube, pred_cube):
	numslice = 100
	ind_z = tf.constant(0)
	opt_thresh = tf.TensorArray(dtype=tf.float32, size=numslice, dynamic_size=False, clear_after_read=False, infer_shape=True)
	dice = tf.TensorArray(dtype=tf.float32, size=numslice, dynamic_size=False, clear_after_read=False, infer_shape=True)
	list_vals = tf.while_loop(cond_fwd, body_fwd, [ind_z, gt_cube, pred_cube, opt_thresh, dice])
	dice_ret_ = list_vals[4]
	dice_ret = dice_ret_.stack()
	return dice_ret

########################################################################################################################
# Optimal thresh
def cond_ov_thresh(ind_t, gt, pred, dice_all_t):
	return ind_t < 256

def body_ov_thresh(ind_t, gt, pred, dice_all_t):
	thresh = tf.cast(ind_t,tf.float32) * 1/256
	seg = tf.cast(tf.greater_equal(pred, thresh), tf.float32)
	dice_now = tf.reduce_sum(seg * gt) * 2.0 / (tf.reduce_sum(seg) + tf.reduce_sum(gt))
	dice_all_t = dice_all_t.write(ind_t, dice_now)
	return ind_t + 1, gt, pred, dice_all_t



def cond_ov_slice(ind_z, gt_cube, pred_cube, opt_thresh, dice_all_z, seg_cube):
	return ind_z < 100

def body_ov_slice(ind_z, gt_cube, pred_cube, opt_thresh, dice_all_z, seg_cube):
	gt =  gt_cube[:, :,ind_z]
	pred = pred_cube[:, :, ind_z]

	ind_t = tf.constant(0)
	dice_all_t = tf.TensorArray(dtype=tf.float32, size=256, dynamic_size=False, clear_after_read=False, infer_shape=True)
	list_vals = tf.while_loop(cond_ov_thresh, body_ov_thresh, [ind_t, gt, pred, dice_all_t])
	dice_all_t_ = list_vals[3]
	dice_all_t_array = dice_all_t_.stack()
	dice_best = tf.reduce_max(dice_all_t_array)
	thresh_best = tf.cast(tf.argmax(dice_all_t_array), tf.float32) * tf.constant(1/256, dtype=tf.float32)
	dice_all_z = dice_all_z.write(ind_z, dice_best)
	opt_thresh = opt_thresh.write(ind_z, thresh_best)
	seg_cube = seg_cube.write(ind_z, tf.cast(tf.greater_equal(pred, thresh_best), tf.uint8))
	return ind_z + 1, gt_cube, pred_cube, opt_thresh, dice_all_z, seg_cube

def dice_opt_gpu(gt_cube, pred_cube):
	numslice = 100
	ind_z = tf.constant(0)
	opt_thresh = tf.TensorArray(dtype=tf.float32, size=numslice, dynamic_size=False, clear_after_read=False, infer_shape=True)
	dice_all_z = tf.TensorArray(dtype=tf.float32, size=numslice, dynamic_size=False, clear_after_read=False, infer_shape=True)
	seg_cube = tf.TensorArray(dtype=tf.uint8, size=100, dynamic_size=False, clear_after_read=False,infer_shape=True)
	list_vals = tf.while_loop(cond_ov_slice, body_ov_slice, [ind_z, gt_cube, pred_cube, opt_thresh, dice_all_z, seg_cube])
	dice_ret_ = list_vals[4]
	thresh_ret_ = list_vals[3]
	seg_cube_ret_ = list_vals[5]
	dice_ret = dice_ret_.stack()
	thresh_ret = thresh_ret_.stack()
	seg_cube_ret = seg_cube_ret_.stack()
	return dice_ret, seg_cube_ret, thresh_ret
