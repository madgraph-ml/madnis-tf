import tensorflow as tf
import numpy as np
madgraph = tf.load_op_library('./madevent_tf.so')
channel = 0
tf.debugging.set_log_device_placement(True)


# @tf.function
def gen(nbatch, chan):
    nrans = 20
    rans = tf.random.uniform(shape=[nbatch, nrans], dtype=tf.float64)
    # shape = tf.constant([nbatch, nrans], dtype=tf.int32)
    # rans = tf.tile(rans, shape)
    chans = tf.random.uniform(shape=[nbatch], dtype=tf.int32, minval=chan, maxval=chan+1)
    wgt = madgraph.call_madgraph(rans, chans)
    wgt2 = wgt**2
    mean = tf.reduce_mean(wgt)
    error = tf.sqrt((tf.reduce_mean(wgt2) - mean**2)/(nbatch-1))
    return mean, error, chans


mean1, err1, chans1 = gen(1000000, 0)
mean2, err2, chans2 = gen(1000000, 1)

print(f"counts = {tf.math.bincount(chans1)}, {tf.math.bincount(chans2)}")
print(f"chans = {chans1[0]}, {chans2[0]}")
print(f"r1: {mean1} +/- {err1}, r2: {mean2} +/- {err2}")
# channel = 1
# rans = tf.random.uniform(shape=[nbatch, nrans], dtype=tf.float64)
# chans = channel*tf.ones(shape=[nbatch], dtype=tf.int32)
# wgt = madgraph.call_madgraph(rans, chans)
# wgt2 = wgt**2
# mean = tf.reduce_mean(wgt)
# error = tf.sqrt((tf.reduce_mean(wgt2) - mean**2)/(nbatch-1))
# print(mean, error)
# rans = np.array([[0.94818978187368286, 0.64758579352377721, 0.53540670827378767, 0.41862633529830906, 0.54952461203150937, 0.77051114067318749, 0.98681435682750385, 5.7246425377724064E-002, 0.46187131034377016, 0.67014653812676861, 0.75167459577737650, 0.87655613684429112, 7.7882039136799364E-002, 0.94665210294432889, 9.7242433603321743E-003, 5.9229003959395010E-002, 8.6394335603854566E-002, 0.72961905463333765, 1.1648770160594069E-002,  0.13726674778251957]])
# channel = np.array([0])
# print(rans, channel)
# print(madgraph.call_madgraph(rans, channel))
