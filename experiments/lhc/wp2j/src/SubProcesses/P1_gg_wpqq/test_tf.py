import tensorflow as tf
import numpy as np

madgraph = tf.load_op_library('./madevent_tf.so')
nbatch = 10000
nrans = 20
channel = 0
# tf.debugging.set_log_device_placement(True)
rans = tf.random.uniform(shape=[nbatch, nrans], dtype=tf.float64)
chans = channel*tf.ones(shape=[nbatch], dtype=tf.int32)
mom, wgt = madgraph.call_madgraph(rans, chans, npart=5)
print(mom)
wgt2 = wgt**2
mean = tf.reduce_mean(wgt)
error = tf.sqrt((tf.reduce_mean(wgt2) - mean**2)/(nbatch-1))
print(mean.numpy(), error.numpy())
# rans = np.array([[0.94818978187368286, 0.64758579352377721, 0.53540670827378767, 0.41862633529830906, 0.54952461203150937, 0.77051114067318749, 0.98681435682750385, 5.7246425377724064E-002, 0.46187131034377016, 0.67014653812676861, 0.75167459577737650, 0.87655613684429112, 7.7882039136799364E-002, 0.94665210294432889, 9.7242433603321743E-003, 5.9229003959395010E-002, 8.6394335603854566E-002, 0.72961905463333765, 1.1648770160594069E-002,  0.13726674778251957]])
# channel = np.array([0])
# print(rans, channel)
# print(madgraph.call_madgraph(rans, channel))
