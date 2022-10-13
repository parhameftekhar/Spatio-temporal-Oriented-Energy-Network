import configure as cfg
import init_SOE_NET as init_net
import numpy as np
import tensorflow as tf
from SOE_NET_1D import conv1d
from util import highestPowerof2
import audio2numpy as a2n
from SOE_NET_1D import SOE_Net
import matplotlib.pyplot as plt       



tf.compat.v1.enable_eager_execution()

gau_filter = init_net.initSeparableFilters('basis', cfg.FILTER_TAPS, filter_type="G3")

x,sr=a2n.audio_from_file("wavy_water_245.mp3")
x = x[:,0][:,None]
n = highestPowerof2(x.shape[0])
x = x[x.shape[0] // 2 - n // 2 : x.shape[0] // 2 + n // 2, :]
vol = tf.constant(x[None,:,:], dtype=tf.float32)
output = SOE_Net(vol, gau_filter)
plt.plot(output.numpy()[0,:])
plt.show()
