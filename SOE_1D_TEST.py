from SOE_NET_1D import SOE_Net
import configure as cfg
import init_SOE_NET as init_net
import numpy as np
import tensorflow as tf
from SOE_NET_1D import conv1d
from util import highestPowerof2
import audio2numpy as a2n
import librosa
import matplotlib.pyplot as plt       
# from SOE_NET_1D import SC1D, TPR, FWR, DivNorm1d, GSP
import os
from util import bar_plot
import pickle


tf.compat.v1.enable_eager_execution()
gau_filter = init_net.initSeparableFilters('basis', cfg.FILTER_TAPS, filter_type="G3")

PATH_AUDIO = "/home/eftekhar/Documents/Databases/AVDT_backup/Audios"

classes = []
outputs = []
for d in os.listdir(PATH_AUDIO):
    classes.append(d)
    audios = os.listdir(os.path.join(PATH_AUDIO, d))
#     m = min(100, len(audios))
    for aud in os.listdir(os.path.join(PATH_AUDIO, d))[:100]:
        print(d)
        audio_path = os.path.join(PATH_AUDIO, d, aud)
        # x,sr=a2n.audio_from_file(audio_path)
        x, s = librosa.load(audio_path, sr=cfg.ORIGINAL_SAMPLING_RATE)
        # x = x / np.amax(np.abs(x))
        # x = ((x[:,0] + x[:,1])/2)
        n = highestPowerof2(x.shape[0])
        x = x[x.shape[0] // 2 - n // 2 : x.shape[0] // 2 + n // 2]
        vol = tf.constant(x[None,:,None], dtype=tf.float32)
        output = SOE_Net(vol, gau_filter)
        outputs.append(output.numpy())

np.save("results_3layer_numscale20_downfac_root_2_no_norm1_librosa_44100_two_path_kaiser_fast_20_classes_100videos.npy", np.concatenate(outputs, axis=0))

# with open("20classes", "wb") as f:   #Pickling
#     pickle.dump(classes, f)
