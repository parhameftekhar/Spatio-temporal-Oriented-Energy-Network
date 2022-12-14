"""
Prepare all global varibales and data paths.
"""
import os
import numpy as np
import math


DATASET     = 'Gamma_frm'
DIR_HOME    = '/media/rwlab'
DIR_DATA_TRAIN   = os.path.join(DIR_HOME,DATASET, 'TRAIN')
DIR_DATA_TEST    = os.path.join(DIR_HOME,DATASET, 'TEST')


TRAINED_MODELS = os.path.join(DIR_HOME,'SOE_Net/models')
MODEL_NAME = 'soe_net_model'
LOG_DIR = os.path.join(DIR_HOME,('SOE_Net/visual_logs/' + MODEL_NAME + '/train'))


PTH_LST = '/home/hadjisma/Dropbox/SOE_Net-tensorflow/data/'
PTH_TRAIN_LST  = PTH_LST + DATASET + '_TRAIN.txt'
PTH_TEST_LST  = PTH_LST + DATASET + '_TEST.txt'
MEANFILE = None

RES_SAVE_PATH = '/home/hadjisma/Dropbox/' + DATASET + '/Results/'

# IMAGE_FORMAT   = '{:06d}.jpg'
IMAGE_FORMAT = '{}.jpg'
IMG_RAW_H = 200 #128 #259
IMG_RAW_W = 200 #171 #327
IMG_S = 112
TIME_S = 42 
CHANNELS = 1
CROP = False #True
NUM_CLASSES = 18


NUM_GPUS = 4
BATCH_SIZE = 2 #64
TOTAL_BATCH_SIZE = BATCH_SIZE*NUM_GPUS
INIT_LEARNING_RATE = 0.0001
DECAY_LEARNING_RATE = 0.1
DECAY_STEP_FACTOR = 2
WEIGHT_INIT = 0.001
WEIGHT_DECAY = 0.0005 
BIAS_DECAY = 0.0005 #0.0005
EPOCHS = 1 #100
TRAIN_DROPOUT = 0.5 
TEST_DROPOUT = 1.0
MOMENTUM = 0.9
MOVING_AVERAGE_DECAY = 0.9999
SHUFFLE = False
USE_PRETRAINED = False
DISPLAY = 20
TEST_INTERVAL = 1000
TEST_ITER = 100
SNAPSHOT = 1000


EPS = 1.1920928955078125e-06
NUML = 5
SPEEDS = 1 # use speed =0.5 for slow speed MSOE features
NUM_DIRECTIONS = 10
ORIENTATIONS = "icosahedron"
FILTER_TAPS = 6
EPSILON = "std_based"
REC_STYLE = 'two_path'

# Audio related configs
DOWN_SAMPLE_FACTOR = math.sqrt(2)
AUDIO_INTERPOLATION_FACTOR = 1
MULTI_SCALING_NORM_FACTORS = np.squeeze(np.load("scales_librosa_25.npy")).tolist()
NUM_SCALES = 20
ORIGINAL_SAMPLING_RATE = 44100
RESAMPLE_TYPE = "kaiser_fast"