import time
import tensorflow as tf
import numpy as np
import sys
import input_data, os
import configure as cfg
import init_SOE_NET as init_net
import SOE_Net_model_full as model


################################################################################################
#""" INITIALIZE SOE-NET PARAMETERS """
################################################################################################
tf.reset_default_graph()

orientations = np.array(init_net.initOrientations (cfg.ORIENTATIONS,cfg.SPEEDS,cfg.NUM_DIRECTIONS), dtype=np.float32)
                
#print("[INFO] INIT", orientations,"STANDARD ORIENTATIONS...")
basis = init_net.initSeparableFilters('basis', cfg.FILTER_TAPS, filter_type="G3")

basis_2d = init_net.initSeparableFilters_SO('basis2d', cfg.FILTER_TAPS, filter_type="G2")
#print("[INFO] INIT", basis,"3D SEPARABLE FILTERS...")
#print("[INFO] INIT", basis_2d,"2D SEPARABLE FILTERS...")

biases_soe = init_net.initBiases('bias', 0)
              
#print("[INFO] INIT", biases_soe,"SEPARABLE FILTERS...")

################################################################################################
#""" READ THE INPUT DATA """
################################################################################################

# TEST DATA
tiny_batch_size = 1 # to extract features for one video at a time for this example code purposes
vid_path = "/home/eftekhar/Documents/Databases/AVDT_backup/Frames/clips.txt"
print("[INFO] READING IN TEST DATA FROM :", vid_path)
test_clips, test_start_indices, test_labels = input_data.load_clips_labels(vid_path)
num_test_clips = len(test_clips)
print("[INFO] TOTAL NUMBER OF TESTING VIDEO CLIPS :", num_test_clips)
#raw_input('Press enter to continue ... ')
TEST_ITERS = int(float(num_test_clips)/float(tiny_batch_size))
print("[INFO] NUMPBER OF ITERATIONS TO GO THROUGH TEST SET IS: ", TEST_ITERS)

################################################################################################
#""" PREPARE DATA FOR TENSORFLOW """
################################################################################################
if cfg.CROP:
    input_shape   = [tiny_batch_size, cfg.TIME_S, cfg.IMG_S, cfg.IMG_S, 1]
else:
    input_shape   = [tiny_batch_size, cfg.TIME_S, cfg.IMG_RAW_H, cfg.IMG_RAW_W, 1]
 
batch_videos_ph  = tf.placeholder(tf.float32, shape=input_shape, name="batch_videos")
print(batch_videos_ph)

################################################################################################   
#""" BUILD THE TENSORFLOW GRAPH """
################################################################################################
# example 4: Extract SOE_Net features
soenet = model.SOE_Net(batch_videos_ph, basis, orientations, biases_soe)

# example 2: Extract MSOE features
msoe = model.get_MSOE(batch_videos_ph, basis, orientations, biases_soe)


init = tf.global_variables_initializer()
sess = tf.Session()
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# initialize the non-trained variables
sess.run(init)

test_indices = np.arange(num_test_clips)
test_start_idx = 0

SOE_NET_FEAT = []
tic = time.time()
excluded_vid = []
for iter in range(TEST_ITERS):
    """ LOAD VIDEOS ONE BATCH AT A TIME """
    test_batch_clips, test_batch_indices, test_batch_labels, test_stop_idx = input_data.select_batch(test_clips,test_start_indices, test_labels, test_start_idx, test_indices, tiny_batch_size)
    print(test_batch_clips)
    print(test_batch_indices)
    try:
        batch_videos=input_data.load_frames(test_batch_clips,test_batch_indices,crop=False)
    except:
        excluded_vid.append(test_batch_clips[0])
        test_start_idx = test_stop_idx
        continue 
    print("size of clips in batch is : ", np.shape(batch_videos))
    print("stop_idx is : ", test_stop_idx)
    test_start_idx = test_stop_idx
    print("This is iteration number:{}".format(iter))
    #tic = time.time()
    soenet_feat = sess.run(soenet, feed_dict={batch_videos_ph: batch_videos}) 
    #print("time to run through one mini-batch is: ", time.time()-tic)
    """ EXAMPLE: SAVE THE RESULTS FOR FURTHER USE (IN ANOTHER APPLICATION FOR EXAMPLE)"""
    for b in range(soenet_feat.shape[0]):
        # TODO: INSTEAD OF APPENDING ACCUMULATE RESULTS OBTAINED BY APPLYING GSP
        SOE_NET_FEAT.append(soenet_feat)
    
SOE_NET_FEAT = np.concatenate(SOE_NET_FEAT, axis=0)   
print(SOE_NET_FEAT.shape)
# save results to npy file

feat_path = "./video_features/" + "features_test_3layer_highresolution.npy"
np.save(feat_path, SOE_NET_FEAT)
print("time to run through one mini-batch is: ", time.time()-tic)
print(excluded_vid)
print('Done')