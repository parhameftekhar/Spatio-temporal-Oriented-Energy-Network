import tensorflow as tf
import numpy as np
import configure as cfg
import librosa
import math

# def SC1D(name, in_data, basis, down_fac=math.sqrt(2), num_scale=20, is_first_layer=True):
#     if is_first_layer:
#         all_scales = []
    
#         all_scales.append(conv1d('G_a_scale{}'.format(0), in_data, basis[0,:]))
    
#         in_data_np = np.squeeze(in_data.numpy())
#         n = in_data_np.shape[0]
#         for i in range(1, num_scale + 1):
#             resample_rate = cfg.ORIGINAL_SAMPLING_RATE // (down_fac ** i)
#             x = librosa.resample(in_data_np, orig_sr = cfg.ORIGINAL_SAMPLING_RATE, target_sr = resample_rate)
#             x = tf.constant(x[None,:,None], dtype=tf.float32)
#             x = conv1d('G_a_scale{}'.format(i), x, basis[0,:])
#             x = np.squeeze(x.numpy())       
#             x = librosa.resample(x, orig_sr = resample_rate, target_sr = cfg.ORIGINAL_SAMPLING_RATE) 
#             x = tf.constant(x[None,:,None], dtype=tf.float32)
#             # x = x / cfg.MULTI_SCALING_NORM_FACTORS[i]
#             x = x[:, x.shape[1] // 2 - n // 2 : x.shape[1] // 2 + n // 2, :]
#             all_scales.append(x)
        
#         return tf.concat(axis=-1, values=all_scales, name='concatScales')

#     else:
#         x = conv1d('G_a', in_data, basis[0,:])
#         return x

        
def down_sampling_multi_channels(x, resample_rate):
    num_channels = x.shape[2]
    output = []
    for ch in range(num_channels):
        output.append(librosa.resample(x[0, :, ch], orig_sr = cfg.ORIGINAL_SAMPLING_RATE, target_sr = resample_rate, res_type=cfg.RESAMPLE_TYPE)[None, :, None])
    return np.concatenate(output, axis=2)

def up_sampling_multi_channels(x, resample_rate):
    num_channels = x.shape[2]
    output = []
    for ch in range(num_channels):
        output.append(librosa.resample(x[0, :, ch], orig_sr = resample_rate, target_sr = cfg.ORIGINAL_SAMPLING_RATE, res_type=cfg.RESAMPLE_TYPE)[None, :, None])
    return np.concatenate(output, axis=2)

def SC1D_MULTI_CH(name, in_data, basis, down_fac, num_scale):
    all_scales = []
    
    all_scales.append(conv1d('G_a_scale{}'.format(0), in_data, basis[0,:]))
    
    in_data_np = in_data.numpy()
    n = in_data_np.shape[1]
    for i in range(1, num_scale + 1):
        resample_rate = cfg.ORIGINAL_SAMPLING_RATE // (down_fac ** i)
        x = down_sampling_multi_channels(in_data_np, resample_rate)
        x = tf.constant(x, dtype=tf.float32)
        x = conv1d('G_a_scale{}'.format(i), x, basis[0,:])
        x = x.numpy()
        x = up_sampling_multi_channels(x, resample_rate)
        x = tf.constant(x, dtype=tf.float32)
        x = x[:, x.shape[1] // 2 - n // 2 : x.shape[1] // 2 + n // 2, :]
        all_scales.append(x)
        
    return tf.concat(axis=-1, values=all_scales, name='concatScales')



# # Num scale is the number of times downsampling is performed (excluding sigma0).
# def SC1D(name, in_data, basis, num_scale):

#     original_temporal_res = in_data.shape[1]
#     # Binomial filter to be used for downsampling
#     binomial_filter = tf.constant(np.array([1, 4, 6, 4, 1])/16, dtype=tf.float32)

#     all_scales = []

#     # downsampling untill the level of desire
#     all_scales.append(in_data)
#     for scale in range(1, num_scale + 1):
#         in_data = conv1d('downsampling_smoothing_scale{}'.format(scale), in_data, binomial_filter)
#         mask = tf.not_equal(tf.range(tf.shape(in_data)[1]) % cfg.DOWN_SAMPLE_FACTOR, 0)
#         in_data = tf.boolean_mask(in_data, mask, axis=1)
#         all_scales.append(in_data)

#     # applying filter of interest and upscaling
#     for i in range(num_scale + 1):
#         vol = conv1d('G_a_scale{}'.format(i), all_scales[i], basis[0,:])
#         while vol.shape.as_list()[1] != original_temporal_res:
#             vol = tf.keras.layers.UpSampling1D(size=cfg.DOWN_SAMPLE_FACTOR)(vol)
#             mask = tf.constant([0,1] * (vol.shape.as_list()[1] // 2), dtype=tf.float32)[None, :, None]
#             mask = tf.repeat(mask, repeats=[vol.shape[-1]], axis=-1)
#             vol = vol * mask
#             vol = cfg.AUDIO_INTERPOLATION_FACTOR * conv1d('downsampling_smoothing_scale{}'.format(i), vol, binomial_filter)

#         # vol = vol / cfg.MULTI_SCALING_NORM_FACTORS[i] 
#         all_scales[i] = vol
    

#     vols = tf.concat(axis=-1, values=all_scales, name='concatScales')            
#     return vols



def conv1d(name, in_data, t):

    ft = t[:,np.newaxis,np.newaxis]
    SepConv1d = []
    for ch in range(in_data.shape[2]):
        vol = in_data[:,:,ch]
        TmpConv1d = tf.nn.conv1d(vol[:,:,None], ft, stride=[1, 1, 1], padding='SAME')
        SepConv1d += [TmpConv1d]
    SepConv1d = tf.concat(axis=2, values=SepConv1d, name='concatSOE')            
    
    return SepConv1d

def TPR(name, C):
    C_plus_minus = []
    
    C_plus_minus += [tf.square(tf.clip_by_value(C, 0, tf.reduce_max(C)), name="branch_pos")]
    C_plus_minus += [tf.square(tf.clip_by_value(C, tf.reduce_min(C), 0), name="branch_neg")]
    C_plus_minus = tf.concat(axis=2, values=C_plus_minus, name='ConcatRec')
    return C_plus_minus

def FWR(name, C):
    C_plus = tf.square(C, name)
    return C_plus


def DivNorm1d(name, in_data, eps):
    if eps=="std_based":
        epsilon = tf.math.reduce_std(in_data, keepdims=True)

    sumE = tf.reduce_sum(in_data, axis=2, keepdims=True)
    normalizer = epsilon + sumE
    norm = tf.divide(in_data, normalizer)
    
    return norm


def AAP(wc, a):
    eta = 2*wc
    wl = eta*a
    sigma = np.divide(3,wl)
    ws = 2.5*wl
    T = np.ceil(np.divide((2*np.pi),ws))
    
    return sigma, wl, T


def STPP(numL):
    sig = 1
    a = 0.35
    wc = 3* np.divide(np.sqrt(3), sig)
    sigma = np.zeros((numL,),dtype=np.float32)
    wl = np.zeros((numL,),dtype=np.float32)
    T = np.zeros((numL,), dtype=np.int)
    for L in range(numL):
        sigma[L], wl[L], T[L] = AAP(wc, a)
        wc = wl[L]   
    return T, sigma


def STPS(numL, L):
    strides, _ = STPP(numL)
    return strides[L]


def Gaussian3d(sigma, taps):
    flen = (cfg.FILTER_TAPS*2)+1
    shape = (1, flen)
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[h< np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh!=0:
        h /=sumh
    g_filter = np.reshape(h, (flen,))
    
    return np.float32(g_filter)

def STPF(name, numL, L):
    _, sigma = STPP(numL)
    g_filter = Gaussian3d(sigma[L], cfg.FILTER_TAPS)
    
    return g_filter


def SP3D(name, in_data, numL, L):
    
    T = STPS(numL, L)
    g_filter = STPF(name, numL, L)
    st_pool = conv1d('STPool', in_data, g_filter)
    st_pool = st_pool[:,::T,:]
    return st_pool

def GSP(name, in_data):
    flen = (cfg.FILTER_TAPS*2)+1
    if in_data.shape[1] >= flen:
        g_pool = tf.reduce_sum(in_data[:,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:], axis=[1])
    else:
        g_pool = tf.reduce_sum(in_data[:,:,:], axis=[1])
    return g_pool


################################
### Modified Version ###########
################################
# def power(v,p):
#     vp = tf.pow(v, p, name='None')
#     return vp

# def S3DG3(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, ox,oy,ot):
#     SteeredConv = power(ox,3)*G_a + 3*power(ox,2)*oy*G_b + 3*ox*power(oy,2)*G_c + power(oy,3)*G_d + 3*power(ox,2)*ot*G_e + 6*ox*oy*ot*G_f + 3*power(oy,2)*ot*G_g + 3*ox*power(ot,2)*G_h + 3*oy*power(ot,2)*G_i + power(ot,3)*G_j   
#     return SteeredConv


def CP3D(name, in_data, num_scales, rectification_type):
    num_ch = in_data.get_shape().as_list()[-1]
    output = []
    if rectification_type == 'one_path':
        for s in range(num_scales+1):
            sliced = in_data[:,:,s:num_ch:num_scales+1]
            output.append(tf.reduce_mean(sliced, axis=-1, keep_dims=True))
            
        return tf.concat(output, axis = -1)
    else:
        num_ch = in_data.get_shape().as_list()[-1]
        l = []
        for i in range(cfg.NUM_SCALES + 1):
            x = np.zeros(((cfg.NUM_SCALES + 1) ** 2, 1), dtype=np.float32)
            x[i::cfg.NUM_SCALES + 1] = 1.0 / (cfg.NUM_SCALES + 1)
            l.append(x)
        chunk = np.concatenate(l, axis=-1)
        f = np.zeros((num_ch, num_ch // (cfg.NUM_SCALES + 1)), dtype=np.float32)
        for i in range(num_ch // ((cfg.NUM_SCALES + 1) ** 2)):
            f[i * ((cfg.NUM_SCALES + 1) ** 2): (i + 1) * ((cfg.NUM_SCALES + 1) ** 2), i * (cfg.NUM_SCALES + 1): (i + 1) * (cfg.NUM_SCALES + 1)] = chunk

        return tf.nn.conv1d(in_data, tf.constant(f[None,:,:], dtype=tf.float32), stride=[1, 1, 1], padding='VALID')

    

# def CPP(name, num_dir, L,rec_style):
#     if rec_style is 'two_path':
#         in_idx = np.arange(num_dir*num_dir*np.power(2,L))
#         #mods = np.mod(in_idx,num_dir)
#         divs = np.divide(in_idx, num_dir)
#         cc_filter = []   
#         for i in range(num_dir*np.power(2,L)):
#             x = np.zeros((num_dir*num_dir*np.power(2,L),))
#             out_idx = np.nonzero(divs==i)            
#             x[out_idx] = 1.
#             cc_filter.append(x)
#     else:
#         in_idx = np.arange(num_dir*num_dir)
#         #mods = np.mod(in_idx,num_dir)
#         divs = np.divide(in_idx, num_dir)
#         cc_filter = []   
#         for i in range(num_dir):
#             x = np.zeros((num_dir*num_dir,))
#             out_idx = np.nonzero(divs==i)            
#             x[out_idx] = 1.
#             cc_filter.append(x)

#     cc_filter = np.divide(np.array(cc_filter), num_dir)
#     cc_filter = np.float32(np.transpose(cc_filter))
    
#     return cc_filter

# def CP3D(name, in_data, L):
        
#     cc_filter = CPP(name, cfg.NUM_DIRECTIONS,L+1, cfg.REC_STYLE)
    
#     if cfg.REC_STYLE is 'two_path':
#         shape_L1 = 2*cfg.NUM_DIRECTIONS
#     else:
#         shape_L1 = cfg.NUM_DIRECTIONS
#     if in_data.shape[-1] == shape_L1:
#         cc_pool = in_data
        
#     else:
#         cc = cc_filter[None, None, None, :,:]
#         cc_pool = tf.nn.conv3d(in_data, cc, strides=[1, 1, 1, 1, 1], padding='VALID')
#     return cc_pool

# def GSP(name, in_data):
#     flen = (cfg.FILTER_TAPS*2)+1
#     if in_data.shape[1] >= flen:
#         g_pool = tf.reduce_sum(in_data[:,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:], axis=[1,2,3])
#     else:
#         g_pool = tf.reduce_sum(in_data[:,:,:,:,:], axis=[1,2,3])
#     return g_pool

def SOE_Net(video, basis):
    
    with tf.name_scope('Layer1'):
        conv1 = SC1D_MULTI_CH('conv1', video, basis, cfg.DOWN_SAMPLE_FACTOR, cfg.NUM_SCALES)
        if cfg.REC_STYLE == 'two_path':
            rec1 = TPR('Rec1', conv1)
        else:
            rec1 = FWR('Rec1', conv1)
            
        norm1 = DivNorm1d('norm1', rec1, cfg.EPSILON)
        # sp1 = SP3D('sp1', norm1, cfg.NUML, 0)
        # feat = GSP('GSP',norm1)

    with tf.name_scope('Layer2'):

        conv2 = SC1D_MULTI_CH('conv2', norm1, basis, cfg.DOWN_SAMPLE_FACTOR, cfg.NUM_SCALES)
        if cfg.REC_STYLE == 'two_path':
            rec2 = TPR('Rec2', conv2)
        else:
            rec2 = FWR('Rec2', conv2)
            
        norm2 = DivNorm1d('norm2', rec2, cfg.EPSILON)
        # feat = GSP('GSP',norm2)
        cp2 = CP3D('cp2', norm2, cfg.NUM_SCALES, cfg.REC_STYLE)

    with tf.name_scope('Layer3'):

        conv3 = SC1D_MULTI_CH('conv3', cp2, basis, cfg.DOWN_SAMPLE_FACTOR, cfg.NUM_SCALES)
        if cfg.REC_STYLE == 'two_path':
            rec3 = TPR('Rec3', conv3)
        else:
            rec3 = FWR('Rec3', conv3)
            
        norm3 = DivNorm1d('norm3', rec3, cfg.EPSILON)
        feat = GSP('GSP',norm3)
    #     cp3 = CP3D('cp3', norm3, cfg.NUM_SCALES, cfg.REC_STYLE)

    # with tf.name_scope('Layer4'):

    #     conv4 = SC1D_MULTI_CH('conv4', cp3, basis, cfg.DOWN_SAMPLE_FACTOR, cfg.NUM_SCALES)
    #     if cfg.REC_STYLE == 'two_path':
    #         rec4 = TPR('Rec4', conv4)
    #     else:
    #         rec4 = FWR('Rec4', conv4)
            
    #     norm4 = DivNorm1d('norm4', rec4, cfg.EPSILON)

    #     feat = GSP('GSP',norm4)

    return feat


# def SOE_Net(video, basis):
    
#     with tf.name_scope('Layer1'):
#         conv1 = SC1D('conv1', video, basis)
#         if cfg.REC_STYLE == 'two_path':
#             rec1 = TPR('Rec1', conv1)
#         else:
#             rec1 = FWR('Rec1', conv1)
            
#         norm1 = DivNorm1d('norm1', rec1, cfg.EPSILON)
        
#         sp1 = SP3D('sp1', norm1, cfg.NUML, 0)
#         cp1 = CP3D('cp1', sp1, 0)
#         print(sp1, cp1)

#     with tf.name_scope('Layer2'):
#         conv2 = SC1D('conv2', cp1, basis)
#         if cfg.REC_STYLE == 'two_path':
#             rec2 = TPR('Rec2', conv2)
#         else:
#             rec2 = FWR('Rec2', conv2)
            
#         norm2 = DivNorm1d('norm2', rec2, cfg.EPSILON)
        
#         sp2 = SP3D('sp2', norm2, cfg.NUML, 1)
#         cp2 = CP3D('cp2', sp2, 1)
#         print(sp2, cp2)
#     with tf.name_scope('Layer3'):
#         conv3 = SC1D('conv3', cp2, basis)
#         if cfg.REC_STYLE == 'two_path':
#             rec3 = TPR('Rec3', conv3)
#         else:
#             rec3 = FWR('Rec3', conv3)
            
#         norm3 = DivNorm1d('norm3', rec3, cfg.EPSILON)
        
#         sp3 = SP3D('sp3', norm3, cfg.NUML, 2)
#         cp3 = CP3D('cp3', sp3, 2)  
#         print(sp3, cp3)
#     with tf.name_scope('Layer4'):
#         conv4 = SC1D('conv4', cp3, basis)
#         if cfg.REC_STYLE == 'two_path':
#             rec4 = TPR('Rec4', conv4)
#         else:
#             rec4 = FWR('Rec4', conv4)
            
#         norm4 = DivNorm1d('norm4', rec4, cfg.EPSILON)
#         feat = GSP('GSP',norm4)
        
#     return feat
