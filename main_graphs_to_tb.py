import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers
from tensorflow_probability import positive_semidefinite_kernels as tfkern

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import sys

# import imports as im
import gp_functions as gpf
# import data_generation as dg
# import plots as plts
# import data_readers as drs
# import main_GP_fit
# import placement_algorithm2 as alg2
# import normalize_delete as nm

import main

def tb_graph_GP(LOGDIR='./tb_graph_GP/'):

    CSV = 'roomselection_1800.csv'
    dim1 = 'Points:0'
    dim2 = 'Points:1'
    dim3 = 'Points:2'
    dim4 = 'Temperature'
    dim5 = 'Pressure'
    dim6 = 'Tracer'
    FEATURES = [dim1, dim2, dim3, dim4, dim5, dim6]
    df = pd.read_csv(CSV, encoding='utf-8', engine='c')

    # Initializations
    AMPLITUDE_INIT = np.array([.1, .1]) # [0.1, 0.1]
    LENGTHSCALE_INIT = np.array([.001, .001]) # [0.1, 0.1]
    K_SENSORS = 1
    SPATIAL_COVER = 2
    SPATIAL_COVER_PRESSURE_TEMP = 2

    # Hyperparameters
    ENCODED_SIZE = 1
    GP_INPUT_IDX_DIM = ENCODED_SIZE
    SELECT_ROW_NUM = 8000#8000
    INIT_OBSNOISEVAR_ = 0.001
    INIT_OBSNOISEVAR = 1e-6
    LEARNING_RATE = .1 #.01
    NUM_ITERS = 10000  # 1000 optimize log-likelihood
    PRED_FRACTION = 50  # 50
    NUM_SAMPLES = 8 # 50
    LINSPACE_GP_INDEX_SAMPLE = 300 # plot GP fragmentation
    XEDGES = 60  # plot ampl and lengthscale optimization
    YEDGES = 60
    ROW_REDUCED = 100  # select fraction of encoded- and tracer row
    ROWS_FOR_EACH_COORD = 100


    # DATA ###############################################################

    # vae = tf.saved_model.load(SAVED_MODEL_DIR)
    col_len = len(FEATURES)
    sample_len = SELECT_ROW_NUM // 2
    values, normal, mean_var, stdev_var = gpf.graph_normalization_factors_from_training_data(sample_len, col_len)
    xyztp_norm = tf.slice(normal, begin=[0, 0], size=[25, 5], name="xyztp_norm")
    t_norm_ = tf.slice(normal, begin=[0, 5], size=[25, 1], name="t_norm_")
    t_norm = tf.reshape(t_norm_, shape=[-1], name="t_norm")
    # VAE
    vae, prior, encoder, decoder = gpf.create_model(ENCODED_SIZE, 5)
    vae.summary()
    #TODO what are these checkpoints
    checkpoint = tf.train.Checkpoint(x=vae)
    # checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))

    # Encoded data
    e_xyztp = encoder(xyztp_norm)
    assert isinstance(e_xyztp, tfd.Distribution)
    e_xyztp_s = e_xyztp.sample()
    stack2 = tf.stack([e_xyztp_s, t_norm_])

    et_Var = tf.cast(stack2, dtype=tf.float64)
    w_pred_linsp_Var = tf.linspace(tf.reduce_min(e_xyztp_s), tf.reduce_max(e_xyztp_s), LINSPACE_GP_INDEX_SAMPLE)
    w_pred_linsp_Var = tf.reshape(w_pred_linsp_Var,[-1,GP_INPUT_IDX_DIM], name="reshape_")

    tf.logging.warn('warn')
    tf.logging.error('error')
    tf.logging.fatal('fatal')
    # Sample decoder
    z = prior.sample(SELECT_ROW_NUM)
    d_xyztp = decoder(z)
    assert isinstance(d_xyztp, tfd.Distribution)
    d_xyztp_s = d_xyztp.sample()


    amp, amp_assign, amp_p, lensc, lensc_assign, lensc_p, log_likelihood, samples_1d, train_op, obs_noise_var = \
        main.graph_GP(  et_Var,
                        t_norm_,
                        w_pred_linsp_Var,
                        e_xyztp_s
                     )

    summ = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(LOGDIR, sess.graph)

        sess.run(tf.global_variables_initializer())
        writer.flush()


def tb_graph_cov(LOGDIR='./tb_graph_cov/'):

    class encoder():

        def __init__(self, x):
            self.x = x

        def sample(self):
            # return tf.slice(self.x, begin=[0, 0], size=[self.x.shape[0], 1], name="encoder_slicer")
            x_ = main.tf_print2(self.x, [self.x], 'x=\n')
            x_ = self.x
            b2 = tf.cast(self.x.shape[1]-1, dtype=tf.int32)
            slice = tf.slice(x_, begin=[0, b2], size=[self.x.shape[0], 1], name="encoder_slicer")
            # r = slice + tf.random.normal(shape=slice.shape, stddev=0.3)
            # return r
            slice_ = main.tf_print2(slice, [slice], 'slice=\n')
            return slice

    cov_vv, last_cov_vv, while_i0_idx, while_i0_end = main.graph_cov(2, 2, encoder)  # depends on encoder

    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        [w0, w1, e] = sess.run([while_i0_idx, while_i0_end, cov_vv])  #, feed_dict={values: test_dataset})
        # elast = sess.run(last_cov_v)  #v, feed_dict={values: train_dataset})

        print("cov_vv, after running while_op")
        print(w0, w1, e)
        # print("last_cov_vv_ij")
        # print(elast)

if __name__ == '__main__':
    # tb_graph_GP()
    tb_graph_cov()
    pass
