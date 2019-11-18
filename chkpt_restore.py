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
# import data_generation as dg
# import data_readers as drs
# import main_GP_fit
import placement_algorithm2 as alg2
# import normalize_delete as nm
# import snippets as snps
import plots as plts
import gp_functions as gpf
import snippets_a2 as snps2
import main
from tensorflow.python.tools import inspect_checkpoint as chkp


def get_variables_for_tfk_sequential(seq, strip_scope_name=''):
    """
    :param seq: tf keras Sequential instance (has .trainable_variables prop)
    :param strip_scope_name: the alien scope (coming from .. with tf.name_scope("data_preprocessing") ..)
    :return: dictionary containing {'var-name': instance_variable} mapping
    """
    vars = seq.trainable_variables  

    saverdef_dict = {}
    N = len(strip_scope_name)
    do_strip_scope_name = True if 0<N else False
    for v in vars:
        name = v.name

        if do_strip_scope_name:
            if name.startswith(strip_scope_name):
                name = name[N:]

        if 0<name.index(':'):
            name = name[:name.index(':')]

        saverdef_dict[name] = v

    return saverdef_dict

def test_restore_chkpt_in_graph():

    ######################################################################
    # CONSTANTS ##########################################################

    # Switches #
    PYTHON_COV_CALC = False

    # Files #
    SAVED_MODEL_PATH = "./vae_training/model.ckpt"
    SAVED_MODEL_DIR = "./vae_model/"

    CHECKPOINT_PATH = "./vae_training/model.ckpt"
    print('checkpoint: ', CHECKPOINT_PATH, '\n')
    chkp.print_tensors_in_checkpoint_file(CHECKPOINT_PATH, tensor_name='', all_tensors=True)
    print('\n')

    LOGDIR = "./log_dir_/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')+"_gp"
    LOGCHECKPT = "model.ckpt"
    NORMCHECKPT = 'normalize.ckpt'
    CSV = 'roomselection_1800.csv'
    dim1 = 'Points:0'
    dim2 = 'Points:1'
    dim3 = 'Points:2'
    dim4 = 'Temperature'
    dim5 = 'Pressure'
    dim6 = 'Tracer'
    FEATURES = [dim1, dim2, dim3, dim4, dim5, dim6]
    df = pd.read_csv(CSV, encoding='utf-8', engine='c')

    # Initializations #
    AMPLITUDE_INIT = np.array([.1, .1]) # [0.1, 0.1]
    LENGTHSCALE_INIT = np.array([.001, .001]) # [0.1, 0.1]
    K_SENSORS = 7
    SPATIAL_COVER = 7
    SPATIAL_COVER_PRESSURE_TEMP = 7

    # Hyperparameters #
    ENCODED_SIZE = 1
    SELECT_ROW_NUM = 8000#8000
    INIT_OBSNOISEVAR_ = 0.001
    INIT_OBSNOISEVAR = 1e-6
    LEARNING_RATE = .1 #.01
    NUM_ITERS = 10000  # 1000 optimize log-likelihood
    PRED_FRACTION = 50  # 50
    NUM_SAMPLES = 8 # 50
    LINSPACE_GP_INDEX_SAMPLE = 300 # plot GP fragmentation
    XEDGES = 160  # plot ampl and lengthscale optimization
    YEDGES = 160
    ROW_REDUCED = 100  # select fraction of encoded- and tracer row
    ROWS_FOR_EACH_COORD = 100


    ######################################################################
    # DATA ###############################################################

    # vae = tf.saved_model.load(SAVED_MODEL_DIR)
    # VAE
    # vae, prior, encoder, decoder = gpf.create_model(ENCODED_SIZE, 5)
    # encoder_saver = tf.train.Saver()  # when parsing the default graph, it has only the VAE_ yet

    with tf.name_scope("data_preprocessing"):
        # VAE : at this point we have mean and stdev as extra
        vae, prior, encoder, decoder = gpf.create_model(ENCODED_SIZE, 5)
        saverdef_dict = get_variables_for_tfk_sequential(encoder, 'data_preprocessing/')
        encoder_saver = tf.train.Saver(saverdef_dict)  # too late to use the default, saverdef_dict is needed

        vae.summary()


        col_len = len(FEATURES)
        sample_len = SELECT_ROW_NUM // 2
        values, normal, mean_var, stdev_var = gpf.graph_normalization_factors_from_training_data(sample_len, col_len)
        xyztp_norm = tf.slice(normal, begin=[0, 0], size=[25, 5], name="xyztp_norm")
        t_norm_ = tf.cast(tf.slice(normal, begin=[0, 5], size=[25, 1]), dtype=tf.float64, name="t_norm_")
        t_norm = tf.reshape(t_norm_, shape=[-1], name="t_norm")

        #TODO what are these checkpoints
        # checkpoint = tf.train.Checkpoint(x=vae)
        # checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))

        # Encoded data
        e_xyztp = encoder(xyztp_norm)
        assert isinstance(e_xyztp, tfd.Distribution)
        e_xyztp_s = e_xyztp.sample() # .sample()
        e_xyztp_s = tf.cast(e_xyztp_s, dtype=tf.float64, name="encoded_sample")
        stack2 = tf.stack([e_xyztp_s, t_norm_])
        stack2 = tf.cast(stack2, dtype=tf.float64, name="enc_tr_")

        et_Var = tf.cast(stack2, dtype=tf.float64)
        w_pred_linsp_Var = tf.linspace(tf.reduce_min(e_xyztp_s), tf.reduce_max(e_xyztp_s), LINSPACE_GP_INDEX_SAMPLE,
                                       name="e_pred_linspace")
        w_pred_linsp_Var = tf.reshape(w_pred_linsp_Var, [-1, ENCODED_SIZE], name="reshape_")

        # Sample decoder
        z = prior.sample(SELECT_ROW_NUM)
        d_xyztp = decoder(z)
        assert isinstance(d_xyztp, tfd.Distribution)
        d_xyztp_s = d_xyztp.sample()


    ######################################################################
    # GRAPH CALLS ########################################################

    amp, amp_assign, amp_p, lensc, lensc_assign, lensc_p, \
    log_likelihood, samples_1d, train_op, obs_noise_var \
        = main.graph_GP(  et_Var,
                     t_norm,
                     w_pred_linsp_Var,
                     e_xyztp_s,
                     amplitude_init=AMPLITUDE_INIT,
                     length_scale_init=LENGTHSCALE_INIT,
                     obs_noise_var_init=INIT_OBSNOISEVAR,
                     LEARNING_RATE=LEARNING_RATE,
                     NUM_SAMPLES=NUM_SAMPLES
                   )

    # if GRAPH_COV_CALC == True:
    [cov_vv, last_cov_vv, while_i0_idx, while_i0_end] = main.graph_cov(SPATIAL_COVER, SPATIAL_COVER_PRESSURE_TEMP, encoder)

    [sel_idx, _, _, _] = snps2.sparse_placement_algorithm_2(cov_vv, K_SENSORS)

    sel_idx_ = main.graph_output(sel_idx)

    ######################################################################
    # GRAPH SAVER ########################################################
    print("LOGDIR",LOGDIR)
    # for i, var in enumerate(saver._var_list):
    #     print('Var {}: {}'.format(i, var))
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    saver = tf.train.Saver()  # all the object in the graph mapped

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # after the init
    # allmodel_saved_path = saver.save(sess, './saved_variable')
    # print('model saved in {}'.format(allmodel_saved_path))
    writer.add_graph(sess.graph)

    checkpoint = tf.train.Checkpoint(x=vae)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))


    ######################################################################
    # SESS RUNS ##########################################################
    for _ in range(1):
        e_test = encoder(tf.reshape(tf.constant([1., 1., 1., 1., 1.]), [-1, 5]))
    assert isinstance(e_test, tfd.Distribution)
    e_test_s = e_test.mean()
    print('encoder([1,1,1,1,1]) before restore', sess.run(e_test_s))

    print(CHECKPOINT_PATH)

    for _ in range(1):
        e_test = encoder(tf.reshape(tf.constant([1., 1., 1., 1., 1.]), [-1, 5]))
    assert isinstance(e_test, tfd.Distribution)
    e_test_s = e_test.mean()
    print('encoder([1,1,1,1,1]) after restore', sess.run(e_test_s))

    for _ in range(1):
        e_test = encoder(tf.reshape(tf.constant([1., 1., 1., 1., 1.]), [-1, 5]))
        assert isinstance(e_test, tfd.Distribution)
        e_test_s = e_test.mean()
        print('encoder([1,1,1,1,1]) before restore', sess.run(e_test_s))

    print(CHECKPOINT_PATH)
    encoder_saver.restore(sess, CHECKPOINT_PATH)
    # checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))

    for _ in range(1):
        e_test = encoder(tf.reshape(tf.constant([1., 1., 1., 1., 1.]), [-1, 5]))
        assert isinstance(e_test, tfd.Distribution)
        e_test_s = e_test.mean()
        print('encoder([1,1,1,1,1]) after restore', sess.run(e_test_s))

    # saver.restore(sess, chkpt_name)
    pass


if __name__ == '__main__':
    test_restore_chkpt_in_graph()
    pass
