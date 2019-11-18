###########################################
'''
All tests go here.
Plots, functions, tf fns, imported.
'''
# TODO
# preprocessing.py as pre_py
# normalize_delete.py as norm_py
# covariance.py as cov_py
# load_data.py as ld_py
# gaussian_process.py as gp_py
# vae.py as vae_py
# placement_algorithm.py as algo_py
###########################################
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#  raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

# !ls /usr/local/
# !nvidia-smi

import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers
from tensorflow_probability import positive_semidefinite_kernels as tfkern

import numpy as np
import pandas as pd
import datetime
import time
import sys

import placement_algorithm2 as alg2
import plots as plts
import gp_functions as gpf
import snippets_a2 as snps2

###################################################################
# GRAPH PRINT
###################################################################
def tf_print2(op, tensors, message=None):

    def print_message2(*args):
        str_ = message
        for s_ in [str(x) for x in args]:
            str_ += s_ + ', '
        sys.stdout.write(str_ + '\n')
        return args

    prints = tf.numpy_function(print_message2, [t for t in tensors], [t.dtype for t in tensors])
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op


###################################################################
# GP GRAPH
###################################################################
def graph_GP(et_Var,
             t_norm_,
             w_pred_linsp_Var,
             e_xyztp_s,
             amplitude_init=np.array([0.1, 0.1]),
             length_scale_init=np.array([.001, .001]),
             obs_noise_var_init=1e-3,
             LEARNING_RATE=.1,
             NUM_SAMPLES=8
             ):

    with tf.name_scope("GP"):
        # ===================================================================
        # AMP LENSC
        # ===================================================================
        with tf.name_scope("amplitude_lengthscale"):
            amp, amp_assign, amp_p, \
            lensc, lensc_assign, lensc_p, \
            emb, emb_assign, emb_p, \
            obs_noise_var \
                = gpf.tf_Placeholder_assign_test(amplitude_init, length_scale_init, obs_noise_var_init)

        # ===================================================================
        # KERNEL
        # ===================================================================
        with tf.name_scope("kernel"):
            kernel = tfkern.MaternOneHalf(amp, lensc)  # ExponentiatedQuadratic # MaternOneHalf

        # ===================================================================
        # GP_FIT
        # ===================================================================
        with tf.name_scope("GP_fit"):
            # gp = gpf.fit_gp(kernel, np.array(enc_df).reshape(-1, enc_df.shape[1]).astype(np.float64), obs_noise_var)  # GP fit to 3D XYZ, where Z is sinusoid of XY
            gp = gpf.fit_gp(kernel,
                            # np.array(enc_df).reshape(-1, enc_df.shape[1]).astype(np.float64),
                            e_xyztp_s, # et_Var
                            obs_noise_var)  # GP fit to 3D XYZ, where Z is sinusoid of XY
            log_likelihood = gp.log_prob(tf.transpose(tf.cast(t_norm_, dtype=tf.float64)))
            tf.summary.scalar("log_likelihood_0", log_likelihood[0])
            tf.summary.scalar("log_likelihood_1", log_likelihood[1])  # 2D GP input case
            tf.summary.scalar("length_scale", lensc[0])
            tf.summary.scalar("amplitude", amp[0])
            train_op = gpf.tf_train_gp_adam(log_likelihood, LEARNING_RATE)

        # ===================================================================
        # GP REGRESSION MODEL
        # ===================================================================
        with tf.name_scope("GP_regression_model"):
            gprm = gpf.tf_gp_regression_model(kernel,
                                              w_pred_linsp_Var,  # pred_idx_pts 1D_emb:(400,1), 2D_emb:(200,2)
                                              e_xyztp_s, # e_xyztp_s # obs_idx_pts(15,1) (15,2)
                                              t_norm_,  # obs (15,) (15,)
                                              obs_noise_var, 0.)
            samples_1d = gprm.sample(NUM_SAMPLES)

    return amp, amp_assign, amp_p, lensc, lensc_assign, lensc_p, log_likelihood, samples_1d, train_op, obs_noise_var


###################################################################
# COVARIANCE GRAPH
###################################################################
def graph_cov(SPATIAL_COVER,
              SPATIAL_COVER_PRESSURE_TEMP,
              encoder
              ):
    ''' Generate placement cov matrix: cov_vv
        - grid of spatial points 20 x 20
        - extend to 5d
        - repeat 300 times
        - transformations as on training data
        - predict tracer
    -> 20x20x300 tracer values -> 300 needs to cover range of all temp, pressure and time values
    -> get cov_vv : 400x400x1
    '''

    #===================================================================
    # COVARIANCE INIT
    #===================================================================
    print("CREATING COV -----")
    with tf.name_scope("covariance"):
        with tf.name_scope("cov_init"):

            sptl_const = tf.constant(SPATIAL_COVER)
            sptl_pt_const = tf.constant(SPATIAL_COVER_PRESSURE_TEMP)

            linsp_t = tf.linspace(-3., 3., sptl_pt_const, name="linsp_t")
            linsp_p = tf.linspace(-3., 3., sptl_pt_const, name="linsp_p")

            samples_t = tf.placeholder(dtype=tf.float64, shape=sptl_pt_const, name="samples_p")
            samples_p = tf.placeholder(dtype=tf.float64, shape=sptl_pt_const, name="samples_t")

            T_, P_ = tf.meshgrid(samples_t, samples_p)
            stack_tp = tf.stack([T_, P_], axis=2, name="grid_tp")
            vec_tp = tf.reshape(stack_tp, [-1, 2], name="vec_tp")

            linsp_x = tf.linspace(-3., 3., sptl_const, name="linsp_x")
            linsp_y = tf.linspace(-3., 3., sptl_const, name="linsp_y")
            linsp_z = tf.linspace(-3., 3., sptl_const, name="linsp_z")

            linsp_x = tf.reshape(linsp_x, [-1], name="reshape_")
            linsp_y = tf.reshape(linsp_y, [-1], name="reshape_")
            linsp_z = tf.reshape(linsp_z, [-1], name="reshape_")

            I0 = tf.constant(SPATIAL_COVER, name="I0")
            I1 = tf.constant(SPATIAL_COVER, name="I1")
            I2 = tf.constant(SPATIAL_COVER, name="I2")

            cov_vv = tf.Variable(tf.zeros((I0 * I1 * I2, I0 * I1 * I2)), name="cov_vv")
            last_cov_vv = tf.Variable([3.14], name="last_cov_vv")
            xyz_cov_idxs = tf.Variable(tf.zeros((I0 * I1 * I2, 3)), name="xyz_cov_idxs")

            cond0 = lambda i_0, I0_: tf.less(i_0, I0_)

        with tf.name_scope("calc_xyz_idxs"):
            def xyz_idxs_to_match_cov(j_0, j_1, j_2, i):

                pass

        # ===================================================================
        # COV I J
        # ===================================================================
        with tf.name_scope("calc_cov_ij"):

            def inside_node(j_0, j_1, j_2, tracers_loc_i, i, j):
                loc_xyz_j = gpf.slice_grid_xyz(j_0, j_1, j_2, linsp_x, linsp_y, linsp_z)
                tracers_loc_j = gpf.graph_get_tracers_for_coordloc(loc_xyz_j, vec_tp, encoder)

                print_tr = tf.print([tracers_loc_i, tracers_loc_j], output_stream=sys.stdout, name='print_test')
                with tf.control_dependencies([print_tr]):
                    tr_mean = tf.constant([0.0018159087825037148])
                    tr_stdev = tf.constant([0.0007434639347162126*3000000])

                    t_i_ = tracers_loc_i - tr_mean
                    t_j_ = tracers_loc_j - tr_mean

                    # t_i_ = tracers_loc_i - tf.fill(dims=tracers_loc_i.shape, value=tr_mean)
                    # t_j_ = tracers_loc_j - tf.fill(dims=tracers_loc_j.shape, value=tr_mean)

                    t_i = t_i_ / tr_stdev
                    t_j = t_j_ / tr_stdev
                    cov_vv_ij = tfp.stats.covariance(t_i, t_j, sample_axis=0, event_axis=None)

                    # i_sub1 = tf.subtract(tracers_loc_i, i_mean1)
                    # j_sub1 = tf.subtract(tracers_loc_j, j_mean1)
                    # i_cast_ = tf.cast(i_sub1, dtype=tf.float32)
                    # i_cast = tf.reshape(i_cast_, [-1])
                    # j_cast_ = tf.cast(j_sub1, dtype=tf.float32)# then reshape to 1d
                    # j_cast = tf.reshape(j_cast_, [-1])
                    # tr_shape_ = tf.shape(tracers_loc_i)[0]
                    # tr_shape = tf.reshape(tr_shape_, [-1])
                    # var_mult = tf.multiply(j_var1, i_var1)
                    # cov_vv_ij = tf.tensordot(i_cast, j_cast, axes=[0])/tf.multiply(var_mult, tf.cast(tr_shape, dtype=tf.float32))

                    update = tf.reshape(cov_vv_ij, [], name="update")

                    indexes1 = tf.cast([i, j], dtype=tf.int32, name='indexes')
                    indexes2 = tf.cast([j, i], dtype=tf.int32, name='indexes')

                    op1 = tf.scatter_nd_update(cov_vv, [indexes1], [update])  # [[i0, 0]], [3.+i0+0])
                    op2 = tf.scatter_nd_update(cov_vv, [indexes2], [update])  # [[i0, 0]], [3.+i0+0])
                    op3 = last_cov_vv.assign(cov_vv_ij)

                    # update a row of x y z
                    update_idx_x = tf.reshape(j_0, [], name="update_x")
                    update_idx_y = tf.reshape(j_1, [], name="update_y")
                    update_idx_z = tf.reshape(j_2, [], name="update_z")
                    indexes_x = tf.cast([j, 0], dtype=tf.int32,name='indexes_x')
                    indexes_y = tf.cast([j, 1], dtype=tf.int32,name='indexes_y')
                    indexes_z = tf.cast([j, 2], dtype=tf.int32,name='indexes_z')
                    # xyz_cov_idxs
                    op4 = tf.scatter_nd_update(xyz_cov_idxs, [indexes_x], [update_idx_x])
                    op5 = tf.scatter_nd_update(xyz_cov_idxs, [indexes_y], [update_idx_y])
                    op6 = tf.scatter_nd_update(xyz_cov_idxs, [indexes_z], [update_idx_z])

                return op1, op2, op3, op4, op5, op6

        with tf.name_scope("cov_calc"):

            def body0(i_0, I0_):
                cond1 = lambda i_1, I1_: tf.less(i_1, I1_)

                def body1(i_1, I1_):
                    cond2 = lambda i_2, I2_: tf.less(i_2, I2_)

                    # loc_xyz = gpf.slice_grid_xyz(i_0, i_1, i_2, linsp_x, linsp_y, linsp_z)
                    # tracers_loc_i = gpf.graph_get_tracers_for_coordloc(loc_xyz, vec_tp, encoder)

                    def body2(i_2, I2_):
                        cond_0 = lambda j_0, J0_: tf.less(j_0, J0_)

                        # we'd like to know the index=i of a position (i_0, i_1, i_2)
                        i_stride0 = I2_ * I1_
                        i_stride1 = I2_
                        i_stride2 = 1

                        i_mult0 = tf.multiply(i_0, i_stride0)
                        i_mult1 = tf.multiply(i_1, i_stride1)
                        i_mult2 = tf.multiply(i_2, i_stride2)
                        i_add = tf.add(i_mult2, i_mult1)
                        i = tf.add(i_mult0, i_add)

                        loc_xyz_i = gpf.slice_grid_xyz(i_0, i_1, i_2, linsp_x, linsp_y, linsp_z)
                        tracers_loc_i = gpf.graph_get_tracers_for_coordloc(loc_xyz_i, vec_tp, encoder)

                        def body_0(j_0, J0_):
                            cond_1 = lambda j_1, J1_: tf.less(j_1, J1_)

                            def body_1(j_1, J1_):
                                def cond_2(j_2, J2_):
                                    j_stride0 = J2_ * J1_
                                    j_stride1 = J2_
                                    j_stride2 = 1

                                    j_mult0 = tf.multiply(j_0, j_stride0)
                                    j_mult1 = tf.multiply(j_1, j_stride1)
                                    j_mult2 = tf.multiply(j_2, j_stride2)
                                    j_add = tf.add(j_mult2, j_mult1)
                                    j = tf.add(j_mult0, j_add)

                                    # j = tf_print2(j_, [i, j_], message='i, j = ')

                                    # while j <= i:
                                    #   while i < I0*I1*I2
                                    return tf.math.logical_and(tf.less(j_2, J2_), tf.less_equal(j, i))

                                def body_2(j_2, J2_):
                                    # we'd like to know the index=j of a position (j_0, j_1, j_2)
                                    j_stride0 = J2_ * J1_
                                    j_stride1 = J2_
                                    j_stride2 = 1

                                    j_mult0 = tf.multiply(j_0, j_stride0)
                                    j_mult1 = tf.multiply(j_1, j_stride1)
                                    j_mult2 = tf.multiply(j_2, j_stride2)
                                    j_add = tf.add(j_mult2, j_mult1)
                                    j_ = tf.add(j_mult0, j_add)

                                    j = tf_print2(j_, [i, j_], message='i, j = ')

                                    op1, op2, op3, op4, op5, op6 = inside_node(j_0, j_1, j_2, tracers_loc_i, i, j)


                                    # print = tf.print([cov_vv[i, j], cov_vv_ij], output_stream=sys.stdout, name='print_test')

                                    with tf.control_dependencies([op1, op2, op3, op4, op5, op6]):
                                        j_2_next = tf.add(j_2, 1)
                                    return [j_2_next, J2_]

                                [loop_j2, _] = tf.while_loop(cond_2, body_2, loop_vars=[0, I2], name="loop_j2")
                                with tf.control_dependencies([loop_j2]):
                                    j_1_next = tf.add(j_1, 1)
                                return [j_1_next, J1_]  # end body_1
                                # tf.while_loop(cond_2, body_2, loop_vars=[j_2, iters])
                                # return [tf.add(j_1, 1), iters]  # end body_1

                            [loop_j1, _] = tf.while_loop(cond_1, body_1, loop_vars=[0, I1], name="loop_j1")
                            with tf.control_dependencies([loop_j1]):
                                j_0_next = tf.add(j_0, 1)
                            return [j_0_next, J0_]  # end body_0
                            # tf.while_loop(cond_1, body_1, loop_vars=[j_1, iters])
                            # return [tf.add(j_0, 1), iters]  # end body_0

                        [loop_j0, _] = tf.while_loop(cond_0, body_0, loop_vars=[0, I0], name="loop_j0")
                        with tf.control_dependencies([loop_j0]):
                            i_2_next = tf.add(i_2, 1)
                        return [i_2_next, I2_]  # end body2
                        # tf.while_loop(cond_0, body_0, loop_vars=[j_0, iters])
                        # return [tf.add(i_2, 1), iters]  # end body2

                    [loop_i2, _] = tf.while_loop(cond2, body2, loop_vars=[0, I2], name="loop_i2")
                    with tf.control_dependencies([loop_i2]):
                        i_1_next = tf.add(i_1, 1)
                    return [i_1_next, I1_]  # end body1
                    # tf.while_loop(cond2, body2, loop_vars=[i_2, iters])
                    # return [tf.add(i_1, 1), iters]  # end body1

                [loop_i1, _] = tf.while_loop(cond1, body1, loop_vars=[0, I1], name="loop_i1")
                with tf.control_dependencies([loop_i1]):
                    i_0_next = tf.add(i_0, 1)
                return [i_0_next, I0_]  # end body0

            [while_i0_idx, while_i0_end] = tf.while_loop(cond0, body0, loop_vars=[0, I0], name="loop_i0")
        return cov_vv, last_cov_vv, while_i0_idx, while_i0_end, xyz_cov_idxs

# with tf.name_scope("placement_coordinates"):
    # def graph_output(sel_idx, xyz_idxs):
    #     # print_pl = tf.print(sel_idx, output_stream=sys.stdout, name='selection_print')
    #     # with tf.control_dependencies([print_pl]):
    #     sel_coord = gpf.py_get_coord_idxs(sel_idx, xyz_idxs)
    #     return sel_coord


####################################################################
# ENCODED TO GP TEST
####################################################################
TEST_encode_to_GP = True
def TEST_encode_to_GP_fn():

    #===================================================================
    # CONSTANTS
    #===================================================================
    # Files #
    CHECKPOINT_PATH = "./vae_training/model.ckpt"
    LOGDIR = "./log_dir_/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')+"_gp"
    LOGCHECKPT = "model.ckpt"
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


    #===================================================================
    # DATA
    #===================================================================
    # vae = tf.saved_model.load(SAVED_MODEL_DIR)
    # VAE
    vae, prior, encoder, decoder = gpf.create_model(ENCODED_SIZE, 5)
    encoder_saver = tf.train.Saver()  # when parsing the default graph, it has only the VAE_ yet

    with tf.name_scope("data_preprocessing"):
        col_len = len(FEATURES)
        sample_len = SELECT_ROW_NUM // 2
        values, normal, mean_var, stdev_var = gpf.graph_normalization_factors_from_training_data(sample_len, col_len)
        xyztp_norm = tf.slice(normal, begin=[0, 0], size=[25, 5], name="xyztp_norm")
        t_norm_ = tf.cast(tf.slice(normal, begin=[0, 5], size=[25, 1]), dtype=tf.float64, name="t_norm_")
        t_norm = tf.reshape(t_norm_, shape=[-1], name="t_norm")

        # VAE : at this point we have mean and stdev as extra
        # vae, prior, encoder, decoder = gpf.create_model(ENCODED_SIZE, 5)
        # encoder_saver = tf.train.Saver()  # too late, mean will be missing from the checkpoint
        """ [   
            'VAE_/decoder_/d_dense_/bias', 
            'VAE_/decoder_/d_dense_/kernel', 
            'VAE_/decoder_/d_dense_10/bias', 
            'VAE_/decoder_/d_dense_10/kernel', 
            'VAE_/encoder_/e_dense_/bias', 
            'VAE_/encoder_/e_dense_/kernel',
            'VAE_/encoder_/e_dense_10/bias', 
            'VAE_/encoder_/e_dense_10/kernel', 
            'VAE_/encoder_/e_mvn_dense_/bias', 
            'VAE_/encoder_/e_mvn_dense_/kernel', 
            'mean', 
            'stdev'
        ] """
        # 'trick' to figure out the list of variables to restore later

        vae.summary()

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


    #===================================================================
    # GRAPH CALLS
    #===================================================================
    # I.3 - GP training
    amp, amp_assign, amp_p, lensc, lensc_assign, lensc_p, \
        log_likelihood, samples_1d, train_op, obs_noise_var \
        = graph_GP(et_Var,
                 t_norm,
                 w_pred_linsp_Var,
                 e_xyztp_s,
                 amplitude_init=AMPLITUDE_INIT,
                 length_scale_init=LENGTHSCALE_INIT,
                 obs_noise_var_init=INIT_OBSNOISEVAR,
                 LEARNING_RATE=LEARNING_RATE,
                 NUM_SAMPLES=NUM_SAMPLES
                 )

    #===================================================================
    # II.1 - COV calc
    #===================================================================
    [cov_vv, last_cov_vv, while_i0_idx, while_i0_end, xyz_cov_idxs] \
        = graph_cov(SPATIAL_COVER, SPATIAL_COVER_PRESSURE_TEMP, encoder)

    #===================================================================
    # II.2 - PLACEMENT
    #===================================================================
    [sel_idx, _, _, _] = snps2.sparse_placement_algorithm_2(cov_vv, K_SENSORS)

    # sel_coord = graph_output(sel_idx)  # doesnt work yet

    #===================================================================
    # GRAPH SAVER
    #===================================================================
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

    #===================================================================
    # ASSERTS
    #===================================================================
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

    #===================================================================
    # DATA - LOAD RANDOMIZE SELECT SPLIT TO TRAIN, TEST
    #===================================================================
    xyztpt_idx_df = gpf.randomize_df(df, SELECT_ROW_NUM, dim1, dim2, dim3, dim4, dim5, dim6)
    xyztpt_idx = xyztpt_idx_df.to_numpy()
    minmax_x = [np.min(xyztpt_idx[:, 0]), np.max(xyztpt_idx[:, 0])]
    minmax_y = [np.min(xyztpt_idx[:, 1]), np.max(xyztpt_idx[:, 1])]
    minmax_z = [np.min(xyztpt_idx[:, 2]), np.max(xyztpt_idx[:, 2])]
    minmax_temperature = [np.min(xyztpt_idx[:,3]),np.max(xyztpt_idx[:,3])]
    minmax_pressure = [np.min(xyztpt_idx[:,4]),np.max(xyztpt_idx[:,4])]
    # INPUT_SHAPE = xyztpt_idx.shape[1]-1
    train_dataset, test_dataset = gpf.load_randomize_select_train_test(xyztpt_idx)


    #===================================================================
    # SESS RUNS
    #===================================================================
    normal_v = sess.run([normal], feed_dict={values: train_dataset})
    [obs_noise_var_] = sess.run([obs_noise_var])  # run session graph
    lls = gpf.tf_optimize_model_params(sess, NUM_ITERS, train_op,
                                       log_likelihood, # (2,0)
                                       summ, writer, saver, LOGDIR,
                                       LOGCHECKPT, train_dataset, values)  # takes long!
    [amp, lensc, obs_noise_var] = sess.run([amp, lensc, obs_noise_var])

    [samples_1d_] = sess.run([samples_1d], feed_dict={values: train_dataset})  # 1D_emb:(5, ?, ?), 2D_emb:(5,2,200)

    H = gpf.calc_H(XEDGES, YEDGES, lensc, lensc_assign, lensc_p, amp, amp_assign, amp_p, log_likelihood, sess, values, train_dataset)

    e_xyztp_s_v = sess.run(e_xyztp_s, feed_dict={values: train_dataset})
    d_xyztp_s_v = sess.run(d_xyztp_s, feed_dict={values: train_dataset})
    w_pred_idx_linsp = sess.run(w_pred_linsp_Var, feed_dict={values: train_dataset})
    e_reduced = sess.run(e_xyztp_s, feed_dict={values: test_dataset}).astype(np.float64)[:ROW_REDUCED, :]
    e_v = sess.run(et_Var, feed_dict={values: train_dataset})
    enc_df = pd.DataFrame(e_v[:,:,0].T)
    tr_idx = sess.run(t_norm_, feed_dict={values: train_dataset})
    tr_idx_reduced = np.array(tr_idx[:ROW_REDUCED]).flatten()
    xyztp_idx = sess.run(xyztp_norm, feed_dict={values: train_dataset})
    print(xyztp_idx)


    [w0, w1, e] = sess.run([while_i0_idx, while_i0_end, cov_vv], feed_dict={values: test_dataset})
    elast = sess.run(last_cov_vv, feed_dict={values: train_dataset})

    print("cov_vv, after running while_op")
    print(w0, w1)
    print(e)
    print("last_cov_vv_ij")
    print(elast)
    cov_vv_ = sess.run(cov_vv)
    print(cov_vv_)
    df = pd.DataFrame(cov_vv_)
    df.to_csv('cov_vv.csv')

    # np_algo1 = alg2.placement_algorithm_1(cov_vv_, 7)
    # print("np algorithm 1",np_algo1)

    np_algo2 = alg2.placement_algorithm_2(cov_vv_, 7)
    print("np algorithm 2",np_algo2)

    select_placement_points = sess.run(sel_idx.values)  # after the print node
    print('tf algorithm 2, select_placement_points: ', select_placement_points)
    xyz_idxs = sess.run(xyz_cov_idxs)
    sel_norm_coord = gpf.py_get_coord_idxs(select_placement_points, xyz_idxs)
    print("xyz_coordinates", sel_norm_coord)
    sel_coord = gpf.denormalize_coord(sel_norm_coord)
    print("denormalized xyz",sel_coord)
    # p = sess.run(force['x'], feed_dict={values: test_dataset})
    # print("p",p)

    # [_, assigned_val] = sess.run([assign_op, cov_vv_ij], feed_dict={assign_placeholder:. val})

    # bigwhileloop = tf.map_fn(simple_args_test, vec3)  #  xyz = tf.constant
    ######################################################################
    #### TEST: 3 sensors from 8 indices ####
    # if GRAPH_COV_CALC==True:


    # if PYTHON_COV_CALC == True:
    #     cov_vv_python = gpf.create_cov_matrix_while_loops(minmax_x, minmax_y, minmax_z, minmax_pressure, minmax_temperature, SPATIAL_COVER, SPATIAL_COVER_PRESSURE_TEMP, encoder, sess)
    #     print(cov_vv_python)
    #     A = snps.placement_algorithm_2(cov_vv_python, k=K_SENSORS)
    #     print("A: ",A)

    # DELETE
    # cov_vv = gpf.tf_create_cov_matrix(minmax_x_tnsr, minmax_y_tnsr, minmax_z_tnsr, minmax_p_tnsr, minmax_t_tnsr, sptl_const, sptl_pt_const, encoder, sess)
    # cov_vv = gpf.create_cov_matrix(minmax_x,minmax_y,minmax_z,minmax_pressure, minmax_temperature, SPATIAL_COVER, SPATIAL_COVER_PRESSURE_TEMP, encoder, sess)

    #===================================================================
    # PLOTS
    #===================================================================
    plts.pairplots(enc_df)  # e_t_df
    plts.plot_encoder_output_distribution(12, 12, xyztp_idx[:, 0])
    plts.plot_encoder_output_distribution(12, 12, e_xyztp_s_v[:,0])
    plts.plot_decoder_output_distribution(12, 12, d_xyztp_s_v[:,0,0,:])

    plts.plot_loss_evolution(12, 4, lls)
    plts.plot_marginal_likelihood3D(XEDGES, H)
    plts.plot_gp_linesamples(12,4, e_reduced, # (15,1)
                             tr_idx_reduced, # (15,)
                             w_pred_idx_linsp, # (100,1)
                             samples_1d_, # (8,2,100)
                             NUM_SAMPLES) # (8)

    plts.plot_range_of_values(12,4, e_xyztp_s_v, tr_idx)
    # plts.plot_placement_xyz(12,12,select_placement_points)
    ''' DIMENSIONS 
    plts.plot2d_sinusoid_samples_section(12, 4, ext_sel_pts,  # (12,4)
                                         line_idx,  # (200,1)
                                         obs_proj_idx_pts,  # (60,2)
                                         line_obs,  # (200,)
                                         samples_section_,  # (50,2,200)
                                         NUM_SAMPLES,  # (50)
                                         BEGIN, END)  # (3,)
                                         '''
    print("==============")

if __name__ == '__main__':
    if TEST_encode_to_GP == True:
        TEST_encode_to_GP_fn()
    pass
