###########################################
'''
All tests go here.
Plots, functions, tf fns, imported.
'''
###########################################
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

import imports as im
import gp_functions as gpf
import data_generation as dg
import plots as plts
import data_readers as drs
import main_GP_fit
import placement_algorithm2 as alg2

#================================================================
# TEST_snippets_tf_where
#================================================================
def TEST_snippets_tf_where():

    V = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [3, 0]], values=[7, 8, 9, 10], dense_shape=[24, 1])
    A = tf.SparseTensor(indices=[[0, 0], [2, 0]], values=[7, 9], dense_shape=[24, 1])
    cache = tf.Variable([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # sparse_argmax_cache_linear fn
    V_minus_A = tf.sets.difference(V, A)
    cache = tf.Variable([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    delta_y_s = tf.reshape(tf.gather_nd(
                            tf.reshape(cache, [-1, 1]),
                            tf.reshape(V_minus_A.indices[:, 0], [-1, 1])
                           ), [-1])
    delta_st = tf.reduce_max(delta_y_s)  # max
    val_idx = tf.where(tf.equal(cache, delta_st))
    # # val_col = 0
    # # index_col = 1
    y_st = val_idx[0, 0]  # get index of delta_st
    # ret y_st

    y_st2 = tf.function(alg2.sparse_argmax_cache_linear)(cache, A, V)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('V: ', sess.run(V.values))
        print('A: ', sess.run(A.values))
        print('V_minus_A: ', sess.run(V_minus_A.values))
        print('V_minus_A.indices[:, 0]: ', sess.run(V_minus_A.indices[:, 0]))
        print('cache: ', sess.run(cache))
        print('delta_y_s: ', sess.run(delta_y_s))
        print('delta_st: ', sess.run(delta_st))
        print('val_idx: ', sess.run(val_idx))
        print('y_st: ', sess.run(y_st))

        print('y_st2: ', sess.run(y_st2))

        # print('y_st: ', y_st)
    pass

#================================================================
# TEST_tf_map_fn_nd_reduce_any
#================================================================
def TEST_tf_map_fn_nd_reduce_any():
    elems = tf.constant([1, 2, 3, 4, 5, 6])
    y = tf.constant(3)
    fn = lambda x: x == y
    eq = tf.map_fn(fn , elems)  # returns tf.Tensor([False, False, True, False, False, False])
    is_eq = tf.reduce_any(tf.cast(eq, tf.bool))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # after the init
    equal = sess.run(is_eq)
    print("equality:")
    print(equal)

    pass

#================================================================
# TEST_tracer_data_GP
#================================================================
def TEST_tracer_data_GP():
    main_GP_fit.TEST_tracer_data_GP_()  # tracer data GP in another file.

#================================================================
# TEST_cov_while_loop
#================================================================
def TEST_cov_while_loop():
    LOGDIR = "./log_dir_/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')+"_cov__test"

    def graph():
        cov_vv = tf.Variable(tf.zeros((5, 5)), name="cov_vv")
        last_cov_ij = tf.Variable([7.], name="last_cov_ij")
        i_0 = tf.constant(0.)
        iters = tf.constant(5.)
        cond0 = lambda i_0, iters : tf.less(i_0, iters)
        # def read_value()
        def body0(i_0, iters):

            i_ = lambda i_0 : tf.less_equal(i_0, 100.)
            zero = lambda: tf.ones([], dtype=tf.int32)
            v = tf.Variable(initial_value=zero, dtype=tf.int32)

            # cov.assign(i_0)
            cov_vv_ij  = tf.constant([2], name="cov_vv_ij")
            op1 = cov_vv[i, j].assign(cov_vv_ij, name="assign_cov_vv_ij")
            op2 = cov_vv[i, j].assign(cov_vv_ij, name="assign_cov_vv_ji")  # symmetric
            op3 = last_cov_ij.assign(cov_vv_ij)

            print = tf.print([cov_vv[i, j], cov_vv_ij], output_stream=sys.stdout, name='print_test')

            with tf.control_dependencies([op1, op2, op3 ,print]):
                i_0_next = tf.add(i_0, 1)

            return [i_0_next, iters]

        tf.while_loop(cond0, body0, loop_vars=[i_0, iters])
        return cov_vv, last_cov_ij


        # def cond(i, _):
        #     return i < 10
        #
        # def body(i, _):
        #     zero = tf.zeros([], dtype=tf.int32)
        #     v = tf.Variable(initial_value=zero)
        #     return (i + 1, v.read_value())
        #
        # def body_ok(i, _):
        #     zero = lambda: tf.zeros([], dtype=tf.int32)
        #     v = tf.Variable(initial_value=zero, dtype=tf.int32)
        #     return (i + 1, v.read_value())
        #
        # tf.while_loop(cond, body_ok, [0, 0])

    cov_vv, last_cov_ij = graph()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    covpy = sess.run(cov_vv)
    print(covpy)
    lastcovpy = sess.run(last_cov_ij)
    print(covpy)
    print(lastcovpy)
    #
    # pred = tf.placeholder()
    # x = tf.Variable([1])


    # def update_x_2():
    #     with tf.control_dependencies([tf.assign(x, [2])]):
    #         return tf.
    #
    #
    # y = tf.cond(pred, update_x_2, lambda: tf.less(x, ))
    #
    # with tf.Session() as session:
    #     session.run(tf.initialize_all_variables())
        # print(y.eval(feed_dict={pred: 5.}))  # ==> [1]
        # print(y.eval(feed_dict={pred: 1.}))  # ==> [2]

    pass

#================================================================
# TEST_VAE
#================================================================
def TEST_VAE():
    '''
    Phases of VAE
    - Train NN with 1 neuron in middle layer
        1. import dataset for 1 timestep
        2. randomize, reduce size to (100, 6)
        2. pause after training is complete
        3. save NN model
    - Create reduced space dataset, use encoder
        4. save new red_ord_dataset (100, 1)
    - Plot new data
        5. 2 d plot, see if its possible to fit GP to it
    - fit GP to new data
        6. fit GP to 2D points.
    - sample from GP
        7. compare if that sample point transformed by decoder is epsilon distance from another given point. > see error
    '''
    SAVED_MODEL_PATH = "./vae_training/model.ckpt"
    SAVED_MODEL_DIR = "./vae_model/"
    CHECKPOINT_PATH = "./vae_training/model.ckpt"
    INPUT_SHAPE = 5
    ENCODED_SIZE = 1
    CSV = 'roomselection_1800.csv'
    SELECT_ROW_NUM = 8000
    dim1 = 'Points:0'
    dim2 = 'Points:1'
    dim3 = 'Points:2'
    dim4 = 'Temperature'
    dim5 = 'Pressure'
    dim6 = 'Tracer'

    # vae = tf.saved_model.load(SAVED_MODEL_DIR)
    ds = drs.load_csv('roomselection_1800.csv')
    xyzt_idx = np.array(ds.loc[:, [dim1, dim2, dim3, dim4, dim5]])
    # xyzt_idx, train_dataset, test_dataset = gpf.load_randomize_select_train_test(CSV, SELECT_ROW_NUM, dim1, dim2, dim3,
    #                                                                              dim4, dim5, dim6)
    train_dataset, test_dataset = gpf.load_randomize_select_train_test(xyzt_idx)  # tracer is unrelated - how
    assert train_dataset.shape[1] == INPUT_SHAPE

    vae, prior, encoder, decoder = gpf.create_model(ENCODED_SIZE, INPUT_SHAPE)
    vae.summary()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.Checkpoint(x=vae)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))

    # sample decoder
    z = prior.sample(8000)
    xtilde = decoder(z)
    assert isinstance(xtilde, tfd.Distribution)
    xtilde_s = xtilde.sample()
    xtilde_s_val = sess.run(xtilde_s)

    etilde = encoder(test_dataset)
    assert isinstance(etilde, tfd.Distribution)
    etilde_s = etilde.sample()
    etilde_s_val = sess.run(etilde_s)

    plts.plot_encoder_output_distribution(12, 12, test_dataset[:, 0])
    plts.plot_encoder_output_distribution(12, 12, etilde_s_val)
    plts.plot_decoder_output_distribution(12, 12, xtilde_s_val[:, 0, 0, :])

    pass

#================================================================
# TEST_algorithm2
#================================================================
def TEST_algorithm2():
    '''  algorithm 1 improved with priority queue
    gpf.placement_algorithm_2
    '''
    k = 5
    NUM_OF_COV_SIZES = 40
    n_sizes = np.linspace(10, NUM_OF_COV_SIZES, NUM_OF_COV_SIZES - 10, dtype=int)

    # n_array = np.array([10,20,30,40,50])
    n_array = np.array([])
    for i in n_sizes:
        n_array = np.append(n_array, i)

    times_array = np.array([])
    for n in n_array:
        n = int(n)
        cov = dg.create_random_cov(n)
        r_time = gpf.calculate_running_time_algorithm_2(cov, k)
        times_array = np.append(times_array, r_time)

    plts.plot_algorithm_scale(12, 4, n_array, times_array)

#================================================================
# TEST_algorithm1
#================================================================
def TEST_algorithm1():
    '''    
    gpf.placement_algorithm_1
    k = fix num of sensors to be placed, based on maximum Mutual Information
    n = number of locations to choose from
    nxn covariance matrix

    10n  ... 50n 100n
    t10  ... t50 t100

    this graph is plotted
    '''
    k = 5
    NUM_OF_COV_SIZES = 40
    n_sizes = np.linspace(10, NUM_OF_COV_SIZES, NUM_OF_COV_SIZES - 10, dtype=int)  # 10-100

    # n_array = np.array([10,20,30,40,50])
    n_array = np.array([])
    for i in n_sizes:
        n_array = np.append(n_array, i)

    times_array = np.array([])
    for n in n_array:
        n = int(n)
        cov = dg.create_random_cov(n)
        r_time = gpf.calculate_running_time_algorithm_1(cov, k)
        times_array = np.append(times_array, r_time)

    plts.plot_algorithm_scale(12, 4, n_array, times_array)

#================================================================
# TEST_section_cut_GP
#================================================================
def TEST_section_cut_GP():
    '''
    2D sin curve, 1D section gp fit and plots
    '''
    #################################################################
    PLOTS = True
    NUM_PTS_GEN = 100  # 1000
    RANGE_PTS = [-2, 2]
    COORDINATE_RANGE = np.array([[-2., 2.], [-2., 2.]])  # xrange, yrange
    NUM_TRAINING_POINTS = 60
    BEGIN = np.array([0, -2, 0])
    END = np.array([2, 2, 0])
    EPSILON = 0.2
    AMPLITUDE_INIT = np.array([0.1, 0.1])
    LENGTHSCALE_INIT = np.array([0.1, 0.1])
    LEARNING_RATE = .01
    INIT_OBSNOISEVAR = 1e-6
    INIT_OBSNOISEVAR_ = 0.001
    XEDGES = 60
    YEDGES = 60
    LOGDIR = "./log_dir_gp/gp_plot/"
    NUM_ITERS = 400  # 1000
    PRED_FRACTION = 50  # 50
    LOGCHECKPT = "model.ckpt"
    NUM_SAMPLES = 50  # 50
    PLOTRASTERPOINTS = 60
    PLOTLINE = 200

    sess = gpf.reset_session()
    amp, amp_assign, amp_p, lensc, lensc_assign, lensc_p, emb, emb_assign, emb_p, obs_noise_var \
        = gpf.tf_Placeholder_assign_test(AMPLITUDE_INIT, LENGTHSCALE_INIT, INIT_OBSNOISEVAR_)

    obs_idx_pts, obs = dg.generate_noisy_2Dsin_data(NUM_TRAINING_POINTS, 0.001, COORDINATE_RANGE)

    obs_proj_idx_pts = np.linspace(-2, 2, 200)
    obs_proj_idx_pts = gpf.project_to_line_coordinates(obs_proj_idx_pts, BEGIN, END)
    obs_proj_idx_pts = gpf.create_1d_w_line_coords(obs_proj_idx_pts, BEGIN, END)  # 1D
    print(obs_proj_idx_pts.shape)

    obs_rowvec = obs
    obs = obs.reshape(-1, 1)  # to column vector
    pts = gpf.add_dimension(obs_idx_pts, obs)
    # sel_pts_ = np.array(gpf.capture_close_pts(obs_idx_pts, BEGIN, END, EPSILON) )  # 2D another way to do it
    sel_pts = gpf.capture_close_pts_XY(pts, BEGIN, END, EPSILON)  # 3D

    sel_obs = np.zeros((0, 1))
    for i in sel_pts[:, 2]:
        sel_obs = np.concatenate((sel_obs, i), axis=None)
    ext_sel_pts = gpf.extend_pts_with_line_coord(sel_pts, BEGIN, END)  # 4D

    train_idx_pts_along_section_line = gpf.project_to_line_coordinates(sel_pts, BEGIN, END)
    train_idx_pts_along_section_line = gpf.create_1d_w_line_coords(train_idx_pts_along_section_line, BEGIN, END)  # 1D

    line_idx = gpf.create_line_coord_linspace(BEGIN, END, PLOTLINE)  # 1D
    line_XY = gpf.create_line_linspace(BEGIN, END, PLOTLINE)  # 2D
    line_obs = gpf.create_Z_sin_along_line_linspace(line_XY)  # 1D
    # GP
    kernel = gpf.create_cov_kernel(amp, lensc)
    gp = gpf.fit_gp(kernel, obs_idx_pts, obs_noise_var)  # GP fit to 3D XYZ, where Z is sinusoid of XY

    log_likelihood = gp.log_prob(obs_rowvec)
    gpf.tf_summary_scalar("log_likelihood[0, 0]", log_likelihood[0])
    gpf.tf_summary_scalar("log_likelihood[1, 0]", log_likelihood[1])
    train_op = gpf.tf_train_gp_adam(log_likelihood, LEARNING_RATE)

    summ, writer, saver = gpf.tf_summary_writer_projector_saver(sess, LOGDIR)  # initializations
    [_, lensc_, ] = gpf.do_assign(sess, lensc_assign, lensc, lensc_p, [0.5, 0.5])
    [obs_noise_var_] = sess.run([obs_noise_var])  # run session graph

    lls = gpf.tf_optimize_model_params(sess, NUM_ITERS, train_op, log_likelihood, summ, writer, saver, LOGDIR,
                                       LOGCHECKPT, obs, _)
    [amp, lensc, obs_noise_var] = sess.run([amp, lensc, obs_noise_var])

    pred_x = np.linspace(BEGIN[0], END[0], PRED_FRACTION, dtype=np.float64)
    pred_y = np.linspace(BEGIN[1], END[1], PRED_FRACTION, dtype=np.float64)
    pred_idx_pts = gpf.create_meshgrid(pred_x, pred_y, PRED_FRACTION)  # posterior = predictions
    pred_sel_idx_pts = gpf.line_coords(sel_pts, BEGIN, END)  # 1D observations along x.
    # print("line_idx.shape : ", line_idx.shape) #(200, 1)
    # print("obs_idx_pts_along_section_line.shape : ", train_idx_pts_along_section_line.shape) #(100, 1)
    # print("sel_obs.shape : ", sel_obs.shape) #(100, )
    # print(repr(train_idx_pts_along_section_line))
    # print(repr(sel_obs))

    # Gaussian process regression model
    gprm = gpf.tf_gp_regression_model(kernel, pred_idx_pts, obs_idx_pts, obs_rowvec, obs_noise_var, 0.)
    gprm_section = gpf.tf_gp_regression_model(kernel,
                                              line_idx,  # (200,1)
                                              train_idx_pts_along_section_line,  # (9,1)
                                              sel_obs,  # (9,)
                                              obs_noise_var, 0.)

    # posterior_mean_predict = im.tfd.gp_posterior_predict.mean()
    # posterior_std_predict = im.tfd.gp_posterior_predict.stddev()

    # samples
    # samples = gprm.sample(NUM_SAMPLES)
    # samples_ = sess.run(samples)

    samples_section = gprm_section.sample(NUM_SAMPLES)
    samples_section_ = sess.run(samples_section)

    print("kernel : ", kernel)
    samples = gprm.sample(NUM_SAMPLES)
    samples_ = sess.run(samples)

    #     print("==================")
    #     ##############
    #     samples_pts = gpf.add_dimension(obs_idx_pts, samples_[0, 0, :]) # obs_idx_pts : (81, 2) XY , sample_[0,0,:](81, 1)
    #     print("sample_pts.shape : ",samples_pts.shape)
    #
    #     samples_l = gpf.capture_close_pts_XY(samples_pts, BEGIN, END, 1*EPSILON)
    #     print("sample_l.shape : ",samples_l.shape)
    #
    #     samples_coord_along_line = np.zeros(samples_l.shape[0])
    #     for i in range(samples_l.shape[0]):
    #         d = samples_l[i,0:2] - BEGIN[0:2]
    #         dist = np.sqrt(np.dot(d,d))
    #         samples_coord_along_line[i] = dist
    #     # samples_l[:,0:2] #1D
    #
    #     samples_z = samples_l[:,2]
    #     s = np.array((samples_coord_along_line, samples_z))
    #     s = np.sort(s)
    #
    #     s = s.T
    #     print("s.shape : ",s.shape)
    #     samples_z = s[:,1]
    #     samples_coord_along_line = s[:,0]
    # ##############

    H = np.zeros([XEDGES, YEDGES])
    for i in range(XEDGES):
        for j in range(YEDGES):
            [_, _, L] = sess.run([lensc_assign, amp_assign, log_likelihood],
                                 feed_dict={lensc_p: [2 * np.double((1 + i) /
                                                                    XEDGES), lensc[1]],
                                            amp_p: [2 * np.double((1 + j) / YEDGES), amp[1]]})
            H[i, j] = L[0]
    # assert (amp.shape[1] == (2))
    # assert (lensc.shape[1] == (2))
    # 5th dimension = log-likelihood at points xyzw
    emb_p = im.tf.placeholder(shape=[XEDGES, YEDGES], dtype=np.float32, name='emb_p')
    emb = im.tf.Variable(im.tf.zeros([XEDGES, YEDGES]), name="log_probability_embedding")
    emb_as_op = emb.assign(emb_p, name='emb_as_op')

    [_] = sess.run([emb_as_op], feed_dict={emb_p: H})
    saver.save(sess, im.os.path.join(LOGDIR, LOGCHECKPT), NUM_ITERS + 1)

    # plts.plot_kernel(12,4, kernel,BEGIN, END)

    # PLOTS
    if PLOTS:
        plts.plot_sin3D_rand_points(12, 12, COORDINATE_RANGE, obs_idx_pts, obs, PLOTRASTERPOINTS)
        plts.plot_2d_observations(12, 12, pts, sel_pts, BEGIN, END)
        plts.plot_capture_line(12, 12, sel_pts, BEGIN, END)
        # plts.plot_section_observations(12, 7, ext_sel_pts, BEGIN, END)
        # GP
        plts.plot_loss_evolution(12, 4, lls)
        plts.plot_marginal_likelihood3D(XEDGES, H)
        # plts.plot_samples2D(12,5,1, pred_idx_pts, obs_idx_pts, obs, samples_, NUM_SAMPLES)

        plts.plot_samples3D(12, 12, 0, pred_idx_pts,
                            samples_,
                            obs_idx_pts,
                            obs,
                            PRED_FRACTION,
                            BEGIN, END)

        plts.plot2d_sinusoid_samples_section(12, 4, ext_sel_pts,  # (12,4)
                                             line_idx,  # (200,1)
                                             obs_proj_idx_pts,  # (60,2)
                                             line_obs,  # (200,)
                                             samples_section_,  # (50,2,200)
                                             NUM_SAMPLES,  # (50)
                                             BEGIN, END)  # (3,)
        # plts.plot_gp_2D_samples(12,7, pred_sel_idx_pts, sel_obs, line_idx, line_obs, NUM_SAMPLES, samples_l[:,0:2], samples_coord_along_line, samples_z)

    PLOTS = False

#================================================================
# TEST_section_cut
#================================================================
def TEST_section_cut():
    '''
    3D surface section: observations and true fn in 2D plot
    '''
    #################################################################
    PLOTS = True
    NUM_PTS_GEN = 100
    RANGE_PTS = [-2, 2]
    COORDINATE_RANGE = np.array([[-2., 2.], [-2., 2.]])  # xrange, yrange
    NUM_TRAINING_POINTS = 2800
    BEGIN = np.array([-2, -2, 0])
    END = np.array([2, 2, 0])
    EPSILON = 0.2

    # pts_rand = dg.generate_noisy_2Dsin_data(NUM_PTS_GEN,0.1, COORDINATE_RANGE)  # 3d pointcloud
    # pts_rand[:, 2] *= 0.5  # transform length of 3rd scale but it was better to calc dist for XY only
    # sel_pts_rand = gpf.capture_close_pts(pts_rand)

    obs_idx_pts, obs = dg.generate_noisy_2Dsin_data(NUM_TRAINING_POINTS, 0.001, COORDINATE_RANGE)
    obs = obs.reshape(-1, 1)  # to column vector
    pts = gpf.add_dimension(obs_idx_pts, obs)
    sel_pts = gpf.capture_close_pts_XY(pts, BEGIN, END, EPSILON)
    ext_sel_pts = gpf.extend_pts_with_line_coord(sel_pts, BEGIN, END)

    s = np.linspace(-2, 2, 120).reshape(-1, 1)

    # PLOTS
    if PLOTS:
        plts.plot_2d_observations(12, 7, pts, sel_pts, BEGIN, END)
        plts.plot_capture_line(12, 7, sel_pts, BEGIN, END)
        plts.plot_section_observations(12, 7, ext_sel_pts, BEGIN, END)
        # plts.plot_sin_and_sel_pts_at_section(x_new, plts, x_original)
        # GP

    PLOTS = False

#================================================================
# TEST_sinusoid_VGP
#================================================================
import variational_Gaussian_process_example as vgpe

def TEST_sinusoid_GP_VGP():
    '''
    2D sin curve gp fit and plots
    '''

    AMPLITUDE_INIT = np.array([0.1, 0.1])
    LENGTHSCALE_INIT = np.array([0.1, 0.1])
    INIT_OBSNOISEVAR_ = 0.001
    NUM_TRAINING_POINTS = 90  # 180 was too much
    LEARNING_RATE = .01
    XEDGES = 60
    YEDGES = 60
    LOGDIR = "./log_dir_gp/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')+"_sin_gp"
    NUM_ITERS = 1000
    LOGCHECKPT = "model_sin_gp.ckpt"
    PRED_FRACTION = 50
    COORDINATE_RANGE = np.array([[-2., 2.], [-2., 2.]])  # xrange, yrange
    COORDINATE_RANGE_1D = np.array([-2., 2.])
    NUM_SAMPLES = 20
    PLOTRASTERPOINTS = 60
    BEGIN = np.array([-2, -2, 0])
    END = np.array([2, 2, 0])
    sess_ = gpf.reset_session()

    # ===================================================================
    # GENERATE SIN DATA
    # ===================================================================
    def sin_2d_data(number_of_obs=NUM_TRAINING_POINTS, number_of_pred=PRED_FRACTION):
        batch, features = 1, 2
        obs_idx_pts, obs = dg.generate_noisy_2Dsin_data(number_of_obs, INIT_OBSNOISEVAR_, COORDINATE_RANGE)
        obs_idx_pts = obs_idx_pts.reshape([batch, number_of_obs, features])
        obs = obs.reshape([batch, number_of_obs])

        pred_x = np.linspace(-2, 2, number_of_pred, dtype=np.float64)
        pred_y = np.linspace(-2, 2, number_of_pred, dtype=np.float64)
        pred_idx_pts_ = gpf.create_meshgrid(pred_x, pred_y)
        pred_idx_pts_ = pred_idx_pts_.reshape([batch, -1, features])
        SELECT_ROW_NUM = pred_idx_pts_.shape[1]  # all
        col_len = features  # pred_idx_pts_.shape[2]
        sample_len = SELECT_ROW_NUM
        # values, normal, mean_var, stdev_var = gpf.graph_normalization_factors_from_training_data(sample_len, col_len)
        values = tf.placeholder(
                                shape=[number_of_obs],
                                # shape=[batch, number_of_obs],
                                dtype=tf.float64, name="input_values")
        return pred_idx_pts_, obs_idx_pts, obs, values


    def sin_1d_data(number_of_obs=NUM_TRAINING_POINTS, number_of_pred=PRED_FRACTION):
        obs_idx_pts, obs = dg.generate_noisy_1Dsin_data(number_of_obs, INIT_OBSNOISEVAR_, COORDINATE_RANGE_1D)

        pred_idx_pts = np.linspace(-2, 2, number_of_pred, dtype=np.float64)
        # pred_y = np.linspace(-2, 2, PRED_FRACTION, dtype=np.float64)
        # pred_idx_pts_ = gpf.create_meshgrid(pred_x, pred_y)
        # SELECT_ROW_NUM = pred_idx_pts.shape[0]  # all
        # col_len = 1 # pred_idx_pts_.shape[1]
        # sample_len = SELECT_ROW_NUM
        # values, normal, mean_var, stdev_var = gpf.graph_normalization_factors_from_training_data(sample_len, col_len)
        values = tf.placeholder(shape=[number_of_obs],
                                dtype=tf.float64, name="input_values")
        return pred_idx_pts, obs_idx_pts, obs, values


    def graph(pred_idx_pts_, obs_idx_pts, obs, values):
        # ===================================================================
        # AMP LENSC
        # ===================================================================
        feat_dims = pred_idx_pts_.shape[2] if 2 < len(pred_idx_pts_.shape) else 1
        obs_dims = obs.shape[1] if 1 < len(obs.shape) else obs.shape[0]
        batch_cnt = 1

        amp, amp_assign, amp_p, \
            lensc, lensc_assign, lensc_p, \
            _, _, _, obs_noise_var \
            = gpf.tf_Placeholder_assign_test(
                        AMPLITUDE_INIT[:1],
                        LENGTHSCALE_INIT[:1],  # shape [1]
                        INIT_OBSNOISEVAR_)

        # ===================================================================
        # KERNEL
        # ===================================================================
        kernel = tfkern.ExponentiatedQuadratic(amp, lensc,
                                               feature_ndims=1,  # number of rightmost dims to include in the squared difference norm
                                               validate_args=True)  # MaternOneHalf

        # ===================================================================
        # GP_FIT
        # ===================================================================
        """
          index_points: `float` `Tensor` representing finite (batch of) vector(s) of
            points in the index set over which the GP is defined. Shape has the form
            `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
            dimensions and must equal `kernel.feature_ndims` and `e` is the number
            (size) of index points in each batch. Ultimately this distribution
            corresponds to a `e`-dimensional multivariate normal. The batch shape
            must be broadcastable with `kernel.batch_shape` and any batch dims
            yielded by `mean_fn`.
        """
        # gp = gpf.fit_gp(kernel,
        #                 obs_idx_pts,
        #                 obs_noise_var)

        gp = tfd.GaussianProcess(
            kernel=kernel,  # ([2,],[2,])
            index_points=obs_idx_pts,
            observation_noise_variance=obs_noise_var,
            validate_args=True)

        print('feature_ndims: ', 1)
        print("kernel batch_shape:", kernel.batch_shape)
        print("kernel feature_ndims:", kernel.feature_ndims)
        print("amp shape:", amp.shape)
        print("lensc shape:", lensc.shape)

        print("obs_idx_pts shape:", obs_idx_pts.shape)
        print("values shape:", values.shape)
        print("Batch shape:", gp.batch_shape)
        print("Event shape:", gp.event_shape)

        assert kernel.batch_shape == gp.batch_shape
        assert kernel.feature_ndims == len(amp.shape) or 0 == len(amp.shape)
        assert kernel.feature_ndims == len(lensc.shape) or 0 == len(lensc.shape)

        """
        ---
        observed_index_points.shape:  (50, 1)
        observed_values.shape:  (50,)
        gp Batch shape: (1,)
        gp Event shape: (50,)
        kernel batch_shape: (1,)
        kernel feature_ndims: 1
        amp shape: (1,)
        lensc shape: (1,)
        neg_log_likelihood shape: (1,)
        """

        """get_marginal_distribution : 
              Compute the marginal of this GP over function values at `index_points`.        
        Args:
          index_points: `float` `Tensor` representing finite (batch of) vector(s) of
            points in the index set over which the GP is defined. Shape has the form
            `[b1, ..., bB, e, f1, ..., fF]`
             and `e` is the number (size) of index points in each batch."""

        log_likelihood = gp.log_prob(value=values)  # obs.reshape([batch_cnt, -1, obs_dims]))  # BATCH_SHAPE=(2,)
        print("neg_log_likelihood shape:", log_likelihood.shape)
        if 0 < len(log_likelihood.shape):
            assert log_likelihood.shape == [1]

        tf.summary.scalar("log_likelihood[0, 0]", log_likelihood[0])
        # tf.summary.scalar("log_likelihood[1, 0]", log_likelihood[1])
        tf.summary.scalar("length_scale", lensc[0])
        tf.summary.scalar("amplitude", amp[0])

        train_op = gpf.tf_train_gp_adam(log_likelihood, LEARNING_RATE)

        # ===================================================================
        # GP REGRESSION MODEL
        # ===================================================================
        gprm = gpf.tf_gp_regression_model(kernel,
                                          pred_idx_pts_.reshape([batch_cnt, -1, feat_dims]),  # (2500,2)  # 1D (50,)
                                          obs_idx_pts.reshape([batch_cnt, -1, feat_dims]),  # (90,2)  # 1D (90, 2)
                                          obs.reshape([batch_cnt, -1, obs_dims]),  # (90,)  # 1D  (90,)
                                          obs_noise_var, 0.)

        return train_op, gprm, \
               amp, amp_assign, amp_p, \
               lensc, lensc_assign, lensc_p, \
               obs_noise_var, log_likelihood


    #===================================================================
    # SESS
    #===================================================================
    def sess_runs(train_op, gprm, amp, amp_assign, amp_p, lensc_assign, lensc, lensc_p,
                  obs_noise_var, pred_idx_pts, values_placeholder, log_likelihood,
                  obs_idx_pts, obs_values):
        #-------------------------------------------------------------------
        # SESS INSTANCIATION
        #-------------------------------------------------------------------
        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOGDIR)
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # after the init
        writer.add_graph(sess.graph)

        #-------------------------------------------------------------------
        # SESS RUNS
        #-------------------------------------------------------------------
        lensc_ = gpf.do_assign(sess, lensc_assign, lensc, lensc_p, [0.1])
        [obs_noise_var_] = sess.run([obs_noise_var])
        lls = gpf.tf_optimize_model_params(sess, NUM_ITERS, train_op,
                                           log_likelihood, summ, writer, saver,
                                           LOGDIR, LOGCHECKPT,
                                           obs_train_dataset=obs_values,
                                           obs_value_placeholder=values_placeholder)  # pred_idx_pts, 2D (2500, 2),  1D (50,)
        [amp, lensc, obs_noise_var] = sess.run([amp, lensc, obs_noise_var])

        samples = sess.run(gprm.sample(NUM_SAMPLES))  #, feed_dict={values_placeholder: pred_idx_pts})

        assert samples.shape[0] == NUM_SAMPLES
        # assert samples.shape[1] == LENGTHSCALE_INIT.shape[0]
        # assert samples.shape[2] == PRED_FRACTION ** 2
        [batch, pred_idx_cnt, features] = pred_idx_pts.shape
        assert samples.shape[1] == batch
        assert samples.shape[2] == pred_idx_cnt

        H = gpf.calc_H(XEDGES, YEDGES, lensc, lensc_assign, lensc_p, amp, amp_assign, amp_p,
                       log_likelihood, sess,
                       values_placeholder,
                       obs_train_dataset=obs_values)
        saver.save(sess, im.os.path.join(LOGDIR, LOGCHECKPT), NUM_ITERS + 1)

        return lls, H, samples

    #===================================================================
    # PLOTS
    #===================================================================
    def plots(pred_idx_pts, obs_idx_pts, obs, samples, lls, H):
        plts.plot_loss_evolution(12, 4, lls)
        plts.plot_marginal_likelihood3D(XEDGES, H)
        plts.plot_samples2D(12, 5, 1, pred_idx_pts, obs_idx_pts, obs, samples, NUM_SAMPLES)
        if 2 <= len(obs_idx_pts.shape):
            plts.plot_samples3D(12, 12, 0, pred_idx_pts, obs_idx_pts, obs, samples, PRED_FRACTION, BEGIN, END)
            plts.plot_sin3D_rand_points(12, 12, COORDINATE_RANGE, obs_idx_pts, obs, PLOTRASTERPOINTS)

    #===================================================================
    # TEST SIN VGP FUNCTIONS
    #===================================================================
    def TEST_GP():
        # 1D
        pred_idx_pts__, obs_idx_pts__, obs__, values_placehldr = sin_1d_data()

        train_op__, gprm__, \
            amp__, amp_assign__, amp_p__, \
            lensc__, lensc_assign__, lensc_p__,\
            obs_noise_var__, log_likelihood__ \
            = graph(pred_idx_pts__, obs_idx_pts__[..., np.newaxis], obs__, values_placehldr)

        lls__, H__, samples__ \
            = sess_runs(train_op__, gprm__, amp__, amp_assign__, amp_p__,
                        lensc_assign__, lensc__, lensc_p__,
                        obs_noise_var__, pred_idx_pts__[np.newaxis, ..., np.newaxis],
                        values_placehldr, log_likelihood__,
                        obs_idx_pts__, obs__)

        plots(pred_idx_pts__, obs_idx_pts__, obs__, samples__, lls__, H__)

    def TEST_GP2():
        # 2D
        pred_idx_pts_, obs_idx_pts_, obs_, values_placehldr_2d = sin_2d_data()

        train_op_, gprm_, \
            amp_, amp_assign_, amp_p_, \
            lensc_, lensc_assign_, lensc_p_,\
            obs_noise_var_, log_likelihood_ \
            = graph(pred_idx_pts_, obs_idx_pts_, obs_, values_placehldr_2d)

        lls_, H_, samples_ \
            = sess_runs(train_op_, gprm_, amp_, amp_assign_, amp_p_,
                        lensc_assign_, lensc_, lensc_p_,
                        obs_noise_var_, pred_idx_pts_,
                        values_placehldr_2d, log_likelihood_,
                        obs_idx_pts_, obs_)

        plots(pred_idx_pts_, obs_idx_pts_, obs_, samples_, lls_, H_)
        pass


    def TEST_VGP():

        # GP -------------------------------
        # pred_idx_pts_,  # (2500,2)
        # obs_idx_pts,  # (90,2)
        # obs, (90,)
        # NUM_SAMPLES = 20
        # sample, (20, 2, 2500)

        # VGP -------------------------------
        # x_train, (100, 1)     x_train_batch, (64,1)  ->  (100,2)
        # y_train, (100,)       y_train_batch, (64,)   ->  (100,)
        # variational_loc, (10,)
        # inducing_index_points, (10,1)
        # index_pts,  (500, 1)
        # mean,  (500,)
        # NUM_GP_SAMPLES = 20
        # sample, (20, 500)

        # NUM_PREDICTIVE_IDX_PTS = 500
        # NUM_INDUCING_POINTS = 10
        # NUM_TRAIN_ITERS = 10
        # NUM_TRAIN_PTS = 100





        NUM_INDUCING_PTS_ = 20
        NUM_PREDICTIVE_IDX_PTS_ = 500
        TRAINING_MINIBATCH_SIZE_ = 64
        NUM_TRAIN_PTS_ = 100
        NUM_TRAIN_ITERS_ = 10
        NUM_TRAIN_LOGS_ = 10
        NUM_GP_SAMPLES_ = 100

        pred_idx_pts_, obs_idx_pts_, obs_, values_placehldr1d = sin_1d_data()

        train_op_, gprm_, \
        amp_, amp_assign_, amp_p_, \
        lensc_, lensc_assign_, lensc_p_, \
        obs_noise_var_, log_likelihood_ \
            = graph(pred_idx_pts_, obs_idx_pts_[..., np.newaxis], obs_, values_placehldr1d)

        lls_, H_, samples_ \
            = sess_runs(train_op_, gprm_, amp_, amp_assign_, amp_p_,
                        lensc_assign_, lensc_, lensc_p_,
                        obs_noise_var_, pred_idx_pts_[np.newaxis, ..., np.newaxis],
                        values_placehldr1d, log_likelihood_,
                        obs_idx_pts=obs_idx_pts_,
                        obs_values=obs_)

        plots(pred_idx_pts_, obs_idx_pts_, obs_, samples_, lls_, H_)

        x_train_, y_train_, f_ = vgpe.data(NUM_TRAIN_PTS_)

        train_op_, loss_, x_train_batch_, y_train_batch_, vgp_, \
            inducing_index_points_, variational_loc_, index_points_ \
            = vgpe.graph(x_train_, y_train_,
                    NUM_INDUCING_PTS_, NUM_PREDICTIVE_IDX_PTS_,
                    TRAINING_MINIBATCH_SIZE_, NUM_TRAIN_PTS_)

        train_op_, loss_, x_train_batch_, y_train_batch_, vgp_, \
            samples_, mean_, inducing_index_points_, variational_loc_ \
            = vgpe.sess_runs(x_train_, y_train_, train_op_, loss_,
                        x_train_batch_, y_train_batch_, vgp_,
                        inducing_index_points_, variational_loc_,
                        NUM_TRAIN_ITERS_, NUM_TRAIN_PTS_, TRAINING_MINIBATCH_SIZE_,
                        NUM_TRAIN_LOGS_, NUM_GP_SAMPLES_)

        vgpe.prints(f_, x_train_, y_train_,
               inducing_index_points_, variational_loc_,
               samples_, mean_, index_points_,
               NUM_GP_SAMPLES_)

    #===================================================================
    # TEST CALLS
    #===================================================================
    # tf.reset_default_graph()
    # TEST_GP()  # the placeholder named 'input_values' is clashing when both TEST_GP() and TEST_GP2() is called : reset the graph ..
    # tf.reset_default_graph()
    # TEST_GP2()
    tf.reset_default_graph()
    TEST_VGP()

if __name__ == '__main__':
    RUN = True

    if (RUN):


        # TEST_section_cut_GP()  # TODO fix
        # TEST_tracer_data_GP()

        # TEST_cov_while_loop()  # TODO fix

        TEST_algorithm1()
        TEST_algorithm2()
    else:
        TEST_section_cut()
        TEST_VAE()
        TEST_tf_map_fn_nd_reduce_any()
        TEST_snippets_tf_where()
        TEST_sinusoid_GP_VGP()

    pass
