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
tfkb = tf.keras.backend
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
sys.path.insert(0, '..')
sys.path.insert(0, 'main_tests')

import os
import placement_algorithm2 as alg2
import plots as plts
import gp_functions as gpf
import snippets_a2 as snps2
import gp_demo.vgp_optvpost_data2d as d2
import gp_demo.VariationalGaussianProcessTracerDataset as vgptd
import main_tests.alg2_split10x10x10_50x50x10_tests as a2t
PRINTS = True
###################################################################
# GRAPH PRINT
###################################################################
def tf_print2(op, tensors, message=None):

    def print_message2(*args):
        str_ = message
        if PRINTS:
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
def graph_GP(t_norm_,
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
            gp = tfd.GaussianProcess(
                kernel=kernel,  # ([2,],[2,])
                index_points=e_xyztp_s,
                observation_noise_variance=obs_noise_var,
                validate_args=True)

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
def graph_cov(COVER_spatial, COVER_temp_pressure, SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z,
              # Dset_bkt_TS, COVER_bkt_time,
              num_training_points_, num_predictive_index_points_one_dir, num_predictive_index_points_,
              num_inducing_points_, batch_size, feature_dims, dataset_shape, dataset_pl,
              cell_timestep_num, boolcreate_cov_2, BETA_val=4
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
    # with tf.name_scope("VGP"):


    # Copy here a VGP only to let saver restore its parameters
    if True:
        dtype = np.float64
        CSV_GP = 'main_datasets_/Dset_xyz_ave_small.csv'
        df_ = pd.read_csv(CSV_GP, encoding='utf-8', engine='c')
        df_.drop(columns=['Unnamed: 0'], inplace=True)

        # but the VGP is trained on normalized data
        # COLUMNS_MEAN, COLUMNS_STDEV, tracer_MEAN, tracer_STDEV = vgptd.get_normalizing_parameters()
        # df_.loc[:, 'Points:0'] -= COLUMNS_MEAN[0]
        # df_.loc[:, 'Points:0'] /= COLUMNS_STDEV[0]
        # df_.loc[:, 'Points:1'] -= COLUMNS_MEAN[1]
        # df_.loc[:, 'Points:1'] /= COLUMNS_STDEV[1]
        # df_.loc[:, 'Points:2'] -= COLUMNS_MEAN[2]
        # df_.loc[:, 'Points:2'] /= COLUMNS_STDEV[2]
        # df_.loc[:, 'Temperature_ave'] -= COLUMNS_MEAN[3]
        # df_.loc[:, 'Temperature_ave'] /= COLUMNS_STDEV[3]
        # df_.loc[:, 'Pressure_ave'] -= COLUMNS_MEAN[4]

        # MIN_x = df_['Points:0'].min()
        # MIN_y = df_['Points:1'].min()
        # MIN_z = df_['Points:2'].min()/5 # 0.0       # because I2 = 1
        # MIN_t = df_['Temperature_ave'].min()
        # MIN_p = df_['Pressure_ave'].min()
        #
        # MAX_x = df_['Points:0'].max()
        # MAX_y = df_['Points:1'].max()
        # MAX_z = df_['Points:2'].max()*4/5  # 0.1
        # MAX_t = df_['Temperature_ave'].max()
        # MAX_p = df_['Pressure_ave'].max()

        #
        # We don't use these other than letting the vgp parameters to load
        #  because it is quite frightening to see these code around,
        # TODO: initialize vith zeros!
        #

        # coord_range = [[-80., 0.], [-80., 0.]]  # [[-2., 2.], [-4., 4.]]  # [[-10., 10.], [-7., 7.]]
        coord_range = [[-2., 2.], [-2., 2.], [-2., 2.], [-2., 2.], [-2., 2.]]
        print('coord_range=', coord_range)
        # .. so the VGP is trained on normalized data
        # COLUMNS_MEAN, COLUMNS_STDEV, tracer_MEAN, tracer_STDEV = vgptd.get_normalizing_parameters()

        pred_idx_pts = d2.generate_5d_idx(num_predictive_index_points_, coord_range=coord_range)
        obs_idx_pts = df_.loc[:, ['Points:0', 'Points:1', 'Points:2', 'Temperature_ave', 'Pressure_ave']].to_numpy()[np.newaxis, ...]
        obs = df_.loc[:, 'Tracer_ave'].to_numpy()[np.newaxis, ...]

        inducing_index_points_init = d2.generate_5d_idx(num_inducing_points_, coord_range=coord_range)

        # Create kernel with trainable parameters, and trainable observation noise
        # variance variable. Each of these is constrained to be positive.
        amplitude_var = tf.Variable(.54, dtype=dtype, name='amplitude', use_resource=True)
        amplitude = (tf.nn.softplus(amplitude_var))
        length_scale_var = tf.Variable(.54, dtype=dtype, name='length_scale', use_resource=True)
        length_scale = (1e-5 + tf.nn.softplus(length_scale_var))
        kernel = tfkern.MaternFiveHalves(amplitude=amplitude, length_scale=length_scale)
        observation_noise_variance_var = tf.Variable(.54, dtype=dtype, name='observation_noise_variance', use_resource=True)
        observation_noise_variance = tf.nn.softplus(observation_noise_variance_var)

        # Create trainable inducing point locations and variational parameters.
        # num_inducing_points_ = 50

        inducing_index_points = tf.Variable(
            inducing_index_points_init,
            dtype=dtype, name='inducing_index_points', use_resource=True)

        # this is using the whole train_data, not just a batch
        variational_loc, variational_scale = (  # variational_loc=mean, variational_scale=sigma
            tfd.VariationalGaussianProcess.optimal_variational_posterior(
                kernel=kernel,
                inducing_index_points=inducing_index_points,
                observation_index_points=obs_idx_pts,
                observations=obs,
                observation_noise_variance=observation_noise_variance))

        # # These are the index point locations over which we'll construct the
        # # (approximate) posterior predictive distribution.
        # # num_predictive_index_points_ = 500
        # index_points_ = np.linspace(-13, 13,
        #                             num_predictive_index_points_one_dir,
        #                             dtype=dtype)[..., np.newaxis]

        # Construct our variational GP Distribution instance.
        vgp = tfd.VariationalGaussianProcess(
            kernel,
            index_points=pred_idx_pts,
            inducing_index_points=inducing_index_points,
            variational_inducing_observations_loc=variational_loc,
            variational_inducing_observations_scale=variational_scale,
            mean_fn=None,
            observation_noise_variance=observation_noise_variance,
            predictive_noise_variance=0.,
            jitter=1e-6, # 1e-6
            validate_args=False
        )


        # For training, we use some simplistic numpy-based minibatching.

        x_train_batch = tf.placeholder(dtype, [batch_size, feature_dims], name='x_train_batch')
        y_train_batch = tf.placeholder(dtype, [batch_size], name='y_train_batch')

        # Create the loss function we want to optimize.
        loss = vgp.variational_loss(
            observations=y_train_batch,
            observation_index_points=x_train_batch,
            kl_weight=float(batch_size) / float(num_training_points_))

        optimizer = tf.train.AdamOptimizer(learning_rate=.01)
        train_op = optimizer.minimize(loss)

        variational_inducing_observations_loc_saved = tf.Variable(
            np.zeros([1, num_inducing_points_], dtype=dtype)
            + np.random.normal(0, 0.01, [num_inducing_points_]),
            name='variational_inducing_observations_loc_saved')

        variational_inducing_observations_scale_saved = tf.Variable(
            np.reshape(np.eye(num_inducing_points_, dtype=dtype), [1, num_inducing_points_, num_inducing_points_]),
            name='variational_inducing_observations_scale_saved')

        # saver_path = './gp_demo/vgp_tests/saved_variables'
        saver_spec = {  'variational_inducing_observations_loc_saved': variational_inducing_observations_loc_saved,
                        'variational_inducing_observations_scale_saved': variational_inducing_observations_scale_saved,
                        'amplitude': amplitude_var,
                        'inducing_index_points': inducing_index_points,
                        'length_scale': length_scale_var,
                        'observation_noise_variance': observation_noise_variance_var
                     }
        saver_VGP = tf.train.Saver(saver_spec)  # all the graph so far






    #===================================================================
    # COVARIANCE INIT
    #===================================================================
    print("CREATING COV -----")
    ts0 = tf.timestamp()

    with tf.name_scope("covariance"):
        with tf.name_scope("calc_cov_ij"):

            # ===================================================================
            # I. FUNCTIONS FOR WHILE LOOP
            # ===================================================================

            # REQ: from inside_node_calc_ij() cov_vv, last_cov_vv, xyz_cov_idxs must be seen 'implicitly',
            # thus stay being a variable, scatter_nd_update is possible

            with tf.name_scope("cov_init"):
                # ===================================================================
                # I. GLOBAL FOR WHILE LOOP
                # ===================================================================
                cover_pt_const = tf.constant(COVER_temp_pressure)  # how many

                linsp_x = tf.linspace(-2., 2., COVER_spatial[0], name="linsp_x")
                linsp_y = tf.linspace(-2., 2., COVER_spatial[1], name="linsp_y")
                linsp_z = tf.linspace(-2., 2., COVER_spatial[2], name="linsp_z")
                linsp_x = tf.cast(linsp_x ,dtype=tf.float64)
                linsp_y = tf.cast(linsp_y, dtype=tf.float64)
                linsp_z = tf.cast(linsp_z, dtype=tf.float64)
                linsp_x = tf.reshape(linsp_x, [-1], name="reshape_")
                linsp_y = tf.reshape(linsp_y, [-1], name="reshape_")
                linsp_z = tf.reshape(linsp_z, [-1], name="reshape_")


                I0 = tf.constant(COVER_spatial[0], name="I0")
                I1 = tf.constant(COVER_spatial[1], name="I1")
                I2 = tf.constant(COVER_spatial[2], name="I2")

                # linsp_t = tf.linspace(-3., 3., cover_pt_const, name="linsp_t")
                # linsp_p = tf.linspace(-3., 3., cover_pt_const, name="linsp_p")
                # T_, P_ = tf.meshgrid(linsp_t, linsp_p)
                # stack_tp = tf.stack([T_, P_], axis=2, name="grid_tp")
                # vec_tp = tf.reshape(stack_tp, [-1, 2], name="vec_tp")

                # ===================================================================
                # I. WORK ON THESE IN WHILE LOOP
                # ===================================================================
                cov_vv = tf.Variable(tf.zeros((I0 * I1 * I2, I0 * I1 * I2), dtype=tf.float64),
                                     dtype = tf.float64,
                                     name="cov_vv")
                last_cov_vv = tf.Variable(3.14,
                                          name="last_cov_vv",
                                          dtype=tf.float64)
                xyz_cov_idxs = tf.Variable(tf.zeros((I0 * I1 * I2, 3), dtype=tf.int32),
                                           dtype = tf.int32,
                                           name="xyz_cov_idxs")

            # ===================================================================
            # II. WHILE LOOP COV I J
            # ===================================================================

            # REQ: from inside_node_calc_ij() cov_vv, last_cov_vv, xyz_cov_idxs must be seen 'implicitly',
            # thus stay being a variable, scatter_nd_update is possible
            # -> 'cov_init' comes before (inside_node_calc_ij)
            #####################################################################################
            def local_kernel_filter(j_0, j_1, j_2, i_0, i_1, i_2, i, j):
                if boolcreate_cov_2  == True:
                    cov_vv_ij_, cov_vv_i_j_, ti, tj = inside_node_calc_ij(j_0, j_1, j_2, i_0, i_1, i_2, i, j)
                    cov_vv_ij, cov_vv_i_j = inside_node_store_ij(j_0, j_1, j_2, i_0, i_1, i_2, i, j, cov_vv_ij_, ti, tj)
                    return cov_vv_ij, cov_vv_i_j

                else: # boolcreate_cov_2 == False:
                    beta = tf.constant(BETA_val, dtype=tf.float64, name="beta")  # or any other beta_small > 0
                    pi = tf.constant(np.pi, dtype=tf.float64, name="pi")
                    i_0 = tf.cast(i_0, tf.float64)
                    i_1 = tf.cast(i_1, tf.float64)
                    i_2 = tf.cast(i_2, tf.float64)
                    j_0 = tf.cast(j_0, tf.float64)
                    j_1 = tf.cast(j_1, tf.float64)
                    j_2 = tf.cast(j_2, tf.float64)
                    # delta_ij = tf.multiply(beta, tf.math.abs(tf.sqrt(
                    #                                                 tf.math.squared_difference(i_0, j_0) +
                    #                                                 tf.math.squared_difference(i_1, j_1) +
                    #                                                 tf.math.squared_difference(i_2, j_2))))  # distance in space not index?

                    delta_ij = tf.math.abs(tf.sqrt(
                        tf.math.squared_difference(i_0, j_0) +
                        tf.math.squared_difference(i_1, j_1) +
                        tf.math.squared_difference(i_2, j_2)))  # distance in space not index?

                    # def decay_fn_wrong(c_ij):
                    #     t_1 = tf.subtract(2*pi, delta_ij)
                    #     t_2 = 1 + tf.math.cos(delta_ij)/2
                    #     nominator = tf.multiply(t_1, t_2) + (3/2)*tf.math.sin(delta_ij)
                    #     return tf.abs(c_ij * nominator / 3*pi)

                    def decay_fn_(delta_ij_):
                        a = tf.cond(tf.less(delta_ij_,1.42), lambda: tf.cast(1.,tf.float64), lambda : tf.cast(0.,tf.float64))
                        return a

                    def decay_fn(delta_ij_):
                        decay = tf.math.exp(-tf.square(tf.multiply(beta, delta_ij_)) / (2 * pi))
                        a = tf.cond(tf.less(decay, 0.01), lambda: tf.cast(0.,tf.float64), lambda : tf.cast(decay, tf.float64))
                        return a

                    def calc_ij():
                        decay_val = decay_fn(delta_ij)

                        c_ij_, c_i_j_, ti, tj = inside_node_calc_ij(j_0, j_1, j_2, i_0, i_1, i_2, i, j)

                        c_ij_filt = decay_val * c_ij_
                        c_i_j_filt = decay_val * c_i_j_

                        c_ij, c_i_j = inside_node_store_ij(j_0, j_1, j_2, i_0, i_1, i_2, i, j, c_ij_filt, ti, tj)


                        print_cij = tf.print(["c_ij : ", c_ij, c_i_j], output_stream=sys.stdout, name='print_c_ij')
                        with tf.control_dependencies([print_cij]):
                            return c_ij, c_i_j

                    def zero_ij():
                        zero = tf.constant(0., tf.float64)
                        print_z = tf.print(["zero"], output_stream=sys.stdout, name='print_z')
                        with tf.control_dependencies([print_z]):
                            return zero, zero

                    decay_val_ = decay_fn(delta_ij)
                    cov_vv_ij, cov_vv_i_j = tf.cond(tf.less(decay_val_, 0.01),
                                                    false_fn=zero_ij,
                                                    true_fn= calc_ij
                                                   )
                    return cov_vv_ij, cov_vv_i_j

            def inside_node_calc_ij(j_0, j_1, j_2, i_0, i_1, i_2, i, j):

                aj_0 = tf.Assert(tf.less(tf.cast(j_0, tf.int32), I0), [j_0])
                aj_1 = tf.Assert(tf.less(tf.cast(j_1, tf.int32), I1), [j_1])
                aj_2 = tf.Assert(tf.less(tf.cast(j_2, tf.int32), I2), [j_2, I2])
                ai_0 = tf.Assert(tf.less(tf.cast(i_0, tf.int32), I0), [i_0])
                ai_1 = tf.Assert(tf.less(tf.cast(i_1, tf.int32), I1), [i_1])
                ai_2 = tf.Assert(tf.less(tf.cast(i_2, tf.int32), I2), [i_2, I2])
                ai = tf.Assert(tf.less(i, I0*I1*I2), [i])
                aj = tf.Assert(tf.less(j, I0*I1*I2), [j])

                # 1. collect xyz indices in while loop
                # 2. collect bucket_fraction_nums to create bkt indices -> bkt_x = (i_0/I0) // (1/SPLIT_bkt_x)
                # 3. vec_tp_i, vec_tp_j sampled across time from bkt in Dset_bkt_TS @ COVER_bkt_time density of values
                #    meaning the tracer values will be sampled across all time - > from a distribution across time ?

                def get_tracer_values(loc_xyz_i_, vec_tp_i_): # also j
                    # from i -> get tracers vector
                    # get XYZ index from indies,

                    vgp_xyztp = tf.cast(tf.expand_dims(gpf.graph_get_vgp_input_xyztp(loc_xyz_i_, vec_tp_i_), axis=0), dtype=tf.float64)

                    assert (len(vgp_xyztp.shape) == 3)
                    assert vgp_xyztp.shape[0] == 1
                    assert vgp_xyztp.shape[2] == 5

                    vgp2 = tfd.VariationalGaussianProcess(
                        kernel,
                        index_points= vgp_xyztp,  #pred_idx_pts2,
                        inducing_index_points=inducing_index_points,
                        variational_inducing_observations_loc=variational_inducing_observations_loc_saved,
                        variational_inducing_observations_scale=variational_inducing_observations_scale_saved,
                        mean_fn=None,
                        observation_noise_variance=observation_noise_variance,
                        predictive_noise_variance=0.,
                        jitter=1e-6,  # 1e-6
                        validate_args=False
                    )

                    tracers_loc_i_ = tf.reshape(vgp2.mean(), shape=[-1])

                    # def vgp_output(np_vgp_xyztp_, np_tracers_loc_i_,
                    #                np_inducing_index_points, np_variational_loc, np_variational_scale):
                    #     if PRINTS:
                    #         print(' ')
                    #         print('vgp_xyztp: ', np_vgp_xyztp_[0,:3,:])
                    #         print('tracers_loc_i: ', np_tracers_loc_i_[:3])
                    #
                    #         print("inducing_index_points", np_inducing_index_points[0, :3, :])
                    #         print("variational_loc_saved", np_variational_loc[0, :3])
                    #         print("variational_scale_saved", np_variational_scale[0, 0, :3])
                    #         print(' ')
                    #     return np_vgp_xyztp, np_tracers_loc_i, np_inducing_index_points, np_variational_loc, np_variational_scale
                    #
                    # vgp_xyztp_, tracers_loc_i_, np_inducing_index_points_, np_variational_loc_, np_variational_scale_ \
                    #     = tf.numpy_function(vgp_output, [vgp_xyztp, tracers_loc_i_, inducing_index_points,
                    #                                      variational_inducing_observations_loc_saved,
                    #                                      variational_inducing_observations_scale_saved],
                    #                         [vgp_xyztp.dtype, tracers_loc_i_.dtype, inducing_index_points.dtype,
                    #                          variational_inducing_observations_loc_saved.dtype,
                    #                          variational_inducing_observations_scale_saved.dtype])

                    with tf.control_dependencies([tracers_loc_i_]):
                        return tracers_loc_i_  # tf.ones([300, 1], dtype=tf.float64)* tf.cast(pi_i, dtype=tf.float64)  # tracers_loc_i

                with tf.control_dependencies([aj_0, aj_1, aj_2, ai_0, ai_1, ai_2, ai, aj]):
                    # get_tracer_values_ = tf.function(get_tracer_values)

                    loc_xyz_i = gpf.slice_grid_xyz(tf.cast(i_0, tf.int64), tf.cast(i_1, tf.int64), tf.cast(i_2, tf.int64), linsp_x, linsp_y, linsp_z)  # xyz values from linspace
                    loc_xyz_j = gpf.slice_grid_xyz(tf.cast(j_0, tf.int64), tf.cast(j_1, tf.int64), tf.cast(j_2, tf.int64), linsp_x, linsp_y, linsp_z)  # xyz values from linspace

                cell_id = get_cell_id(i_0, i_1, i_2, I0, I1, I2, SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z)
                vec_tp_i_uniform = sample_uniform()  # OPTION 1.- SAMPLE TP UNIFORM
                vec_tp_i_distribution = sample_cell(dataset_pl, cell_id, cell_timestep_num) # OPTION 2.- SAMPLE TIMERANGE TP HISTOGRAM

                tracers_loc_i = get_tracer_values(loc_xyz_i, vec_tp_i_distribution)
                tracers_loc_j = get_tracer_values(loc_xyz_j, vec_tp_i_distribution)

                pr_op_tr = tf.print(['tracers_loc_i, j=', tracers_loc_i, tracers_loc_j],
                                    output_stream=sys.stdout,
                                    name='print_tracers_loc_i_j')

                # def numpy_fun_tracer(tracers_loc_i, tracers_loc_j):
                #     if PRINTS or 1:
                #         print(' ')
                #         print('tracers_loc_i: ', tracers_loc_i[:5])
                #         print('tracers_loc_j: ', tracers_loc_j[:5])
                #         print(' ')
                #     return tracers_loc_i, tracers_loc_j
                #
                # [tracers_loc_i, tracers_loc_j] = tf.numpy_function(numpy_fun_tracer, [tracers_loc_i, tracers_loc_j], [tracers_loc_i.dtype, tracers_loc_j.dtype])

                with tf.control_dependencies([tracers_loc_i, tracers_loc_j]):

                    # tr_mean = tf.constant([0.0018159087825037148], dtype=tf.float64)
                    # tr_stdev = tf.constant([0.0007434639347162126*3000000], dtype=tf.float64)

                    tr_mean = tf.constant([0], dtype=tf.float64)
                    tr_stdev = tf.constant([1], dtype=tf.float64)

                    t_i_ = tracers_loc_i - tr_mean
                    t_j_ = tracers_loc_j - tr_mean

                    t_i = t_i_ / tr_stdev
                    t_j = t_j_ / tr_stdev
                    cov_vv_ij = tfp.stats.covariance(t_i, t_j, sample_axis=0, event_axis=None)

                    cov_vv_ij_pr = tf.print(['cov_vv_ij=', cov_vv_ij])

                    return cov_vv_ij, cov_vv[i, j], t_i, t_j
                    # def numpy_fun(ti, tj, cov_vvij):
                    #     if PRINTS:
                    #         print(' ')
                    #         print('ti: ', ti[:5])
                    #         print('tj: ', tj[:5])
                    #         print('cov_vvij: ', cov_vvij)
                    #         print(' ')
                    #     return ti, tj, cov_vvij
                    #
                    # [t_i, t_j, cov_vv_ij] = tf.numpy_function(numpy_fun, [t_i, t_j, cov_vv_ij], [t_i.dtype, t_j.dtype, cov_vv_ij.dtype])

            def inside_node_store_ij(j_0, j_1, j_2, i_0, i_1, i_2, i, j, cov_vv_ij, t_i, t_j):

                # with tf.control_dependencies([t_i, t_j, cov_vv_ij
                                              #   , cov_vv_ij_pr
                                              # ]):
                update = tf.reshape(cov_vv_ij, [], name="update")

                indexes1 = tf.cast([i, j], dtype=tf.int32, name='indexes')
                indexes2 = tf.cast([j, i], dtype=tf.int32, name='indexes')

                op1 = tf.scatter_nd_update(cov_vv, [indexes1], [update])  # [[i0, 0]], [3.+i0+0])
                op2 = tf.scatter_nd_update(cov_vv, [indexes2], [update])  # [[i0, 0]], [3.+i0+0])
                op3 = tf.assign(last_cov_vv, update)  # test assignment is working

                pr_op = op3
                # pr_op = tf.print(['inside', i, j, cov_vv[i, j], update],
                #                  output_stream=sys.stdout,
                #                  name='print_inside_cov_vv_i_j_update')

                with tf.control_dependencies([op1, op2, op3, pr_op]):
                    # update a row of x y z
                    indexes_x = tf.cast([j, 0], dtype=tf.int32,name='indexes_x')
                    indexes_y = tf.cast([j, 1], dtype=tf.int32,name='indexes_y')
                    indexes_z = tf.cast([j, 2], dtype=tf.int32,name='indexes_z')

                    # get samples
                    # def calc_uniform_index():
                    update_idx_x = tf.reshape(j_0, [], name="update_x")
                    update_idx_y = tf.reshape(j_1, [], name="update_y")
                    update_idx_z = tf.reshape(j_2, [], name="update_z")
                    # return update_idx_x_, update_idx_y_, update_idx_z_

                    # update_idx_x, update_idx_y, update_idx_z = calc_bucket_index(j_0, j_1, j_2, i)

                    # xyz_cov_idxs - SAVE xyz to be remapped to after algorithm 2
                    op4 = tf.scatter_nd_update(xyz_cov_idxs, [indexes_x], [update_idx_x])
                    op5 = tf.scatter_nd_update(xyz_cov_idxs, [indexes_y], [update_idx_y])
                    op6 = tf.scatter_nd_update(xyz_cov_idxs, [indexes_z], [update_idx_z])

                    with tf.control_dependencies([op1, op2, op3, op4, op5, op6, cov_vv_ij]):
                        return cov_vv_ij, cov_vv[i, j]  # END inside_node_store_ij


            def get_cell_id(i_0, i_1, i_2, I0, I1, I2, SPLIT_cell_x, SPLIT_cell_y,
                               SPLIT_cell_z):  # j_0, j_1, j_2, i):
                # calculate bucket number and return it.
                i_0_ = tf.cast(i_0, tf.float64)
                i_1_ = tf.cast(i_1, tf.float64)
                i_2_ = tf.cast(i_2, tf.float64)
                I0_ = tf.cast(I0, tf.float64)
                I1_ = tf.cast(I1, tf.float64)
                I2_ = tf.cast(I2, tf.float64)

                cell_x_i = i_0_ / I0_ // (1 / SPLIT_cell_x)  # tf.floordiv(tf.math.divide(i_0, I0), tf.math.divide(1, SPLIT_bkt_x))
                cell_y_i = i_1_ / I1_ // (1 / SPLIT_cell_y)
                cell_z_i = i_2_ / I2_ // (1 / SPLIT_cell_z)
                cell_id = cell_x_i + (cell_y_i * SPLIT_cell_x) + (cell_z_i * SPLIT_cell_x * SPLIT_cell_y)

                return cell_id

            def sample_cell(dataset_pl, cell_id, cell_timestep_num):
                # takes a constant to make shapes of placeholders: timesteps_in_cell, cell_id and dataset

                # Get all timesteps from cell: cell_id
                cell_rows_selection = tf.slice(dataset_pl, begin=[tf.cast(cell_id * cell_timestep_num, tf.int64), 0],
                                          size=[cell_timestep_num, dataset_pl.shape[1]],
                                          name="cell_ts_pl")

                op1 = tf.Assert(tf.equal(cell_rows_selection[0, 0], cell_id),
                                [cell_rows_selection[0, :], cell_timestep_num, cell_id])
                op2 = tf.Assert(tf.equal(cell_rows_selection[-1, 0], cell_id),
                                [cell_rows_selection[0, :], cell_timestep_num, cell_id])

                # Get temp_ave and pressure_ave columns from cell_selection
                cell_selection = cell_rows_selection[ : ,2:4]
                with tf.control_dependencies([op1, op2]):
                    return cell_selection

            def sample_uniform():
                linsp_t = tf.linspace(-3., 3., cover_pt_const, name="linsp_t")
                linsp_p = tf.linspace(-3., 3., cover_pt_const, name="linsp_p")
                T_, P_ = tf.meshgrid(linsp_t, linsp_p)
                stack_tp = tf.stack([T_, P_], axis=2, name="grid_tp")
                vec_tp = tf.reshape(stack_tp, [-1, 2], name="vec_tp")
                vec_tp_sample = vec_tp
                return vec_tp_sample


        with tf.name_scope("cov_calc"):

            cond0 = lambda i_0, I0_: tf.less(i_0, I0_)
            def body0(i_0, I0_):
                cond1 = lambda i_1, I1_: tf.less(i_1, I1_)

                def body1(i_1, I1_):
                    cond2 = lambda i_2, I2_: tf.less(i_2, I2_)

                    def body2(i_2, I2_):
                        cond_0 = lambda j_0, J0_: tf.less(j_0, J0_)

                        # we'd like to know the index=i of a position (i_0, i_1, i_2)
                        # I can't remember why did we go inside out, TODO: write it here
                        i_strde0 = I2_ * I1_
                        i_strde1 = I2_
                        i_strde2 = 1

                        # if i_0 = 0..2, i1=0..2, i_2=0, i_strde0 or i_strde1 have to be 1 (and i_strde1 = I2_ == 1)
                        i_mult0 = tf.multiply(i_0, i_strde0)
                        i_mult1 = tf.multiply(i_1, i_strde1)
                        i_mult2 = tf.multiply(i_2, i_strde2)
                        i_add = tf.add(i_mult2, i_mult1)
                        i_ = tf.add(i_mult0, i_add)
                        ai_ = tf.Assert(tf.less(i_, I0 * I1 * I2), [i_, i_strde0, i_strde1, i_strde2,
                                                                    i_mult0, i_mult1, i_mult2,
                                                                    I0_, I1_, I2_])
                        # for long running cov_vv generation, let's see the progress
                        pr_i = tf.print(['i =', i_, tf.timestamp()-ts0],
                                        output_stream=sys.stdout,
                                        name='print_tracers_loc_i_j')

                        # it is repeated below the functions: (this may help to see the progress on a bigger cov_vv)
                        with tf.control_dependencies([i_,
                                                      # pr_i,
                                                      ai_]):  # does not work here: control_dependencies: pr_i
                            i = i_

                        def body_0(j_0, J0_):
                            cond_1 = lambda j_1, J1_: tf.less(j_1, J1_)

                            def body_1(j_1, J1_):
                                def cond_2(j_2, J2_):
                                    j_strde0 = J2_ * J1_
                                    j_strde1 = J2_
                                    j_strde2 = 1

                                    j_mult0 = tf.multiply(j_0, j_strde0)
                                    j_mult1 = tf.multiply(j_1, j_strde1)
                                    j_mult2 = tf.multiply(j_2, j_strde2)
                                    j_add = tf.add(j_mult2, j_mult1)
                                    j_ = tf.add(j_mult0, j_add)
                                    aj_ = tf.Assert(tf.less(j_, I0 * I1 * I2), [j_, j_strde0, j_strde1, j_strde2,
                                                                                j_mult0, j_mult1, j_mult2,
                                                                                J0_, J1_, J2_])
                                    with tf.control_dependencies([aj_]):
                                        j = j_

                                    return tf.math.logical_and(tf.less(j_2, J2_), tf.less_equal(j, i))

                                def body_2(j_2, J2_):
                                    # we'd like to know the index=j of a position (j_0, j_1, j_2)
                                    j_strde0 = J2_ * J1_
                                    j_strde1 = J2_
                                    j_strde2 = 1

                                    j_mult0 = tf.multiply(j_0, j_strde0)
                                    j_mult1 = tf.multiply(j_1, j_strde1)
                                    j_mult2 = tf.multiply(j_2, j_strde2)
                                    j_add = tf.add(j_mult2, j_mult1)
                                    j_ = tf.add(j_mult0, j_add)
                                    a2j_ = tf.Assert(tf.less(j_, I0 * I1 * I2), [j_, j_strde0, j_strde1, j_strde2,
                                                                                j_mult0, j_mult1, j_mult2,
                                                                                J0_, J1_, J2_])
                                    with tf.control_dependencies([a2j_]):
                                        j = j_

                                    # j = tf_print2(j, [i, j], message='i, j = ')

                                    #####################################################################################
                                    cov_vv_ij, cov_vv_i_j = local_kernel_filter(j_0, j_1, j_2, i_0, i_1, i_2, i, j)
                                    #####################################################################################

                                    print_op = cov_vv_ij
                                    #
                                    # For some reason I don't know, cov_vv[i, j] seems to be 0 here, even if
                                    #  tf.print shows the updates inside ..
                                    #

                                    # print_op = tf.print(['loop: ', i, j, cov_vv[i, j], cov_vv_ij, cov_vv_i_j],
                                    #                     output_stream=sys.stdout,
                                    #                     name='print_cov_vv_i_j_cov_vv_ij_2x')

                                    with tf.control_dependencies([cov_vv_ij, cov_vv_i_j, print_op]):
                                        j_2_next = tf.add(j_2, 1)
                                    return [j_2_next, J2_]

                                [loop_j2, _] = tf.while_loop(cond_2, body_2, loop_vars=[0, I2], name="loop_j2")
                                with tf.control_dependencies([loop_j2]):
                                    j_1_next = tf.add(j_1, 1)
                                return [j_1_next, J1_]  # end body_1

                            [loop_j1, _] = tf.while_loop(cond_1, body_1, loop_vars=[0, I1], name="loop_j1")
                            with tf.control_dependencies([loop_j1]):
                                j_0_next = tf.add(j_0, 1)
                            return [j_0_next, J0_]  # end body_0

                        with tf.control_dependencies([i_,
                                                      # pr_i,
                                                      ai_]):
                            [loop_j0, _] = tf.while_loop(cond_0, body_0, loop_vars=[0, I0], name="loop_j0")

                        with tf.control_dependencies([loop_j0]):
                            i_2_next = tf.add(i_2, 1)
                        return [i_2_next, I2_]  # end body2

                    [loop_i2, _] = tf.while_loop(cond2, body2, loop_vars=[0, I2], name="loop_i2")
                    with tf.control_dependencies([loop_i2]):
                        i_1_next = tf.add(i_1, 1)
                    return [i_1_next, I1_]  # end body1

                [loop_i1, _] = tf.while_loop(cond1, body1, loop_vars=[0, I1], name="loop_i1")
                with tf.control_dependencies([loop_i1]):
                    i_0_next = tf.add(i_0, 1)
                return [i_0_next, I0_]  # end body0

            [while_i0_idx, while_i0_end] = tf.while_loop(cond0, body0, loop_vars=[0, I0], name="loop_i0")

            with tf.control_dependencies([while_i0_idx, while_i0_end]):
                return cov_vv, last_cov_vv, while_i0_idx, while_i0_end, xyz_cov_idxs, saver_VGP, \
                       variational_inducing_observations_loc_saved, \
                       variational_inducing_observations_scale_saved, \
                       variational_loc, \
                       variational_scale

                print('loc_set_op_: ', loc_set_op_[0, :5])
                print("variational_loc_v", variational_loc_v[0, :5])

                print('scale_set_op_: ', scale_set_op_[0, :1, :5])
                print("variational_scale_v", variational_scale_v[0, :1, :5])


###################################################################
# TEST
###################################################################
def TEST_cov_buckets(COVER_spatial, boolcreate_cov_2, CSV_CELL,
                     SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z,
                     SAVE_CACHE, SAVE_SELECTION, BETA_val=4.):
    # Files #
    LOGDIR = "./log_dir_/"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')+"_vgp"
    # CSV_COV = 'main_datasets_/Dset_bkt_TS_small_600.csv'
    # df_COV = pd.read_csv(CSV_COV, encoding='utf-8', engine='c')
    # df_COV.drop(columns=['Unnamed: 0'], inplace=True)

    # CSV_GP = 'main_datasets_/Dset_xyz_ave_small.csv'
    # df_GP = pd.read_csv(CSV_GP, encoding='utf-8', engine='c')
    # df_GP.drop(columns=['Unnamed: 0'], inplace=True)

    #
    # here we select a dataset that will represent the distribution of the pressure, temperature values
    #
    # SET_ID = 2
    # CSV_2_CELL = './dataset_2_cov/dataset_2_cov_' + str(SET_ID) + '.csv'
    df_cell = pd.read_csv(CSV_CELL, encoding='utf-8', engine='c')
    df_cell.drop(columns=['Unnamed: 0'], inplace=True)

    # normalize Temperature_ave, Pressure_ave, so the input is scaled to be used by the VGP
    eTempCol, ePressCol = 3, 4
    COLUMNS_MEAN, COLUMNS_STDEV, tracer_MEAN, tracer_STDEV = vgptd.get_normalizing_parameters()
    # print(df_cell.loc[0, 'Temperature_ave'])
    df_cell.loc[:, 'Temperature_ave'] -= COLUMNS_MEAN[eTempCol]
    # print(df_cell.loc[0, 'Temperature_ave'])
    df_cell.loc[:, 'Temperature_ave'] /= COLUMNS_STDEV[eTempCol]
    # print(df_cell.loc[0, 'Temperature_ave'])
    df_cell.loc[:, 'Pressure_ave'] -= COLUMNS_MEAN[ePressCol]
    df_cell.loc[:, 'Pressure_ave'] /= COLUMNS_STDEV[ePressCol]
    # now the input is scaled to be used by the VGP

    def fill_missing_cell_df(df_cell):
        MAXCELLS = 52

        df_corrected = df_cell.loc[df_cell['Bucket'] == 0, :]
        assert len(df_corrected) == 250
        prev_filter = None

        for i in range(1, MAXCELLS):
            df_cell_filter = df_cell.loc[df_cell['Bucket'] == i, :]
            if len(df_cell_filter) == 250:
                df_corrected = df_corrected.append(df_cell_filter)
                prev_filter = df_cell_filter

            else:  # only works if there is a Cell 0.
                # prev_filter.loc[:, prev_filter.columns.get_loc('Bucket')] = i
                to_append = prev_filter.copy()
                to_append.loc[:, 'Bucket'] = i
                df_corrected = df_corrected.append(to_append)

        df_corrected = df_corrected.reset_index(drop=True)
        return df_corrected

    df_cell = fill_missing_cell_df(df_cell)
    dataset_shape = df_cell.shape
    dataset_pl = tf.constant(df_cell.to_numpy(), tf.float64)
    cell_timestep_num = 250

    # Initializations #
    AMPLITUDE_INIT = np.array([.1, .1]) # [0.1, 0.1]
    LENGTHSCALE_INIT = np.array([.001, .001]) # [0.1, 0.1]
    INIT_OBSNOISEVAR = 0.001
    LEARNING_RATE = 0.1
    NUM_SAMPLES = 50
    K_SENSORS = 7


    # sample_uniform() used cover_pt_const==COVER_temp_pressure
    # we are using sample_cell(, , cell_timestep_num == 250) now (which maybe is more than enough ..)
    COVER_temp_pressure = 12  # Unused variable!

    # COVER_bkt_time = len(df_COV)
    saver_path = 'gp_demo/vgp/saved_variables_dataset_1_gp'  #'gp_demo/vgp/saved_variables'
    num_training_points_, num_predictive_index_points_one_dir, \
    num_predictive_index_points_,num_inducing_points_, \
    batch_size, feature_dims, saver_path_ \
        = vgptd.get_vgp_parameters()  # last VGP fitted
    saver_path = 'gp_demo/' + saver_path_  # like 'gp_demo/vgp/saved_variables'

    #===================================================================
    # II.1 - COV calc
    #===================================================================
    [cov_vv_r, last_cov_vv_r, while_i0_idx, while_i0_end, xyz_cov_idxs_r, saver_vgp,
     variational_inducing_observations_loc_saved, variational_inducing_observations_scale_saved, variational_loc, variational_scale] \
        = graph_cov(COVER_spatial, COVER_temp_pressure, SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z,
                    num_training_points_, num_predictive_index_points_one_dir, num_predictive_index_points_,
                    num_inducing_points_, batch_size, feature_dims, dataset_shape, dataset_pl, cell_timestep_num, boolcreate_cov_2)

    #===================================================================
    # II.2 - PLACEMENT
    #===================================================================
    # with tf.control_dependencies([cov_vv_r, while_i0_idx, while_i0_end]):
    [selection_idxs_A, _, delta_cached_iters_tensor, A_selection_and_delta] = snps2.sparse_placement_algorithm_2(cov_vv_r, K_SENSORS, COVER_spatial)

    #===================================================================
    # GRAPH SAVER
    #===================================================================
    print("LOGDIR", LOGDIR)
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    saver = tf.train.Saver()  # all the object in the graph mapped

    # ===================================================================
    # GRAPH CALLS
    # ===================================================================
    # --- done by now ---

    with tf.Session() as sess:
        tfkb.set_session(sess)  # keras backend joined
        sess.run(tf.global_variables_initializer())  # after the init

        writer.add_graph(sess.graph)
        saver_vgp.restore(sess, saver_path)

        [loc_set_op_, scale_set_op_] = sess.run([variational_inducing_observations_loc_saved,
                                                 variational_inducing_observations_scale_saved])
        [variational_loc_v, variational_scale_v] = sess.run([variational_loc,
                                                             variational_scale])

        print('loc_set_op_: ', loc_set_op_[0, :5])
        print("variational_loc_v", variational_loc_v[0, :5])

        print('scale_set_op_: ', scale_set_op_[0, :1, :5])
        print("variational_scale_v", variational_scale_v[0, :1, :5])

        #===================================================================
        # SESS
        #===================================================================

        [w0, w1, cov_vv_] = sess.run([while_i0_idx, while_i0_end, cov_vv_r])  #, feed_dict={values: df_COV})
        # elast = sess.run(last_cov_vv_r, feed_dict={values: df_COV})

        print("cov_vv[0, :w1], after running while_op")
        print('w0=', w0, 'w1=', w1)
        print(cov_vv_[:, :])
        _, file_cov_vv, file_cov_idxs, _, _, _, _ = a2t.get_spatial_splits(a2t.indirection)
        pd.DataFrame(cov_vv_).to_csv(file_cov_vv)

        xyz_cov_idxs_ = sess.run(xyz_cov_idxs_r)  #, feed_dict={values: df_COV})
        pd.DataFrame(xyz_cov_idxs_).to_csv(file_cov_idxs)

        # --- if I do the following, it will do the selection algo twice
        # selection_idxs_ = sess.run(selection_idxs_A)  # selection algo 1st
        # delta_cached_its_ = sess.run(delta_cached_iters_tensor)  # selection algo 2nd

        # do the same as above in one single pass
        [selection_idxs_, delta_cached_its_] = sess.run([selection_idxs_A, delta_cached_iters_tensor])  # selection algo once

        # xyz_selection = xyz_cov_idxs_[selection_idxs_]
        print(selection_idxs_)
        print(xyz_cov_idxs_)

        pd.DataFrame(delta_cached_its_).to_csv(SAVE_CACHE)
        pd.DataFrame(selection_idxs_).to_csv(SAVE_SELECTION)

        # fig_rows, fig_cols = 3, 4
        # fig, ax = plt.subplots(fig_rows, fig_cols,
        #                        figsize=(15, 15),
        #                        squeeze=True,  # use just one index: ax[i]
        #                        constrained_layout=True)
        return selection_idxs_
        # pass

def TEST_cov_2_equal_cov_3():
    #----------------------------------------------------------
    # dataframe export paths
    #----------------------------------------------------------
    SAVE_CACHE_alg2test = 'main_datasets_tests/placement_algorithm2_cache_test.csv'
    SAVE_SELECTION_alg2test = 'main_datasets_tests/placement_algorithm2_selection_idxs_test.csv'
    SAVE_CACHE_alg3test = 'main_datasets_tests/placement_algorithm3_cache_test.csv'
    SAVE_SELECTION_alg3test = 'main_datasets_tests/placement_algorithm3_selection_idxs_test.csv'

    SET_ID = 2
    CSV_CELL = './dataset_2_cov/dataset_2_cov_' + str(SET_ID) + '.csv'
    SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z = 4, 4, 4  # SPLIT_cell_ property of file generation

    COVER_spatial_small = [5, 5, 1]

    #----------------------------------------------------------
    # Generate cov 2, use algorithm 2
    #----------------------------------------------------------
    bool_cov2 = True
    selection_alg2_cov2 = TEST_cov_buckets(COVER_spatial_small, bool_cov2, CSV_CELL,
                                           SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z,
                                           SAVE_CACHE_alg2test, SAVE_SELECTION_alg2test,
                                           BETA_val=None)

    #----------------------------------------------------------
    # Generate cov 3, use algorithm 2
    #----------------------------------------------------------
    bool_cov2 = False  # meaning create cov3
    selection_alg2_cov3 = TEST_cov_buckets(COVER_spatial_small, bool_cov2, CSV_CELL,
                                           SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z,
                                           SAVE_CACHE_alg3test, SAVE_SELECTION_alg3test,
                                           BETA_val=4.)  # -> cutoff = 3

    assert np.allclose(selection_alg2_cov2.values, selection_alg2_cov3.values)
    pass


def TEST_cov_generation():
    # SAVE_CACHE_original = 'main_datasets_/placement_algorithm_cache.csv'

    SET_ID = 2
    CSV_CELL = './dataset_2_cov/dataset_2_cov_' + str(SET_ID) + '.csv'
    split_cell_x, split_cell_y, split_cell_z = 4, 4, 4  # SPLIT_cell_ property of file generation

    # COVER_spatial_original = [25, 25, 1]  # 70  # SPATIAL_COVER = 7  also size of covariance matrix in the end
    cover_spatial, file_cov_vv, file_cov_idxs, \
        file_delt_ch_it, SAVE_SELECTION_file, GEN_LOCAL_kernel, BETA_val \
        = a2t.get_spatial_splits(a2t.indirection)
    RUN_ALGO_2 = not GEN_LOCAL_kernel
    selection_alg2_cov2 = TEST_cov_buckets(cover_spatial, RUN_ALGO_2, CSV_CELL,
                                           split_cell_x, split_cell_y, split_cell_z,
                                           file_delt_ch_it, SAVE_SELECTION_file, BETA_val)
    return selection_alg2_cov2


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    RUN_SHORT_TESTS = False
    RUN_ORIGINAL = True
    if RUN_SHORT_TESTS:
        TEST_cov_2_equal_cov_3()

    if RUN_ORIGINAL:
        _, _, _ = TEST_cov_generation()

    # CSV_GP = 'main_datasets_/Dset_xyz_ave_small.csv'
    # boolcreate_cov_2 = True
    #TEST_cov_buckets(boolcreate_cov_2, CSV_GP)
    # TEST_encode_to_GP_fn()

    end_time = datetime.datetime.now()
    diff_time = end_time - start_time
    diff_time_str = str(diff_time)
    print('placement time: ', diff_time_str)

    pass
