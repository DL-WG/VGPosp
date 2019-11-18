import numpy as np
import tensorflow as tf
import data_readers as drs

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from matplotlib import cm

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tf.keras
from tensorflow_probability import positive_semidefinite_kernels as tfkern

import plots as plts

DTYPE = np.float64


#==============================================================
# DATA
#==============================================================
# Generate noisy data from a known function.
def data(NUM_TRAIN_PTS):
    f = lambda x: np.exp(-x[..., 0]**2 / 20.) * np.sin(1. * x[..., 0])
    true_observation_noise_variance_ = DTYPE(1e-1) ** 2

    x_train_ = np.random.uniform(-10., 10., [NUM_TRAIN_PTS, 1])
    y_train_ = (f(x_train_) +
                np.random.normal(
                    0., np.sqrt(true_observation_noise_variance_),
                    [NUM_TRAIN_PTS]))
    return x_train_, y_train_, f
# Create kernel with trainable parameters, and trainable observation noise
# variance variable. Each of these is constrained to be positive.


#==============================================================
# GRAPH
#==============================================================
def graph(x_train, y_train, NUM_INDUCING_PTS, NUM_PREDICTIVE_IDX_PTS, TRAINING_MINIBATCH_SIZE, NUM_TRAIN_PTS):
    amplitude = (tf.nn.softplus(
      tf.Variable(.54, dtype=DTYPE, name='amplitude', use_resource=True)))

    length_scale = (
      1e-5 +
      tf.nn.softplus(
        tf.Variable(.54, dtype=DTYPE, name='length_scale', use_resource=True)))

    kernel = tfkern.ExponentiatedQuadratic( # MaternOneHalf
        amplitude=amplitude,
        length_scale=length_scale)

    obs_noise_var = tf.nn.softplus(
        tf.Variable(
          .54, dtype=DTYPE, name='observation_noise_variance', use_resource=True))

    # Create trainable inducing point locations and variational parameters.
    inducing_index_points = tf.Variable(
        np.linspace(-10., 10., NUM_INDUCING_PTS)[..., np.newaxis],
        dtype=DTYPE, name='inducing_index_points', use_resource=True)

    variational_loc, variational_scale = (
        tfd.VariationalGaussianProcess.optimal_variational_posterior(
            kernel=kernel,
            inducing_index_points=inducing_index_points,
            observation_index_points=x_train,
            observations=y_train,
            observation_noise_variance=obs_noise_var))

    # These are the index point locations over which we'll construct the
    # (approximate) posterior predictive distribution.
    index_points = np.linspace(-13, 13,
                                NUM_PREDICTIVE_IDX_PTS,
                                dtype=DTYPE)[..., np.newaxis]

    # Construct our variational GP Distribution instance.
    vgp = tfd.VariationalGaussianProcess(
        kernel,
        index_points=index_points,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=variational_loc,
        variational_inducing_observations_scale=variational_scale,
        observation_noise_variance=obs_noise_var)

    # For training, we use some simplistic numpy-based minibatching.
    x_train_batch = tf.placeholder(DTYPE, [TRAINING_MINIBATCH_SIZE, 1], name='x_train_batch')
    y_train_batch = tf.placeholder(DTYPE, [TRAINING_MINIBATCH_SIZE], name='y_train_batch')

    # Create the loss function we want to optimize.
    loss = vgp.variational_loss(
        observations=y_train_batch,
        observation_index_points=x_train_batch,
        kl_weight=float(TRAINING_MINIBATCH_SIZE) / float(NUM_TRAIN_PTS))

    optimizer = tf.train.AdamOptimizer(learning_rate=.01)
    train_op = optimizer.minimize(loss)

    return train_op, loss, x_train_batch, y_train_batch, vgp, inducing_index_points, variational_loc, index_points


#==============================================================
# SESS
#==============================================================
def sess_runs(x_train, y_train, train_op, loss,
              x_train_batch, y_train_batch, vgp,
              inducing_index_points, variational_loc,
              NUM_TRAIN_ITERS, NUM_TRAIN_PTS, TRAINING_MINIBATCH_SIZE,
              NUM_TRAIN_LOGS, NUM_GP_SAMPLES):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(NUM_TRAIN_ITERS):
            batch_idxs = np.random.randint(NUM_TRAIN_PTS, size=[TRAINING_MINIBATCH_SIZE])
            x_train_batch_ = x_train[batch_idxs, ...]  # x_train[batch_idxs, ...]
            y_train_batch_ = y_train[batch_idxs]

            [_, loss_] = sess.run([train_op, loss],
                                  feed_dict={x_train_batch: x_train_batch_,
                                             y_train_batch: y_train_batch_})
            if i % (NUM_TRAIN_ITERS / NUM_TRAIN_LOGS) == 0 or i + 1 == NUM_TRAIN_ITERS:
              print(i, loss_)

        # Generate a plot with
        #   - the posterior predictive mean
        #   - training data
        #   - inducing index points (plotted vertically at the mean of the
        #     variational posterior over inducing point function values)
        #   - 50 posterior predictive samples
            [
              samples_,
              mean_,
              inducing_index_points_,
              variational_loc_,
            ] = sess.run([
              vgp.sample(NUM_GP_SAMPLES),
              vgp.mean(),
              inducing_index_points,
              variational_loc
            ])

    return train_op, loss, x_train_batch, y_train_batch, vgp, \
           samples_, mean_, inducing_index_points_, variational_loc_

def prints(f, x_train, y_train, inducing_index_points, variational_loc, samples, mean, index_points, NUM_GP_SAMPLES):
    # plts.plot_gp_linesamples(12, 4, index_points,  # (15,1)
    #                          f(index_points),  # (15,)
    #                          inducing_index_points,  # (100,1)
    #                          samples,  # (8,2,100)
    #                          NUM_GP_SAMPLES)  # (8)

    plt.figure(figsize=(15, 5))
    plt.scatter(inducing_index_points[..., 0], variational_loc,
              marker='x', s=50, color='k', zorder=10, label='variational_loc')
    plt.scatter(x_train[..., 0], y_train, color='#00ff00', alpha=.1, zorder=9, label='y_train')
    plt.plot(np.tile(index_points, NUM_GP_SAMPLES),
           samples.T, color='r', alpha=.1, label=None)
    plt.plot(index_points, mean, color='k', label='mean')
    # plt.gca().fill_between(index_points, mean - s, mean + s, color="#dddddd")
    plt.plot(index_points, f(index_points), color='b', label='f(index_points)')

    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.show()
    pass


#==============================================================
# HYPERPARAMETER TEST
#==============================================================
def TEST_hyperparameters(NUM_INDUCING_PTS_,
                         NUM_PREDICTIVE_IDX_PTS_,
                         TRAINING_MINIBATCH_SIZE_,
                         NUM_GP_SAMPLES_,
                         NUM_TRAIN_ITERS_,
                         NUM_TRAIN_LOGS_,
                         NUM_TRAIN_PTS_):

    x_train_, y_train_, f_ = data(NUM_TRAIN_PTS_)

    #--------------------------------------------------
    # GRAPH
    #--------------------------------------------------
    train_op, loss, x_train_batch, y_train_batch, vgp, \
        inducing_index_points, variational_loc, index_points \
        = graph(x_train_, y_train_,
                NUM_INDUCING_PTS_, NUM_PREDICTIVE_IDX_PTS_,
                TRAINING_MINIBATCH_SIZE_, NUM_TRAIN_PTS_)

    #--------------------------------------------------
    # SESS
    #--------------------------------------------------
    train_op, loss, x_train_batch, y_train_batch, vgp, \
    samples, mean, inducing_index_points, variational_loc \
        = sess_runs(x_train_, y_train_, train_op, loss,
                    x_train_batch, y_train_batch, vgp,
                    inducing_index_points, variational_loc,
                    NUM_TRAIN_ITERS_, NUM_TRAIN_PTS_, TRAINING_MINIBATCH_SIZE_,
                    NUM_TRAIN_LOGS_, NUM_GP_SAMPLES_)

    #--------------------------------------------------
    # PRINT
    #--------------------------------------------------
    prints(f_, x_train_, y_train_,
           inducing_index_points, variational_loc,
           samples, mean, index_points,
           NUM_GP_SAMPLES_)


if __name__ == '__main__':
    '''
        NUM_INDUCING_PTS_, 
        NUM_PREDICTIVE_IDX_PTS_, 
        TRAINING_MINIBATCH_SIZE_, 
        NUM_GP_SAMPLES_, 
        NUM_TRAIN_ITERS_, 
        NUM_TRAIN_LOGS_, 
        NUM_TRAIN_PTS_ 
    '''
    TEST_hyperparameters(10, 500, 64, 20, 10, 10, 100)
    TEST_hyperparameters(20, 500, 64, 20, 50, 10, 300)
    TEST_hyperparameters(30, 500, 64, 20, 100, 10, 500)



    pass
