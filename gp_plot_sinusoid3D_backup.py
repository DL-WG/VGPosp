import numpy as np
import tensorflow as tf
from matplotlib import cm

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()
from mpl_toolkits.mplot3d import Axes3D
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk
import matplotlib.pyplot as plt
from tensorboard.plugins import projector
import os

#%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
#%config InlineBackend.figure_format = 'png'
print(tf.__version__)

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

DEBUG_LOG=False
AMPLITUDE_INIT=np.array([0.1, 0.1]) # defines num of dimensions the kernel is in
LENGTHSCALE_INIT=np.array([1., 1.])
INIT_OBSNOISEVAR=1e-6
NUM_TRAINING_POINTS = 180
num_iters = 1000 # for length scale and amplitude
PRINT_PLOTS=True
PLOTRASTERPOINTS=60
num_samples = 20 # samples of GP
PRED_FRACTION=50 # 200 exhausted resources for 2D graph plot
COORDINATE_RANGE=np.array([[-2., 2.],[-2., 2.]]) # xrange , yrange
# COORDINATE_RANGE={"xrange": [0., 2.], "yrange": [0., 1.]}


def reset_session():
  """Creates a new global, interactive session in Graph-mode."""
  global sess
  try:
    tf.reset_default_graph()
    sess.close()
  except:
    pass
  sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=DEBUG_LOG))

reset_session()

#####################################
TEST_FN_PARAM = np.array([3 * np.pi, np.pi])

def sinusoid(x, scale=TEST_FN_PARAM): # R^N -> R
    s = 0
    #  iteration 1 ->
    for i in range(x.shape[1]):
       s += np.sin(3 * np.pi * x[:, i]) # eg [300,2]

    #  iteration 2 -> s = sin(scale[0] * x[:, 0] + scale[1] * [x[:, 1]])
    # a = np.dot(x, scale)
    # assert a.shape == (x.shape[0],)
    #
    # s = np.sin(a)
    return s

def sinusoid_test():
    N = 10
    x = np.random.uniform(-1., 1., (N, 2))
    y = sinusoid(x)
    print(y.shape)
    pass

#####################################

#3d plot of log marginal likelihood
def marginal_likelihood3D(figx, figy):
    fig2 = plt.figure(figsize=(figx, figy))
    ax2 = fig2.gca(projection='3d')

    # start2, stop2, step2 = np.min(r2), np.max(r2), (np.max(r2)-np.min(r2))/20
    start2, stop2, step2 = 1/xedges, 1, 1/xedges
    xedges2 = np.arange(start2, stop2+step2, step2)
    yedges2 = np.arange(start2, stop2+step2, step2)

    # Make data.
    X2 = np.linspace(start2, stop2, len(xedges2))
    Y2 = np.linspace(start2, stop2, len(yedges2))
    X2, Y2 = np.meshgrid(X2, Y2, sparse=False)    # 20x20

    # Plot the surface.
    H2 = (H-np.min(H))/(np.max(H)-np.min(H))
    # print(np.max(H2), np.min(H2))
    print(X2.shape, Y2.shape, H2.shape)
    surf2 = ax2.plot_surface(X2, Y2, H2, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)

    # Add a color bar whic(H2-np.min(H2))/(np.max(H2)-np.min(H2))h2.colorbar(surf2, shrink=0.5, aspect=5)
    ax2.set_xlabel('length_scale_assign')
    ax2.set_ylabel('amplitude_assign')
    ax2.set_zlabel('Z Label')
    # plt.show()


def generate_2d_data(num_training_points, observation_noise_variance, coord_range=COORDINATE_RANGE):
    """Generate noisy sinusoidal observations at a random set of points.

    Returns:
       observation_index_points, observations
    """
    index_points_ = np.random.uniform(0., 1., (num_training_points, 2))
    assert(index_points_.shape == (num_training_points, 2))
    index_points_ = index_points_.astype(np.float64)

    x_scale = coord_range[0][1] - coord_range[0][0]
    index_points_[:, 0] *= x_scale
    index_points_[:, 0] += coord_range[0][0]
    y_scale = coord_range[1][1] - coord_range[1][0]
    index_points_[:, 1] *= y_scale
    index_points_[:, 1] += coord_range[1][0]

    print("index_points_[:, 0]", index_points_[:, 0])
    print("index_points_[:, 1]", index_points_[:, 1])

    # y = f(x) + noise
    noise = np.random.normal(loc=0,
                             scale=np.sqrt(observation_noise_variance),
                             size=(num_training_points))

    assert(noise.shape==(num_training_points,))
    observations_ = (sinusoid(index_points_) + noise)
    return index_points_, observations_



def plot_samples2D(figx, figy, axis_k):
    plt.figure(figsize=(figx, figy))
    plt.plot(predictive_index_points_[:, axis_k], sinusoid(predictive_index_points_),label='True fn')
    plt.scatter(observation_index_points_[:, axis_k], observations_,label='Observations')
    for i in range(num_samples):
        plt.plot(predictive_index_points_[:, axis_k], samples_[i, axis_k, :].T, c='r', alpha=.1,
                 label='Posterior Sample' if i == 0 else None)
    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    # plt.show()

def plot_samples3D(figx,figy, axis_j, pred_index, pred_samples,obs_index_pts, obs):
    fig_ = plt.figure(figsize=(figx, figy))
    ax = fig_.gca(projection='3d')

    max_x = np.max(pred_index[:,0]) #define range
    max_y = np.max(pred_index[:,1])
    min_x = np.min(pred_index[:,0])
    min_y = np.min(pred_index[:,1])
    step_x = (max_x - min_x) / PRED_FRACTION
    step_y = (max_y - min_y) / PRED_FRACTION

    # stepx = (np.max(samples[:,0,:][1] - coord_range[0][0]) / number_of_points  # coord_range[0] = 'xrange', coord_range[1] = 'yrange'
    x = np.arange(min_x, max_x, step_x) # from to is excluding max_x to value -> we want to include max_x so +step_x

    # stepy = (coord_range[1][1] - coord_range[1][0]) / number_of_points
    y = np.arange(min_y, max_y, step_y)

    X, Y = np.meshgrid(x, y, sparse=False)
    # Z = np.zeros(X.shape)
    Z = pred_samples[0, 0, :]
    # Z = np.swapaxes(Z, 0,1)
    # assert Z.shape[2] == 2
    # Z = Z[:,0]
    Z = Z.reshape(X.shape[0],-1)
    # Z = np.reshape(0,1)
    print("max_x: ", max_x.shape)
    print("y: ", y.shape)
    print("x: ", x.shape)
    print("y: ", y.shape)
    print("X: ",X.shape)
    print("Y: ",Y.shape)
    print("Z: ",Z.shape)

    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False, alpha=0.4)
    ax.scatter(obs_index_pts[:, 0],
               obs_index_pts[:, 1],
               obs)    # ax.plot(predictive_index_points_[:, axis_j], sinusoid(predictive_index_points_),label='True fn')
    # ax.scatter(observation_index_points_[:, axis_j], observations_,label='Observations')
    # for i in range(num_samples):
    #     ax.plot(predictive_index_points_[:, axis_j], samples_[i, axis_j, :].T, c='r', alpha=.1,
    #              label='Posterior Sample' if i == 0 else None)
    # leg = ax.legend(loc='upper right')
    # for lh in leg.legendHandles:
    #     lh.set_alpha(1)
    # ax.xlabel(r"Index points ($\mathbb{R}^1$)")
    # ax.ylabel("Observation space")
    # plt.show()

    #####################################
    # Plot 3D sin

def plot_sin3d_rand_points(sinusoid_fn, coord_range, obs_index_pts, obs, number_of_points=PLOTRASTERPOINTS):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca(projection='3d')
    stepx = (coord_range[0][1] - coord_range[0][
        0]) / number_of_points  # coord_range[0] = 'xrange', coord_range[1] = 'yrange'
    x = np.arange(coord_range[0][0], coord_range[0][1] + stepx, stepx)
    stepy = (coord_range[1][1] - coord_range[1][0]) / number_of_points
    y = np.arange(coord_range[1][0], coord_range[1][1] + stepy, stepy)
    X, Y = np.meshgrid(x, y, sparse=False)
    # Z = np.sin((X)) * np.sin((Y))

    Z = np.zeros(X.shape)
    for j in range(Z.shape[1]):
        for i in range(Z.shape[0]):
            Z[i, j] = sinusoid_fn(np.array([[X[0, j], Y[i, 0]]]))  # draw above x,y, 1,2 shape

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
    # ax.scatter(xyz_pts[:,0],xyz_pts[:,1],xyz_pts[:,2])
    ax.scatter(obs_index_pts[:, 0],
               obs_index_pts[:, 1],
               obs)
    # ax.plot
    # plt.show()

#####################################

# Plot the loss evolution
def plot_loss():
    plt.figure(figsize=(12, 4))
    plt.plot(lls_[:, 0])
    plt.plot(lls_[:, 1])
    plt.xlabel("Training iteration")
    plt.ylabel("Log marginal likelihood")
    # plt.show()

#####################################
# Generate training data with a known noise level (we'll later try to recover
# this value from the data).
DO_ASSIGN = True

observation_index_points_, observations_ = generate_2d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=0.001)
print(observation_index_points_.shape, observations_.shape)
pass
#####################################

# draw boxes around the assignment op

# tf.nn.softplus(x) = log(exp(x) + 1) = z
# inv:   exp(z) = exp(x) + 1
#      exp(z)-1 = exp(x)
# log(exp(z)-1) = x
def invert_softplus(place_holder, variable, name='assign_op'):
    with tf.name_scope("invert_softplus"):
        value_of_placeholder = tf.log(tf.exp(place_holder) - 1)
        return variable.assign(value=value_of_placeholder, name=name)

# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.
# if DO_ASSIGN:

with tf.name_scope("amplitude"):
    amp_var = tf.Variable(initial_value=AMPLITUDE_INIT, name='amplitude',
                      dtype=np.float64)
    amplitude = (np.finfo(np.float64).tiny + tf.nn.softplus(amp_var))

with tf.name_scope("amplitude_assign"):
    amp_p = tf.placeholder(shape=AMPLITUDE_INIT.shape, dtype=np.float64, name='amp_p')
    amplitude_assign = invert_softplus(amp_p, amp_var)

with tf.name_scope("length_scale"):
    len_var = tf.Variable(initial_value=LENGTHSCALE_INIT, name='length_scale',
                          dtype=np.float64)
    length_scale = (np.finfo(np.float64).tiny + tf.nn.softplus(len_var))

with tf.name_scope("length_scale_assign"):
    len_p = tf.placeholder(shape=LENGTHSCALE_INIT.shape, dtype=np.float64, name='len_p')
    length_scale_assign = invert_softplus(len_p, len_var)

observation_noise_variance = (
        np.finfo(np.float64).tiny +
        tf.nn.softplus(tf.Variable(initial_value=INIT_OBSNOISEVAR,
                                   name='observation_noise_variance',
                                   dtype=np.float64)))
#####################################

# Create the covariance kernel, which will be shared between the prior (which we
# use for maximum likelihood training) and the posterior (which we use for
# posterior predictive sampling)
assert amplitude.shape == AMPLITUDE_INIT.shape
assert length_scale.shape == LENGTHSCALE_INIT.shape
assert amplitude.shape == length_scale.shape

kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

def kernel_test():
    k = None
    # select two points: X1, X2; k = kernel(X1, X2)
    for i in range(observation_index_points_.shape[0]):
        for j in range(observation_index_points_.shape[0]):
            if i != j:
                k = kernel._apply(observation_index_points_[i, :], observation_index_points_[j, :])
    pass
# kernel_test()
#####################################

# Create the GP prior distribution, which we will use to train the model
# parameters.
gp = tfd.GaussianProcess(
    kernel=kernel, # ([2,],[2,])
    index_points=observation_index_points_,
    observation_noise_variance=observation_noise_variance)
print("observation_index_points_", observation_index_points_.shape)
print("observation_noise_variance", observation_noise_variance.shape)

# This lets us compute the log likelihood of the observed data. Then we can
# maximize this quantity to find optimal model parameters.
log_likelihood = gp.log_prob(observations_)
tf.summary.scalar("log_likelihood[0, 0]", log_likelihood[0])
tf.summary.scalar("log_likelihood[1, 0]", log_likelihood[1])
#####################################

# Define the optimization ops for maximizing likelihood (minimizing neg
# log-likelihood!)
optimizer = tf.train.AdamOptimizer(learning_rate=.01)
train_op = optimizer.minimize(-log_likelihood)
#####################################

xedges = 60
yedges = 60
emb_p = tf.placeholder(shape=[xedges, yedges], dtype=np.float32, name='emb_p')
# emb will hold the same values as H in the 3d graph at the end
# (looking at it with PCA, though, does not make much sense)
emb = tf.Variable(tf.zeros([xedges, yedges]), name="log_probability_embedding")
emb_as_op = emb.assign(emb_p, name='emb_as_op')

#####################################

# ready with the graph
LOGDIR = "./log_dir_gp/gp_plot/"
summ = tf.summary.merge_all()

def projector_add(embedding, writer, SPRITES=None, LABELS=None, IMG_DIM=None):
    # config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    if SPRITES:
        embedding_config.sprite.image_path = SPRITES
    if LABELS:
        embedding_config.metadata_path = LABELS
    if IMG_DIM:
        # Specify the width and height of a single thumbnail.
        embedding_config.sprite.single_image_dim.extend(IMG_DIM)  # like [28, 28]
    #tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    projector.visualize_embeddings(writer, config)


#####################################
#####################################
#####################################
# save the graph into a log folder
sess.run(tf.global_variables_initializer()) # after the init

# !rm -f LOGDIR
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
# projector_add(kernel, writer)         # does not seem to be a corresponding tf.Variant
projector_add(emb, writer)              # log_probability_embedding
saver = tf.train.Saver()

#####################################

if DO_ASSIGN:
    [_, amplitude_] = \
        sess.run([amplitude_assign,
                  amplitude  # evaluated in order
                  ], feed_dict={amp_p: [0.5, 0.5]})

# else:
#     [amplitude_,
#      length_scale_,
#      observation_noise_variance_] = \
#       sess.run([
#                 amplitude,
#                 length_scale,
#                 observation_noise_variance])

print('{} parameters:'.format('Assigned' if DO_ASSIGN else 'Default'))
print('amplitude_: {}'.format(amplitude_))
#####################################

if DO_ASSIGN:
    [_, length_scale_,
     ] = \
        sess.run([length_scale_assign,
                  length_scale,
                  ], feed_dict={len_p: [0.5, 0.5]})

print('length_scale_: {}'.format(length_scale_))
#####################################

if DO_ASSIGN:
    [observation_noise_variance_] = sess.run([observation_noise_variance])

print('observation_noise_variance: {}'.format(observation_noise_variance_))
#####################################

# Now we optimize the model parameters.
# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros([num_iters+1, 2], np.float64)

# initial test tun
# [_, l, s] = sess.run([train_op, log_likelihood, summ])

# train amplitude and length_scale
for i in range(num_iters+1):
  _, lls_[i], s = sess.run([train_op, log_likelihood, summ])
  writer.add_summary(s, i)
  if i%200 == 0:
    saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
#####################################

# print the value of the trained parameters
[amplitude_,
 length_scale_,
 observation_noise_variance_] \
    = sess.run([
        amplitude,
        length_scale,
        observation_noise_variance])
print('Trained parameters:'.format(amplitude_))
print('amplitude_: {}'.format(amplitude_))
print('length_scale_: {}'.format(length_scale_))
print('observation_noise_variance: {}'.format(observation_noise_variance_))

# Trained parameters:
# amplitude_: [1.29834517]
# length_scale_: [0.22509762]
# observation_noise_variance: 0.009865463915545956
# lls_[num_iters-1] ~= 50


#####################################

# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.

pred_x = np.linspace(-2, 2, PRED_FRACTION, dtype=np.float64)
pred_y = np.linspace(-2, 2, PRED_FRACTION, dtype=np.float64)

print("predx: ", pred_x.shape)
print("predy: ", pred_y.shape)

# [2, 200,200]
h = np.array(np.meshgrid(pred_x, pred_y,sparse=False))
print("h pred_ind", h.shape)
# [2, 200,200] -> [200*200, 2]
h2 = h.swapaxes(0,-1)
h3 = h2.reshape(-1, 2)
print("h3 pred_ind", h3.shape) #[200*200, 2]

predictive_index_points_ = h3
print("pred_ind", predictive_index_points_.shape)

# print("pred", predictive_index_points_.shape)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
# assert predictive_index_points_.shape == (PRED_FRACTION, 2)
# predictive_index_points_ = predictive_index_points_[..., np.newaxis]

gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel,  # Reuse the same kernel instance, with the same params
    index_points=predictive_index_points_,  # (2500,2)
    observation_index_points=observation_index_points_, # (180, 2)
    observations=observations_,  # (180,)
    observation_noise_variance=observation_noise_variance,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
samples = gprm.sample(num_samples)


#####################################
# Draw samples and visualize.
samples_ = sess.run(samples)
print(samples_.shape)
print(samples_[0,0,:])
assert samples_.shape[0] == num_samples
assert samples_.shape[1] == LENGTHSCALE_INIT.shape[0]
assert samples_.shape[2] == PRED_FRACTION**2

print("sample, max", np.max(samples_[0,0,:]))
print("sample, min", np.min(samples_[0,1,:]))

# samples_.shape == (50,2,200)
# Plot the true function, observations, and posterior samples.

pass
# define inpdata
#xedges = 60
#yedges = 60
H = np.zeros([xedges, yedges])

for i in range(xedges):
    for j in range(yedges):
        #print('####################', i, j)
        [ _,
          _,
          L
        ] = sess.run([
            length_scale_assign,
            amplitude_assign,

            log_likelihood
        ], feed_dict={len_p: [2*np.double((1+i)/xedges), length_scale_[1]],
                      amp_p: [2*np.double((1+j)/yedges), amplitude_[1]]})
        H[i, j] = L[0]

assert(amplitude.shape==(2))
assert(length_scale.shape==(2))
# 5th dimension = log-likelihood at points xyzw

#H
# set the value of emb
[_] = sess.run([emb_as_op], feed_dict={emb_p: H})
saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), num_iters+1)


if PRINT_PLOTS:
    plot_loss()
    plot_samples2D(12, 4, 1)
    plot_samples3D(12, 12, 0, predictive_index_points_, samples_, observation_index_points_, observations_)
    marginal_likelihood3D(12, 12)
    plot_sin3d_rand_points(sinusoid, COORDINATE_RANGE, observation_index_points_, observations_)
    sinusoid_test()
    pass

plt.show()
pass



