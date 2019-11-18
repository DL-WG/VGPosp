import numpy as np
import tensorflow as tf
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

#####################################
def reset_session():
  """Creates a new global, interactive session in Graph-mode."""
  global sess
  try:
    tf.reset_default_graph()
    sess.close()
  except:
    pass
  sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

reset_session()

#####################################

def sinusoid(x):
    return np.sin(3 * np.pi * x[..., 0])


def generate_1d_data(num_training_points, observation_noise_variance):
    """Generate noisy sinusoidal observations at a random set of points.

    Returns:
       observation_index_points, observations
    """
    index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
    index_points_ = index_points_.astype(np.float64)
    # y = f(x) + noise
    observations_ = (sinusoid(index_points_) +
                     np.random.normal(loc=0,
                                      scale=np.sqrt(observation_noise_variance),
                                      size=(num_training_points)))
    return index_points_, observations_
#####################################

# Generate training data with a known noise level (we'll later try to recover
# this value from the data).
DO_ASSIGN = True

NUM_TRAINING_POINTS = 100
observation_index_points_, observations_ = generate_1d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=.01)
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
    amp_var = tf.Variable(initial_value=[1.0], name='amplitude',
                      dtype=np.float64)
    amplitude = (np.finfo(np.float64).tiny + tf.nn.softplus(amp_var))

with tf.name_scope("amplitude_assign"):
    amp_p = tf.placeholder(shape=[1], dtype=np.float64, name='amp_p')
    amplitude_assign = invert_softplus(amp_p, amp_var)

with tf.name_scope("length_scale"):
    len_var = tf.Variable(initial_value=[1.0], name='length_scale',
                          dtype=np.float64)
    length_scale = (np.finfo(np.float64).tiny + tf.nn.softplus(len_var))

with tf.name_scope("length_scale_assign"):
    len_p = tf.placeholder(shape=[1], dtype=np.float64, name='len_p')
    length_scale_assign = invert_softplus(len_p, len_var)

observation_noise_variance = (
        np.finfo(np.float64).tiny +
        tf.nn.softplus(tf.Variable(initial_value=1e-6,
                                   name='observation_noise_variance',
                                   dtype=np.float64)))
#####################################

# Create the covariance kernel, which will be shared between the prior (which we
# use for maximum likelihood training) and the posterior (which we use for
# posterior predictive sampling)
kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
#####################################

# Create the GP prior distribution, which we will use to train the model
# parameters.
gp = tfd.GaussianProcess(
    kernel=kernel,
    index_points=observation_index_points_,
    observation_noise_variance=observation_noise_variance)

# This lets us compute the log likelihood of the observed data. Then we can
# maximize this quantity to find optimal model parameters.
log_likelihood = gp.log_prob(observations_)
tf.summary.scalar("log_likelihood", log_likelihood[0])
#####################################

# Define the optimization ops for maximizing likelihood (minimizing neg
# log-likelihood!)
optimizer = tf.train.AdamOptimizer(learning_rate=.01)
train_op = optimizer.minimize(-log_likelihood)
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

# save the graph into a log folder
sess.run(tf.global_variables_initializer()) # after the init

#!rm -f LOGDIR
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
# projector_add(kernel, writer)         # does not seem to be a corresponding tf.Variant
saver = tf.train.Saver()

#####################################

if DO_ASSIGN:
    [_, amplitude_] = \
        sess.run([amplitude_assign,
                  amplitude  # evaluated in order
                  ], feed_dict={amp_p: [0.5]})

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
                  ], feed_dict={len_p: [0.5]})

print('length_scale_: {}'.format(length_scale_))
#####################################

if DO_ASSIGN:
    [observation_noise_variance_
     ] = \
        sess.run([observation_noise_variance
                  ])

print('observation_noise_variance: {}'.format(observation_noise_variance_))
#####################################

# Now we optimize the model parameters.
num_iters = 600
# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters+1, np.float64)

for i in range(num_iters+1):
  _, lls_[i], s = sess.run([train_op, log_likelihood, summ])
  writer.add_summary(s, i)
  if i%200 == 0:
    saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
#####################################

[amplitude_,
 length_scale_,
 observation_noise_variance_] = sess.run([
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

# Plot the loss evolution
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
# plt.show()
#####################################

# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.
predictive_index_points_ = np.linspace(-1.2, 1.2, 200, dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel,  # Reuse the same kernel instance, with the same params
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
num_samples = 50
samples = gprm.sample(num_samples)
#####################################

# Draw samples and visualize.
samples_ = sess.run(samples)

# Plot the true function, observations, and posterior samples.
plt.figure(figsize=(12, 4))
plt.plot(predictive_index_points_, sinusoid(predictive_index_points_),
         label='True fn')
plt.scatter(observation_index_points_[:, 0], observations_,
            label='Observations')
for i in range(num_samples):
    plt.plot(predictive_index_points_, samples_[i, :].T, c='r', alpha=.1,
             label='Posterior Sample' if i == 0 else None)
leg = plt.legend(loc='upper right')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
# plt.show()


#####################################
#3d plot of log marginal likelihood
def plot3D(edges, figX, figY, xlabel=None, ylabel=None, zlabel=None):

    fig = plt.figure(figsize = (figX,figY))
    axFig = fig.gca(projection='3d')
    axisX, axisY = np.meshgrid(calc_axisZ(edges), calc_axisZ(edges), sparse=False)  # 20x20
    H = calc_log_likelihood(edges, edges)
    H2 = (H-np.min(H))/(np.max(H)-np.min(H)) # Plot the surface.
    surf=axFig.plot_surface(axisX, axisY, H2, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False) # Surface

    # print(np.max(H2), np.min(H2))
    # print(axisX.shape, axisY.shape, H2.shape)
    # (H2-np.min(H2))/(np.max(H2)-np.min(H2))*H2.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar
    axFig.set_xlabel(xlabel) # Axis labels
    axFig.set_ylabel(ylabel)
    axFig.set_zlabel(zlabel)
    plt.show()
#####################################
# Define input data.
def calc_log_likelihood(X, Y): # edges
    H = np.zeros([X, Y])
    for i in range(X):
        for j in range(Y):
            [_, _, _, H[i, j]] = sess.run([
                length_scale_assign,
                amplitude_assign,
                train_op,
                log_likelihood
            ], feed_dict={len_p: [np.double((1 + i) / X)],
                          amp_p: [np.double((1 + j) / Y)]})
    return H
#####################################
def calc_axisZ(edges):
    start, stop, step = 1/edges, 1, 1/edges
    edges_ = np.arange(start, stop+step, step)
    o_axis = np.linspace(start, stop, len(edges_))
    return o_axis

#####################################

plot3D(60,12,8,'length_scale_assign','amplitude_assign','Z Label')

#####################################


