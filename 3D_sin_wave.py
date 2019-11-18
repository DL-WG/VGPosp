import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk
psd_kernels = tfp.positive_semidefinite_kernels

#################### Configure plot defaults
#%pylab inline
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
#%config InlineBackend.figure_format = 'png'
print(tf.__version__)
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

#################### """Creates a new global, interactive session in Graph-mode."""
def reset_session():
  global sess
  try:
    tf.reset_default_graph()
    sess.close()
  except:
    pass
  sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

reset_session()

# def generate_3Ddata(numXY_pts, obs_noisevar, coord_range):
#     """Generate noisy sinusoidal observations at a random set of points at xy coordinates given
#     by idx_pts and numXY_pts.
#     Returns:
#        observation_index_points, observations
#     """
#     idx_pts = np.random.uniform(-coord_range, coord_range, (numXY_pts, 2)) # create XY coordinate pairs
#     idx_pts = idx_pts.astype(np.float64)
#     # y = f(x) + noise
#     obs = (sinusoid(idx_pts) +
#                      np.random.normal(loc=0,
#                                       scale=np.sqrt(obs_noisevar),
#                                       size=idx_pts.shape))
#     # idx_pts: (100,2)
#     # obs: (100,) values taken on axis z (up)
#     return idx_pts, obs

###########################################################
# GENERATE TRAINING DATA

PLOTFIGUREWIDTH=10
PLOTFIGUREHEIGHT=8
COORDINATE_RANGE=6
NUM_OF_COORDINATES=100
NOISE=0.1
LEARNING_RATE=0.01
INIT_AMPLITUDE=0.1
INIT_LENSCALE=0.9
INIT_OBSNOISEVAR=1e-6
LOGDIR = "./log_dir_gp/gp_sin_3D_plot/"
DO_ASSIGN=True
PRINT_PLOTS=False
NUM_OPTIMIZATION_ITER=600
NUM_GP_SAMPLES=10 #10
FIRSTELEM = True # for observations


def plot_sin3d(coord_range):
    fig = plt.figure(figsize=(PLOTFIGUREWIDTH, PLOTFIGUREHEIGHT))
    ax = fig.gca(projection='3d')
    x = np.arange(-coord_range, coord_range, 0.1)
    y = np.arange(-coord_range, coord_range, 0.1)
    X, Y = np.meshgrid(x, y, sparse=True)
    Z = np.sin((X)) * np.sin((Y))
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

def plot_sin3d_rand_points(coord_range, xyz_pts):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca(projection='3d')
    x = np.arange(-coord_range, coord_range, 0.1)
    y = np.arange(-coord_range, coord_range, 0.1)
    X, Y = np.meshgrid(x, y, sparse=True)
    Z = np.sin((X)) * np.sin((Y))
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False, alpha=0.5)
    ax.scatter(xyz_pts[:,0],xyz_pts[:,1],xyz_pts[:,2])
    plt.show()

def random_xy(numXY_pts, coord_range):
    rand_xy = np.random.uniform(-coord_range, coord_range, (numXY_pts, 2)) # create XY coordinate pairs
    rand_xy = rand_xy.astype(np.float64)
    return rand_xy

def addnoisy_z_sin_coord(xy, noise):
    z = np.sin(xy[0]) * np.sin(xy[1]) + noise
    xyz = np.array([xy[0], xy[1], z])
    return xyz

def noisy_z_sin_coord(xy, noise):
    z = np.sin(xy[0]) * np.sin(xy[1]) + noise
    return z

def rand_noisy_3Dsin_coord(numXY_pts, coord_range):
    xy_pts = random_xy(numXY_pts, coord_range)
    xyz_pts = np.zeros((0, 3))
    for i in range(xy_pts.shape[0]):
        xyz = addnoisy_z_sin_coord(xy_pts[i], NOISE)
        if xyz_pts.shape[0]==0:
            xyz_pts = xyz
        else:
            xyz_pts = np.vstack((xyz_pts, xyz))
    return xyz_pts

#Generate
trainingpts = rand_noisy_3Dsin_coord(NUM_OF_COORDINATES, COORDINATE_RANGE)

#Plot
if PRINT_PLOTS:
    plot_sin3d_rand_points(COORDINATE_RANGE, trainingpts)

###################################################################
# GP

observation_noise_variance = (
        np.finfo(np.float64).tiny +
        tf.nn.softplus(tf.Variable(initial_value=INIT_OBSNOISEVAR,
                                   name='observation_noise_variance',
                                   dtype=np.float64)))

# Draw boxes around the assignment op
def invert_softplus(place_holder, variable, name='assign_op'):
    with tf.name_scope("invert_softplus"):
        value_of_placeholder = tf.log(tf.exp(place_holder) - 1)
        return variable.assign(value=value_of_placeholder, name=name)

# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.
with tf.name_scope("amplitude"):
    amp_var = tf.Variable(initial_value=[INIT_AMPLITUDE], name='amplitude',
                      dtype=np.float64)
    amplitude = (np.finfo(np.float64).tiny + tf.nn.softplus(amp_var))

with tf.name_scope("amplitude_assign"):
    amp_p = tf.placeholder(shape=[1], dtype=np.float64, name='amp_p')
    amplitude_assign = invert_softplus(amp_p, amp_var)

with tf.name_scope("length_scale"):
    len_var = tf.Variable(initial_value=[INIT_LENSCALE], name='length_scale',
                          dtype=np.float64)
    length_scale = (np.finfo(np.float64).tiny + tf.nn.softplus(len_var))

with tf.name_scope("length_scale_assign"):
    len_p = tf.placeholder(shape=[1], dtype=np.float64, name='len_p')
    length_scale_assign = invert_softplus(len_p, len_var)

def cov_kernel(amplitude_, length_scale_):
    return tfk.ExponentiatedQuadratic(amplitude_, length_scale_)

def gp_priordistrib(krl,ipts,onv):
    ''' Create the GP prior distribution,
    which we will use to train the model parameters.'''
    gp = tfd.GaussianProcess(
        kernel=krl,
        index_points=ipts,
        observation_noise_variance=onv)
    return gp

def calc_log_likelihood(observations):
    ''' Use: Optimize model parameters wrt log-likelihood'''
    return gp.log_prob(observations[:,0])

kernel = cov_kernel(amplitude, length_scale)

gp = gp_priordistrib(kernel, trainingpts, observation_noise_variance)

loglh = calc_log_likelihood(trainingpts)
tf.summary.scalar("log_likelihood", loglh[0])

# Define the optimization ops for maximizing likelihood (minimizing neg log-likelihood!)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(-loglh)
summ = tf.summary.merge_all()
sess.run(tf.global_variables_initializer()) # after the init
#!rm -f LOGDIR
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

[_, amplitude_] = sess.run([amplitude_assign, amplitude], feed_dict={amp_p: [0.5]})
print('{} parameters:'.format('Assigned' if DO_ASSIGN else 'Default'))
print('amplitude_: {}'.format(amplitude_))

if DO_ASSIGN:
    [_, length_scale_,] = sess.run([length_scale_assign,
                  length_scale,], feed_dict={len_p: [0.5]})
print('length_scale_: {}'.format(length_scale_))

if DO_ASSIGN:
    [observation_noise_variance_] = sess.run([observation_noise_variance])
print('observation_noise_variance: {}'.format(observation_noise_variance_))


# Store the likelihood values during training, so we can plot the progress
loglh_ = np.zeros(NUM_OPTIMIZATION_ITER, np.float64)

for i in range(NUM_OPTIMIZATION_ITER):
  _, loglh_[i], s = sess.run([train_op, loglh, summ])
  writer.add_summary(s, i)

[ampl, lensc, obsnoisevar] = sess.run([amplitude,
                                    length_scale,
                                    observation_noise_variance])
print('Trained parameters:'.format(ampl))
print('amplitude_: {}'.format(ampl))
print('length_scale_: {}'.format(lensc))
print('observation_noise_variance: {}'.format(obsnoisevar))

LENGTH = 200

a_ = np.linspace(-1.2, 1.2, LENGTH, dtype=np.float64)
b_= np.linspace(-6, 6, LENGTH, dtype=np.float64)
FIRSTELEM = True
for i in range(LENGTH):
    c_ = np.meshgrid(a_[i], b_[i])
    if FIRSTELEM == True:
        FIRSTELEM = False
        pred_pts = c_
    else:
        pred_pts = np.vstack((pred_pts, c_))

pred_pts = np.reshape(pred_pts, [-1, 2]) #[:, :, 0]
pass

xy_pts = random_xy(100, 6)
xyz_pts = np.zeros((0, 3))
for i in range(xy_pts.shape[0]):
    xyz = addnoisy_z_sin_coord(xy_pts[i], NOISE)
    if xyz_pts.shape[0] == 0:
        xyz_pts = xyz
    else:
        xyz_pts = np.vstack((xyz_pts, xyz))

observations=np.zeros((1,1))
FIRSTELEM_=True
for i in range(xy_pts.shape[0]):
    z = noisy_z_sin_coord(xy_pts[i], NOISE)
    if FIRSTELEM_ == True:
        observations[0] = z
        FIRSTELEM_=False
    else:
        observations = np.vstack((observations, z))
observations = np.reshape(observations, [-1])

observation_noise_variance=0.01

# print(kernel)
print(pred_pts)
# print(xy_pts)
# print(observations)

gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel,  # Reuse the same kernel instance, with the same params
    index_points=pred_pts,  # 1D_emb:(400,1), 2D_emb:(200,2)
    observation_index_points=xy_pts,  # (15,1) (15,2)
    observations=observations,  # (15,)
    observation_noise_variance=observation_noise_variance,
    validate_args=True)

samples = gprm.sample(NUM_GP_SAMPLES)
gpsamples = sess.run(samples)

#TODO
# fix gprm index_.....
# condition somehow

def plot2D_samples(X, Y): # Plot the true function, observations, and posterior samples.

    plt.figure(figsize=(X, Y))
    # plt.plot(pred_pts, sinusoid(predictive_index_points_),
    #          label='True fn')
    plt.scatter(trainingpts[:, 0], trainingpts[:,2],
                label='Observations')
    for i in range(NUM_GP_SAMPLES):
        plt.plot(pred_pts, gpsamples[i, :].T, c='r', alpha=.1,
                 label='Posterior Sample' if i == 0 else None)
    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.show()

plot2D_samples(PLOTFIGUREWIDTH,PLOTFIGUREHEIGHT) # log-marginal-likelihood

pass