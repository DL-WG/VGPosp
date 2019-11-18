import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()
from mpl_toolkits.mplot3d import Axes3D
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfkern
import matplotlib.pyplot as plt
from tensorboard.plugins import projector
import os
import time
#%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
#%config InlineBackend.figure_format = 'png'
print(tf.__version__)

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

import data_generation as dg
# import data_readers as drs
import normalize_delete as nm

###############################################
# TEST_FN_PARAM = np.array([3 * np.pi, np.pi])
TEST_FN_PARAM = 1#np.array([1 * np.pi, np.pi])
DEBUG_LOG=False

def sinusoid_test():
    N = 10
    x = np.random.uniform(-1., 1., (N, 2))
    y = sinusoid(x, [1,1])
    y2 = sinusoid_(x)


def tf_Placeholder_assign_test(AMPLITUDE_INIT, LENGTHSCALE_INIT, INIT_OBSNOISEVAR):
    amp_var, amp = tf_Variable("amplitude",'amplitude',AMPLITUDE_INIT)
    amp_plh, amp_assign = tf_Placeholder_assignments(amp_var, "amplitude_assign",'amplitude_assign', AMPLITUDE_INIT)

    lensc_var, lensc = tf_Variable("lengthscale",'lengthscale', LENGTHSCALE_INIT)
    lensc_plh, lensc_assign = tf_Placeholder_assignments(lensc_var, "lengthscale_assign", 'lengthscale_assign', LENGTHSCALE_INIT)

    _, obs_noise_var = tf_Variable("observation_noise_variance",'observation_noise_variance', INIT_OBSNOISEVAR)
    emb_var, emb = tf_Variable("log_probability_embedding",'log_probability_embedding',LENGTHSCALE_INIT)
    emb_plh, emb_assign = tf_Placeholder_assignments(emb_var,"log_probability_embedding_assign",'log_probability_embedding_assign', LENGTHSCALE_INIT)
    assert amp_assign.shape == AMPLITUDE_INIT.shape
    assert lensc_assign.shape == LENGTHSCALE_INIT.shape
    assert amp_assign.shape == lensc_assign.shape

    return amp, amp_assign, amp_plh, \
            lensc, lensc_assign, lensc_plh,\
            emb, emb_assign, emb_plh, \
            obs_noise_var


def create_cov_kernel_test(obs_idx_pts):
    kernel = create_cov_kernel()
    # select two points: X1, X2; k = kernel(X1, X2)
    for i in range(obs_idx_pts.shape[0]):
        for j in range(obs_idx_pts.shape[0]):
            if i != j:
                k = kernel._apply(obs_idx_pts[i, :], obs_idx_pts[j, :])
    return k


def sinusoid(x, scale=TEST_FN_PARAM):
    '''
    :param x: ND array (num, N)
    :param scale: ( , ) R^N -> R
    :return: 1D
    '''
    s = 0
    SIN_DENSITY= 2
    # print("x.shape : ", x.shape)

    if 1 < len(x.shape):
        for i in range(x.shape[1]):
            s += np.sin(SIN_DENSITY * np.pi * x[:, i])  # eg [300,2]
    else:
        assert 1 == len(x.shape)
        s += np.sin(SIN_DENSITY * np.pi * x[:])

    return s


def sinusoid_(x, scale=TEST_FN_PARAM):  # R^N -> R
    # s = np.sin(scale[0] * x[:, 0] + scale[1] * [x[:, 1]])
    a = np.dot(x, scale)
    assert a.shape == (x.shape[0],)
    s = np.sin(a)
    return s


def invert_softplus(place_holder, variable, name='assign_op'):
    with tf.name_scope("invert_softplus"):
        value_of_placeholder = tf.log(tf.exp(place_holder) - 1)
        return variable.assign(value=value_of_placeholder, name=name)


def reset_session():
  """Creates a new global, interactive session in Graph-mode."""
  global sess
  try:
    tf.reset_default_graph()
    sess.close()
  except:
    pass
  sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=DEBUG_LOG))
  return sess


def tf_Variable(FEATURE_s, FEATURE_n, INIT):
    '''
    with tf.name_scope("amplitude"):
        amp_var = tf.Variable(initial_value=AMPLITUDE_INIT, name='amplitude',dtype=np.float64)
        amplitude = (np.finfo(np.float64).tiny + tf.nn.softplus(amp_var))
    '''
    with tf.name_scope(FEATURE_s):
        feature_var = tf.Variable(initial_value=INIT,
                                  dtype=np.float64,
                                  name=FEATURE_n)
        feature = (np.finfo(np.float64).tiny + tf.nn.softplus(feature_var))
    return feature_var, feature


def tf_Placeholder_assignments(feature_var,FEATUREPLH_s,FEATUREPLH_n, INIT):
    '''
    with tf.name_scope("amplitude_assign"):
        amp_p = tf.placeholder(shape=AMPLITUDE_INIT.shape, dtype=np.float64, name='amp_p')
        amplitude_assign = invert_softplus(amp_p, amp_var)
    '''
    with tf.name_scope(FEATUREPLH_s):
        feature_plh = tf.placeholder(shape=INIT.shape,
                                     dtype=np.float64,
                                     name=FEATUREPLH_n)
        feature_assign = invert_softplus(feature_plh, feature_var)
    return feature_plh, feature_assign


def tf_Placeholder_assignments_zeros(feature_var,FEATUREPLH_s,FEATUREPLH_n, SHAPE):
    with tf.name_scope(FEATUREPLH_s):
        feature_plh = tf.placeholder(shape=SHAPE,
                                     name=FEATUREPLH_n)
        feature_assign = invert_softplus(feature_plh, feature_var)
    return feature_plh, feature_assign


def create_cov_kernel(amp, lensc):

    kernel = tfkern.MaternOneHalf(amp, lensc) #ExponentiatedQuadratic # MaternOneHalf
    return kernel


def fit_gp(kernel, obs_idx_pts, obs_noise_var):
    gp = tfd.GaussianProcess(
        kernel=kernel,  # ([2,],[2,])
        index_points=obs_idx_pts,
        observation_noise_variance=obs_noise_var,
        validate_args=True)
    return gp


def tf_summary_scalar(FEATURESTRING, feature):
    return tf.summary.scalar(FEATURESTRING, feature)


def tf_train_gp_adam(feature, LEARNING_RATE):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(-feature)
    return train_op


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


def tf_summary_writer_projector_saver(sess, LOGDIR):
    summ = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())  # after the init
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    # projector_add(emb, writer)  # log_probability_embedding
    saver = tf.train.Saver()
    return summ, writer, saver


def tf_summary_writer_saver(sess, LOGDIR):
    summ = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())  # after the init
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    # tf.summary.trace_on(graph=True, profiler=True)
    saver = tf.train.Saver()
    return summ, writer, saver



def do_assign(sess, feature, feature_assign, feature_plh, feature_arr):
    [_, feature_] = sess.run([feature_assign, feature],
                                   feed_dict={feature_plh: feature_arr})
    return feature_


def tf_optimize_model_params(sess, num_iters, train_op, log_likelihood,
                             summ, writer, saver, LOGDIR, LOGCHECKPT,
                             obs_train_dataset,
                             obs_value_placeholder):
    # Now we optimize the model parameters.
    # Store the likelihood values during training, so we can plot the progress
    if isinstance(obs_train_dataset, pd.DataFrame):
        if obs_train_dataset.shape == 2:
            lls = np.zeros([num_iters + 1, 2], np.float64)
        if obs_train_dataset.ndims == 1:
            lls = np.zeros([num_iters + 1, 1], np.float64)
    elif isinstance(obs_train_dataset, np.ndarray):
        if len(obs_train_dataset.shape) == 2:
            lls = np.zeros([num_iters + 1, 2], np.float64)
        if len(obs_train_dataset.shape) == 1:
            lls = np.zeros([num_iters + 1, 1], np.float64)

    # initial test tun
    # [_, l, s] = sess.run([train_op, log_likelihood, summ])

    feed_obs_data = obs_train_dataset.reshape(obs_value_placeholder.shape)
    [a, b, c] = sess.run([train_op, log_likelihood, summ],
                         feed_dict={obs_value_placeholder: feed_obs_data})

    # train amplitude and length_scale
    for i in range(num_iters + 1):
        [_, lls[i], s] = sess.run([train_op, log_likelihood, summ],
                feed_dict={obs_value_placeholder: obs_train_dataset.reshape(obs_value_placeholder.shape)})
        writer.add_summary(s, i)
        if i % 200 == 0:
            saver.save(sess, os.path.join(LOGDIR, LOGCHECKPT), i)
    return lls


def create_meshgrid(pred_x, pred_y):
    '''
    Having trained the model, we'd like to sample from the posterior conditioned
    on observations. We'd like the samples to be at points other than the training
    inputs.
    :return: predictive_idx_pts
    '''
    # print("predx: ", pred_x.shape)
    # print("predy: ", pred_y.shape)
    # [2, 200,200]
    h = np.array(np.meshgrid(pred_x, pred_y, sparse=False))
    # print("h pred_ind", h.shape)
    # [2, 200,200] -> [200*200, 2]
    h2 = h.swapaxes(0, -1)
    h3 = h2.reshape(-1, 2)
    # print("h3 pred_ind", h3.shape)  # [200*200, 2]
    pred_idx_pts = h3
    # print("pred_ind", pred_idx_pts.shape)
    return pred_idx_pts # 50x50 meshgrid


def tf_gp_regression_model(kernel, pred_idx_pts, obs_idx_pts, obs, obs_noise_var, pred_noise_var):
    pred_idx_pts_Var = tf.cast(pred_idx_pts, dtype=tf.float64, name="pred_idx_pts_Var")
    obs_idx_pts_Var = tf.cast(obs_idx_pts, dtype=tf.float64, name="obs_idx_pts_Var")
    obs_Var = tf.cast(obs, dtype=tf.float64, name="obs_Var")
    obs_Var = tf.reshape(obs_Var, shape=[-1], name="obs_Var")

    obs_noise_Var = tf.Variable(obs_noise_var, dtype=tf.float64, name="obs_noise_var")
    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=pred_idx_pts_Var,
        observation_index_points=obs_idx_pts_Var,
        observations=obs_Var,
        observation_noise_variance=obs_noise_Var,
        predictive_noise_variance=pred_noise_var)
    return gprm


def distance_2d(p, a, n):
    '''
    p=point (1, 3),
    a=unit vector to BEGIN,
    n=line vector ( -1, 3)
    '''
    p = np.array(p)
    p=p.reshape(-1,2)
    t = (a-p) - np.dot((a-p), n.T)*n
    dist = np.sqrt(np.dot(t, t.T))
    return dist


def distance(p, a, n):
    '''
    p=point (1, 3),
    a=unit vector to BEGIN,
    n=line vector ( -1, 3)
    '''
    # assert np.array(p).shape[1] == 3
    # assert a.shape[1] == 3
    # assert n.shape[1] == 3
    t = (a-p) - np.dot((a-p), n)*n
    dist = np.sqrt(np.dot(t, t))
    return dist


def project_3Dpts_to_2Dline(pts, u):
    prjd_pts = np.array([u[0], u[1]])
    for p in pts:
        p = np.array([p[0], p[1]])
        prjd_pts = np.vstack((prjd_pts ,p))
    return prjd_pts


def capture_close_3d_pts(pts, u, v, eps):
    '''
    input pts can be any dimension, here we calc distance from first 2 dims
    distance in XY plane, however everything for Z axis
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#A_vector_projection_proof
    '''
    a = u # start vector
    l = dg.create_line(u, v) # line
    n = l / np.sqrt(np.dot(l, l)) # unit length vector
    sel_pts = np.zeros((0,3)) # points in epsilon distance
    for p in pts:
        p_distance = distance(p, a, n)
        if p_distance < eps:
            sel_pts = np.vstack((sel_pts,p))
    sel_pts = sel_pts.reshape(-1, 3)
    return sel_pts


def capture_close_3d_pts_with_2d_distance(pts, u, v, eps):
    '''
    input pts can be any dimension, here we calc distance from first 2 dims
    distance in XY plane, however everything for Z axis
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#A_vector_projection_proof
    '''
    a = np.array(u[0:2]).reshape(-1,2) # start vectors
    b = np.array(v[0:2]).reshape(-1,2)
    l = np.array(dg.create_line(a, b)) # line
    n = l / np.sqrt(np.dot(l, l.T)) # unit length vector
    sel_pts = np.zeros((0,3)) # points in epsilon distance

    for index, row in pts.iterrows(): # 3d
        i_xy = row[0:2].T
        p_distance = distance_2d(i_xy, a, n)
        if p_distance < eps:
            sel_pts = np.vstack((sel_pts,row.T))
    sel_pts = sel_pts.reshape(-1, 3) # 3rd D is observation, not coordinate
    return sel_pts # 3D


def capture_close_2d_distance_pts(pts, u, v, eps): # 3D input, 3D output, 2D distance calc
    '''
    input pts can be any dimension, here we calc distance from first 2 dims
    distance in XY plane, however everything for Z axis
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#A_vector_projection_proof
    '''
    assert pts.shape[1] < 4
    assert 1 < pts.shape[1]

    a = u[0:2] # start vector
    b = v[0:2]
    l = dg.create_line(a, b) # line
    n = l / np.sqrt(np.dot(l, l)) # unit length vector
    sel_pts = np.zeros((0,2)) # points in epsilon distance
    if pts.shape[1] >2:
        xy_pts = pts[:, 0:2]
        rest_pts = np.array(pts[:,2:].reshape(-1,1))
    else:
        xy_pts = pts[:, 0:1]
        # no rest points exist

    for p in xyt_pts:
        p_distance = distance(p[0:2], a, n)
        if p_distance < eps:
            sel_pts = np.vstack((sel_pts,p))
    sel_pts = sel_pts.reshape(-1, 2)
    if pts.shape[1] >2:
        sel_pts = np.append(sel_pts, rest_pts, axis=1)
    return sel_pts


def capture_close_pts_XY(pts, u, v, eps):
    '''
    distance in XY plane, however everything for Z axis
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#A_vector_projection_proof
    1. XYZ -> project to line. -> take Z'
    2. XYZ -> XY
    3. combine to get XYZ' where distance will only matter in XY
    4. see if XYZ' are in epsilon distance
    5. add original XYZ to sel_pts
    '''
    a = u # start vector
    l = dg.create_line(u, v) # line
    n = l / np.sqrt(np.dot(l, l)) # unit length vector

    t_pts = project_to_line_coordinates(pts,u,v)
    t_pts = np.array(t_pts[:,2])

    proj_pts = np.array(pts[:,0:2])
    e_pts = add_dimension(proj_pts, t_pts).reshape(-1,3)

    i=0
    sel_pts = np.zeros((0,3)) # points in epsilon distance
    for p in e_pts:
        p_distance = distance(p, a, n)
        if p_distance < eps:
            sel_pts = np.vstack((sel_pts,pts[i]))
        i+=1
    sel_pts = sel_pts.reshape(-1, 3)
    return sel_pts


def add_dimension(pts, x_): # up to 3 dimension input + 1 for output
    original_dim = pts.shape[1]
    ext_dim = original_dim + 1
    ext_pts = np.zeros((0, ext_dim))
    i =0
    for pt in pts:
        # ext_pt = np.array([it[0], it[1], it[2], x_])
        if pts.shape[1] == 3:
            ext_pt = np.array([pt[0], pt[1], pt[2], x_[i]])
        if pts.shape[1] == 2:
            ext_pt = np.array([pt[0], pt[1], x_[i]])
        if pts.shape[1] == 1:
            ext_pt = np.array([pt[0], x_[i]])
        ext_pts = np.vstack((ext_pts, ext_pt))
        i += 1
        # print("_")
    return ext_pts


def del_last_dimension(pts):
    original_dim = pts.shape[1]
    reduced_dim = original_dim - 1
    reduced_pts = np.zeros((0, reduced_dim))
    for pt in pts:
        assert pts.shape[1] > 1
        assert pts.shape[1] < 4
        if pts.shape[1] == 3:
            reduced_pt= np.array([pt[0] ,pt[1]])
        if pts.shape[1] == 2:
            reduced_pt= np.array([pt[0]])
        reduced_pts = np.vstack((reduced_pts, reduced_pt))
    return reduced_pts


def del_index_dimension(del_index, pts):
    original_dim = pts.shape[1]
    reduced_dim = original_dim - 1
    reduced_pts = np.zeros((0, reduced_dim))
    for pt in pts:
        if pts.shape[1] == 3:
            if del_index ==0:
                reduced_pt= np.array([pt[1] ,pt[2]])
            if del_index == 1:
                reduced_pt = np.array([pt[0], pt[2]])
            if del_index == 2:
                reduced_pt = np.array([pt[0], pt[1]])
        if pts.shape[1] == 2:
            if del_index == 0:
                reduced_pt = np.array([pt[1]])
            if del_index == 1:
                reduced_pt = np.array([pt[0]])
        reduced_pts = np.vstack((reduced_pts, reduced_pt))
    return reduced_pts


def project_to_line_coordinates(pts, u, v): # 3D to 2D coords on line
    proj_coords = np.zeros((0, 3))

    for x in pts:
        # cos_alpha = np.dot((v - u), (x - u)) / (np.linalg.norm(x - u) * np.linalg.norm(v - u))

        d = np.dot((v - u), (x - u)) / np.linalg.norm(v - u)
        projpt = u + d * (v - u) / np.linalg.norm(v - u)
        proj_coords = np.vstack((proj_coords, projpt))
    return proj_coords # 3D coords on the line


def project_2d_to_line_coordinates(pts, u, v):
    '''
    3D to 2D coords on line
    '''
    proj_coords = np.zeros((0, 2))
    u = np.array(u[:2])
    v = np.array(v[:2])
    pts_ = pts[:,:2]
    for x in pts_:
        # cos_alpha = np.dot((v - u), (x - u)) / (np.linalg.norm(x - u) * np.linalg.norm(v - u))
        d = np.dot((v - u), (x - u).T) / np.linalg.norm(v - u)
        projpt = u + d * (v - u) / np.linalg.norm(v - u)
        proj_coords = np.vstack((proj_coords, projpt))
    return proj_coords # 2D coords on the line


def create_1d_w_line_coords(proj_pts, u, v):
    '''
    from 2D coords, previously projected to line to 1D coord along section line
    :param pts: (num, 2) XY
    :param u: BEGIN of section cut
    :param v: END
    :return: 1D distances of each point projection
    '''
    # proj_pts = np.array(proj_pts).reshape(-1,2)
    proj_pts = np.array(proj_pts[:,:2])
    x_ = np.zeros((0, 1)) # 1D vstack to collect column of distances
    for p in proj_pts:
        a = np.array(p[0] - u[0])
        b = np.array(p[1] - u[1])
        dist = np.sqrt(a**2 + b**2)
        x_ = np.vstack((x_, dist))
    return x_ # 1D coords x_ new axis


def extend_pts_with_line_coord(pts, u ,v):
    '''
    take 3d pts XYZ, create X'Y' on line
    projected dimension_x() # calcuate new x_ from X'Y'
    return extended dim
    '''
    proj_pts_xy = project_2d_to_line_coordinates(pts, u, v) # 3D input, 2D output
    x_coords = create_1d_w_line_coords(proj_pts_xy, u, v) # 2D input, 1D output
    e_pts = add_dimension(pts, x_coords) # old pts & new x_ coord! # 3D input, 4D output
    # print(e_pts)
    return e_pts


def line_coords(pts, u, v):
    '''
    3D to 1D line coords
    '''
    pts = np.array(pts)
    u = np.array(u).T
    v = np.array(v).T
    proj_pts_xy = project_to_line_coordinates(pts, u, v) # 3D input, 2D output
    x_coords = create_1d_w_line_coords(proj_pts_xy, u, v) # 2D input, 1D output
    return x_coords


def create_line_coord_linspace(u, v, num): #1D
    lin_pts = np.linspace(u, v, num)
    return create_1d_w_line_coords(lin_pts, u, v)


def create_line_linspace(u, v, num): #2D
    return np.linspace(u, v, num)


def create_Z_sin_along_line_linspace(line_XY): #1D
    return sinusoid(line_XY)


def argmax_(A, A_bar, V, cov_vv):
    y_st = -1  # index of coordinate
    delta_st = -1

    for y in V:  # improved in algorithm 2
        if y in A:  # if y in A then dont consider: continue
            continue  # skip delta
        else:  # y is not in A, consider, calc delta.
            nom = nominator(y, A, cov_vv)
            denom = denominator(y, A_bar, cov_vv)
            if np.abs(denom) < 1e-8 or np.abs(nom) < 1e-8:
                delta_y = 0
            else:
                delta_y = nom / denom

        if delta_st < delta_y:
            delta_st = delta_y  # update largest delta yet, delta_star
            y_st = y  # update index

    return y_st, delta_st


def placement_algorithm_1(cov_vv, k):
    '''
    assume U is empty
    assume V = S, indexes where there may be sensors [0 .. cov_vv.shape[0])
    A - selection of placements eventually lngth = k
    y = len(V) + len(S) -> MOVE selections from V to S
    '''
    A = [] # selected_indexes
    V = np.linspace(0, cov_vv.shape[0]-1, cov_vv.shape[0], dtype=np.int64)
    A_bar = [] # complementer set to A.
    for i in V: # iterate indices where we can have sensors
        A_bar.append(i)

    while len(A) < k: # "for j=1 to k"
        y_st, _ = argmax_(A, A_bar, V, cov_vv) # choose y* mutual information which is the highest of the nom/denom in the formula given
        A.append(y_st) # add largest MI's corresponding index
        A_bar.remove(y_st) # remove it from complementer set
    return A # A contains k elements.


def placement_algorithm_2(cov_vv, k):
    '''
    assume U is empty
    assume V = S
    A - selection of placements eventually lngth = k
    y = len(V) + len(S) -> MOVE selections from V to S
    '''
    A = [] # selected_indexes
    V = np.linspace(0, cov_vv.shape[0]-1, cov_vv.shape[0], dtype=np.int64)
    A_bar = [] # complementer set to A.
    delta_all = []
    current_y_cont = []
    INF = 1e1000

    for i in V: # iterate indices where we can have sensors
        A_bar.append(i)
        delta_all.append(INF)
        current_y_cont.append(0)

    while len(A) < k: # "for j=1 to k"
        y_st = -1 # index of coordinate
        delta_st = -1
        # "S \ A do current_y <- false"
        while True:
            y_st, delta_st = argmax_(A, A_bar, V, cov_vv)
            if current_y_cont[y_st] == True:
                break
            current_y_cont[y_st] = True

        A.append(y_st) # add largest MI's corresponding index
        A_bar.remove(y_st) # remove it from complementer set

    return A # A contains k elements.


def nominator(y, A, cov_vv):
    A_ = A.copy()
    A_.append(y)
    s = slice(1,3)
    sigm_yy = make_slice(cov_vv, [y], [y])
    cov_yA = make_slice(cov_vv, [y], A_)
    cov_AA = make_slice(cov_vv, A_, A_)
    cov_Ay = make_slice(cov_vv, A_, [y])
    return sigm_yy - np.dot(np.dot(cov_yA, call_pinv(cov_AA)), cov_Ay)


def make_slice(cov_vv, y, A):
    cov_yA = np.zeros(shape=[len(y), len(A)])
    for i in range(cov_yA.shape[0]):
        for j in range(cov_yA.shape[1]):
            cov_yA[i,j] = cov_vv[y[i], A[j]]
    return cov_yA


def call_pinv(a):
    assert a.shape[0] == a.shape[1]
    if a.shape[0] == 1:
        return 1 / a
    else:
        r = np.linalg.pinv(a)
        return r


def denominator(y, A_hat, cov_vv):
    A_hat_ = A_hat.copy()
    A_hat_.remove(y)

    return nominator(y, A_hat_, cov_vv)


def calculate_running_time_algorithm_1(cov_vv, k):
    strt = time.time()
    pl = placement_algorithm_1(cov_vv, k)
    print("placement_array : ",pl)
    end = time.time()
    return (end - strt)


def calculate_running_time_algorithm_2(cov_vv, k):
    strt = time.time()
    pl = placement_algorithm_2(cov_vv, k)
    print("placement_array : ",pl)
    end = time.time()
    return (end - strt)


def create_dataframe_from_6d_dataset(dataset):
    trainA = {}
    trainA['a'] = dataset[:,0]
    trainA['b'] = dataset[:,1]
    trainA['c'] = dataset[:,2]
    trainA['d'] = dataset[:,3]
    trainA['e'] = dataset[:,4]
    trainA['f'] = dataset[:,5]
    trainA = pd.DataFrame(trainA)
    return trainA


def create_dataframe_from_4d_dataset(dataset):
    trainA = {}
    trainA['a'] = dataset[:,0]
    trainA['b'] = dataset[:,1]
    trainA['c'] = dataset[:,2]
    trainA['d'] = dataset[:,3]
    trainA = pd.DataFrame(trainA)
    return trainA


def normalize_feature(df):
    norm_df = (df - df.min()) / (df.max() - df.min())
    return norm_df


def mean_stdev_feature(norm_df):
    norm_df_mean = norm_df.mean()
    norm_df_std = norm_df.std()
    return norm_df_std, norm_df_mean


def standardize_normalized_feature(df):
    ''' Normalization = scale dataset between 0 - 1
        Standardization = center data around 0 wrt stdev
    '''
    norm_df = normalize_feature(df)
    norm_df_std, norm_df_mean = mean_stdev_feature(norm_df)
    norm_stdz_df = (norm_df - norm_df_mean)/norm_df_std
    return norm_stdz_df


def tf_normalize_and_standardize_feature(df):
    ''' Normalization = scale dataset between 0 - 1
        Standardization = center data around 0 wrt stdev
    '''
    norm_df = tf.Variable(tf.math.divide(tf.subtract(df, tf.reduce_min(df)), (tf.subtract(tf.reduce_max(df), tf.reduce_min(df)))))
    norm_df_mean, norm_df_var = tf.nn.moments(norm_df)
    norm_stdz_df = tf.Variable(tf.math.divide(tf.subtract(norm_df, norm_df_mean)),norm_df_var)
    return norm_stdz_df


def randomize_df(df, SELECT_ROW_NUM, dim1, dim2, dim3, dim4, dim5, dim6):
    # randomize index set.
    ds_idx_length = df.shape[0]
    ds_idx_arr = np.linspace(0, ds_idx_length - 1, ds_idx_length).T
    np.random.shuffle(ds_idx_arr)
    ds_r = np.array(ds_idx_arr[:SELECT_ROW_NUM])
    ds = df.iloc[ds_r, :]
    xyztpt_idx = ds[[dim1, dim2, dim3, dim4, dim5, dim6]]
    return xyztpt_idx


def load_randomize_select_train_test(xyzt_idx):
    ''' Load CSV data, randomize and divide to train and test sets
    '''
    SELECT_ROW_NUM = xyzt_idx.shape[0]
    train_dataset = xyzt_idx[:SELECT_ROW_NUM // 2, :]
    test_dataset = xyzt_idx[SELECT_ROW_NUM // 2:, :]
    return train_dataset, test_dataset


def separate_to_encode_dataset_and_tracer_dataset(xyztpt_idx):
    ''' divide to train and test sets
    '''
    INPUT_DIM_SHAPE = xyztpt_idx.shape[1]
    xyztp_idx = xyztpt_idx[:,:INPUT_DIM_SHAPE-1]
    t_idx = xyztpt_idx[:,INPUT_DIM_SHAPE-1:INPUT_DIM_SHAPE]
    return xyztp_idx, t_idx


def calculate_neg_log_likelihood(x, rv_x):
    return -rv_x.log_prob(x)


def create_model(encoded_size, INPUT_SHAPE):
    '''1. select and simplify model
    '''
    # input_shape = datasets_info.features['image'].shape

    # indep gaussian distribution, no learned parameters
    with tf.name_scope("VAE_"):
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                reinterpreted_batch_ndims=1)

        # encoder = normal Keras Sequential model, output passed to MultivatiateNormalTril()
        #   which splits activations from final Dense() layer into that is
        #   needed to spec mean and lower triangular cov matrix
        # Add KL div between encoder and prior to loss.
        # full covariance Gaussian distribution
        # mean and covariance matricies parameterized by output of NN
        # with tf.name_scope("encoder"):

        e_in_var = tfkl.InputLayer(input_shape=INPUT_SHAPE, name="e_input")


        encoder = tfk.Sequential(None, name="encoder_")
        encoder.add(e_in_var)
        # tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),

        encoder.add(tfkl.Dense(10, activation=tf.nn.leaky_relu, name="encoder_/e_dense_10"))
        encoder.add(tfkl.Dense(encoded_size, activation=tf.nn.leaky_relu, name="encoder_/e_dense_"))
        encoder.add(tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
                               activation=None, name="encoder_/e_mvn_dense_"))

        encoder.add(tfpl.MultivariateNormalTriL(
            encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior), name="encoder_/e_mvn_"))

        # encoder_var = tf.Variable(encoder.output, name="encoder_/encoder_var", shape=encoder.output.sample().shape[-1])

        # with tf.name_scope("decoder_"):
        decoder = tfk.Sequential(None, "decoder_")
        decoder.add(tfkl.InputLayer(input_shape=[encoded_size], name="decoder_/d_input_"))
        decoder.add(tfkl.Reshape([1, 1, encoded_size], name="decoder_/d_reshape_"))

        # tfkl.Dense(3, input_shape=(encoded_size,)),
        decoder.add(tfkl.Dense(10, activation=tf.nn.leaky_relu, name="decoder_/d_dense_10"))
        decoder.add(tfkl.Dense(INPUT_SHAPE, activation=tf.nn.leaky_relu, name="decoder_/d_dense_"))
        decoder.add(tfpl.IndependentBernoulli(INPUT_SHAPE, tfd.Bernoulli.logits, name="decoder_/d_Bern_"))
        # decoder.add(tfpl.MixtureNormal(INPUT_SHAPE, 0, name="decoder_/d_Bern_"))
        # decoder_var = tf.Variable(decoder.outputs, name="decoder_/decoder_var")

        vae = tfk.Model(inputs=encoder.input, outputs=decoder(encoder.output))

    # output defined as composition of encoder and decoder
    #   only need to specify the reconstruction loss: first term of ELBO

    '''Define model input and output
    '''

    # with tf.name_scope("model_"):
    # vae = tfk.Model(inputs=encoder, outputs = decoder, name="e_d_model_")

    # original input: x
    # output of model: rv_x (is random variable x)
    # loss function is negative log likelihood of
    #   the data given the model.
    # negloglik = lambda x, rv_x: -rv_x.log_prob(x)

    '''2. Train model
    '''
    # Configure model for training
    # strategy = tf.contrib.distribute.MirroredStrategy()
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=calculate_neg_log_likelihood)#, distribute=strategy)

    return vae, prior, encoder, decoder


def calc_H(XEDGES, YEDGES, lensc, lensc_assign, lensc_p, amp, amp_assign, amp_p, log_likelihood, sess, obs_values_placeholder, obs_train_dataset):

    H = np.zeros([XEDGES, YEDGES])
    for i in range(XEDGES):
        for j in range(YEDGES):
            lensc_v = [40 * np.double((1 + i) / XEDGES)]
            amp_v = [40 * np.double((1 + j) / YEDGES)]
            [_, _, L] = sess.run([lensc_assign, amp_assign, log_likelihood],
                                 feed_dict={lensc_p: lensc_v,  # 40/60
                                            amp_p: amp_v,
                                            obs_values_placeholder: obs_train_dataset.reshape(obs_values_placeholder.shape)})
            H[i, j] = L[0]
    return H


def calc_H_1d(XEDGES, YEDGES, lensc, lensc_assign, lensc_p, amp, amp_assign, amp_p, log_likelihood, sess):

    H = np.zeros([XEDGES, YEDGES])
    for i in range(XEDGES):
        for j in range(YEDGES):
            [_, _, L] = sess.run([lensc_assign, amp_assign, log_likelihood],
                                 feed_dict={lensc_p: [2 * np.double((1 + i) /
                                                                    XEDGES)],
                                            amp_p: [2 * np.double((1 + j) / YEDGES)]})
            H[i, j] = L[0]
    return H


# def gp_train_and_hyperparameter_optimize(amp, amp_assign, amp_p,
#                                          lensc, lensc_assign, lensc_p,
#                                          emb, xy_idx, t_idx, obs_noise_var,
#                                          LEARNING_RATE,NUM_ITERS, LOGDIR, LOGCHECKPT, sess):
#
#
#     return kernel, log_likelihood, lls, saver, amp, lensc, obs_noise_var


def split_to_test_train(idx):
    num_rows = idx.shape[0]
    if num_rows % 2 != 0:
        num_rows -=1
    div = num_rows // 2
    train_dataset = idx[:div, :]
    test_dataset = idx[div:2*div, :]
    return train_dataset, test_dataset


def get_tracers_for_coordloc(i0, i1, i2, grid_pt, encoder, sess):
    # generate 300 x xyztp values for this spatial location
    tracers_at_location = np.array([])
    xyz_ = np.array([i0, i1, i2])
    for i_pt in range(grid_pt.shape[0]):  # 300 in linspace
        p_ = grid_pt[i_pt,0]
        t_ = grid_pt[i_pt,1]
        xyzt_ = np.append(xyz_, p_)
        xyztp_ = np.append(xyzt_, t_)
        n_xyztp_ = standardize_normalized_feature(pd.DataFrame(xyztp_))
        n_xyztp_ = np.array(n_xyztp_).reshape(-1,5)

        en_xyztp_ = encoder(n_xyztp_)
        en_xyztp_s = en_xyztp_.sample()
        pred_tracer = sess.run(en_xyztp_s )
        # pred_tracer = gprm.sample(xyztp)
        tracers_at_location = np.append(tracers_at_location, pred_tracer)  # this is what we will use later
    return tracers_at_location



    # generate 300 x xyztp values for this spatial location
    # tracers_at_location = np.array([])
    # xyz_ = np.array([i0, i1, i2])
    # xyz_ = tf.stack([i0, i1, i2], axis=0, name="xyz")
    # tracers_at_location = tf.Variable([], name="tracers_at_location")
    # i = tf.constant(0)
    # cond_ = lambda i, iters : tf.less(i, iters_)
    # def body_(i, iters_):

    # iters = tf.constant(loc_xyz.shape[0])
    # p_ = tf.slice()
    # p_ = tf.gather_nd(grid_pt, [iters, 0, 0])
    # t_ = tf.gather_nd(grid_pt, [i, 1, 0])

    # x_ = tf.gather_nd(linsp_x, [0:i])
    # y_ = tf.gather_nd(linsp_y, [0:i])
    # z_ = tf.gather_nd(linsp_z, [0:i])
    # norm_x_ = graph_normalize_with_mean_stdev(mean_x, stdev_x, linsp_x)  # x_)
    # norm_y_ = graph_normalize_with_mean_stdev(mean_y, stdev_y, linsp_y)  # y_)
    # norm_z_ = graph_normalize_with_mean_stdev(mean_z, stdev_z, linsp_z)  # z_)
    # norm_t_ = graph_normalize_with_mean_stdev(mean_t, stdev_t, t_)
    # norm_p_ = graph_normalize_with_mean_stdev(mean_p, stdev_p, p_)
    # make grid_pt linear

def graph_get_tracers_for_coordloc(loc_xyz, vec_pt, encoder):

    # p, t = tf.linspace(1., 3., 3), tf.linspace(4., 7., 4)  # [1, 2, 3] , [4, 5, 6]
    # P, T = tf.meshgrid(p, t)
    #
    # st2 = tf.stack([P, T], axis=2)   #  [[[1,4],[1,5],[1,6]],[[2,4],[2,5],[2,6]], ...]  # st[0,0] == [1, 4]
    # vec_grid_pt = tf.reshape(st2, [-1, 2])

    xyz_ = tf.constant([-3., -2.99, -2.98])

    fx = tf.fill(dims=[vec_pt.shape[0], 1], value=loc_xyz[0])
    fy = tf.fill(dims=[vec_pt.shape[0], 1], value=loc_xyz[1])
    fz = tf.fill(dims=[vec_pt.shape[0], 1], value=loc_xyz[2])
    t_ = tf.slice(vec_pt, [0, 1], [vec_pt.shape[0], 1])
    p_ = tf.slice(vec_pt, [0, 0], [vec_pt.shape[0], 1])

    fx_ = tf.reshape(fx, shape=[-1], name="fx")
    fy_ = tf.reshape(fy, shape=[-1], name="fy")
    fz_ = tf.reshape(fz, shape=[-1], name="fz")
    ft_ = tf.reshape(t_, shape=[-1], name="ft_")
    fp_ = tf.reshape(p_, shape=[-1], name="fp_")

    fxyztp = tf.stack([fx_, fy_, fz_, fp_, ft_], axis=1, name="fxyztp")

    en_xyztp_ = encoder(fxyztp)
    tr_sample = en_xyztp_.sample()

    # grid_pt_2 = tf.reshape(vec_pt, shape=[-1,2])
    # num = tf.shape(grid_pt_2)[0]
    # gridlen = tf.constant([3])
    # loc_xyz_mult = tf.reshape(tf.tile(loc_xyz, gridlen))
    # n_xyztp = tf.stack([loc_xyz_mult, vec_pt], axis=0, name="normalized_xyztp")
    # n_xyztp_ = tf.reshape(n_xyztp, [-1, 5], name="reshape_")
    # en_xyztp_ = encoder(n_xyztp_)
    # tr_sample = en_xyztp_.sample()

    return tr_sample


def graph_get_vgp_input_xyztp(loc_xyz, vec_pt):
    # xyz_ = tf.constant([-3., -2.99, -2.98])
    dim0 = vec_pt.shape[0]
    val_0 = loc_xyz[0]
    val_1 = loc_xyz[1]
    val_2 = loc_xyz[2]
    fx = tf.fill(dims=[dim0, 1], value=val_0)
    fy = tf.fill(dims=[dim0, 1], value=val_1)
    fz = tf.fill(dims=[dim0, 1], value=val_2)

    fx_ = tf.cast(tf.reshape(fx, shape=[-1], name="fx"), dtype=tf.float64)
    fy_ = tf.cast(tf.reshape(fy, shape=[-1], name="fy"), dtype=tf.float64)
    fz_ = tf.cast(tf.reshape(fz, shape=[-1], name="fz"), dtype=tf.float64)

    t_ = tf.slice(vec_pt, [0, 1], [dim0, 1])
    p_ = tf.slice(vec_pt, [0, 0], [dim0, 1])
    ft_ = tf.cast(tf.reshape(t_, shape=[-1], name="ft_"), dtype=tf.float64)
    fp_ = tf.cast(tf.reshape(p_, shape=[-1], name="fp_"), dtype=tf.float64)

    fxyztp = tf.stack([fx_, fy_, fz_, ft_, fp_], axis=1, name="fxyztp")

    return fxyztp


def create_cov_matrix(minmax_x,minmax_y,minmax_z,minmax_pressure, minmax_temperature, SPATIAL_COVER, SPATIAL_COV_PR_TEMP, encoder, sess):

    linsp_x = np.linspace(minmax_x[0], minmax_x[1], SPATIAL_COVER)
    linsp_y = np.linspace(minmax_y[0], minmax_y[1], SPATIAL_COVER)
    linsp_z = np.linspace(minmax_z[0], minmax_z[1], SPATIAL_COVER)
    mesh_xyz = np.zeros((linsp_x.shape[0], linsp_y.shape[0], linsp_z.shape[0]))

    linsp_p = np.linspace(minmax_pressure[0], minmax_pressure[1], SPATIAL_COV_PR_TEMP).reshape(-1,1)
    linsp_t = np.linspace(minmax_temperature[0], minmax_temperature[1], SPATIAL_COV_PR_TEMP).reshape(-1,1)
    X, Y = np.meshgrid(linsp_p, linsp_t)
    grid_pt = np.array([X.flatten(), Y.flatten()]).T

    I0 = mesh_xyz.shape[0]
    I1 = mesh_xyz.shape[1]
    I2 = mesh_xyz.shape[2]
    cov_vv = np.zeros((I0*I1*I2, I0*I1*I2))
    for i0 in range(I0): # take loc i
        for i1 in range(I1):
            for i2 in range(I2):
                tracers_loc_i = get_tracers_for_coordloc(i0, i1, i2, grid_pt, encoder, sess)
                print(".")
                for j0 in range(I0): # take loc j
                    for j1 in range(I1):
                        for j2 in range(I2): # 2 spatial coord indices
                            tracers_loc_j = get_tracers_for_coordloc(j0, j1, j2, grid_pt, encoder, sess)
                            print("_")
                            i = i0 + i1 * I0 + i2 * I0 * I1 # calc index of location i
                            assert (0 <= i)
                            assert (i < I0 * I1 * I2)

                            j = j0 + j1 * I0 + j2 * I0 * I1 # calc index of location j
                            assert (0 <= j)
                            assert (j < I0 * I1 * I2)

                            # get a: i0, i1, i2 -> 300
                            # get b: j0, j1, j2 -> 300 (tracer)
                            cov_vv[i, j] = np.cov(tracers_loc_i, tracers_loc_j, bias=True)[0,1] # i= index of a location(0-399), 300 tracer vector covary with another -> get 1 number
                            cov_vv[j, i] = cov_vv[i, j] # symmetric
    return cov_vv


def create_cov_matrix_while_loops(minmax_x,minmax_y,minmax_z,minmax_pressure, minmax_temperature, SPATIAL_COVER, SPATIAL_COV_PR_TEMP, encoder, sess):

    linsp_x = np.linspace(minmax_x[0], minmax_x[1], SPATIAL_COVER)
    linsp_y = np.linspace(minmax_y[0], minmax_y[1], SPATIAL_COVER)
    linsp_z = np.linspace(minmax_z[0], minmax_z[1], SPATIAL_COVER)
    mesh_xyz = np.zeros((linsp_x.shape[0], linsp_y.shape[0], linsp_z.shape[0]))

    linsp_p = np.linspace(minmax_pressure[0], minmax_pressure[1], SPATIAL_COV_PR_TEMP).reshape(-1,1)
    linsp_t = np.linspace(minmax_temperature[0], minmax_temperature[1], SPATIAL_COV_PR_TEMP).reshape(-1,1)
    X, Y = np.meshgrid(linsp_p, linsp_t)
    grid_pt = np.array([X.flatten(), Y.flatten()]).T

    I0 = mesh_xyz.shape[0]
    I1 = mesh_xyz.shape[1]
    I2 = mesh_xyz.shape[2]
    cov_vv = np.zeros((I0*I1*I2, I0*I1*I2))

    i0 = 0
    while i0 < I0:
    # for i0 in range(I0): # take loc i
        i1 = 0
        while i1 < I1:
        # for i1 in range(I1):
            i2 = 0
            while i2 < I2:
            # for i2 in range(I2):
                tracers_loc_i = get_tracers_for_coordloc(i0, i1, i2, grid_pt, encoder, sess)
                print(".")
                j0 = 0
                while j0 < I0:
                # for j0 in range(I0): # take loc j
                    j1 = 0
                    while j1 < I1:
                    # for j1 in range(I1):
                        j2 = 0
                        while j2 < I2:
                        # for j2 in range(I2): # 2 spatial coord indices
                            tracers_loc_j = get_tracers_for_coordloc(j0, j1, j2, grid_pt, encoder, sess)
                            print("_")
                            i = i0 + i1 * I0 + i2 * I0 * I1 # calc index of location i
                            assert (0 <= i)
                            assert (i < I0 * I1 * I2)

                            j = j0 + j1 * I0 + j2 * I0 * I1 # calc index of location j
                            assert (0 <= j)
                            assert (j < I0 * I1 * I2)

                            # get a: i0, i1, i2 -> 300
                            # get b: j0, j1, j2 -> 300 (tracer)
                            cov_vv[i, j] = np.cov(tracers_loc_i, tracers_loc_j, bias=True)[0,1] # i= index of a location(0-399), 300 tracer vector covary with another -> get 1 number
                            cov_vv[j, i] = cov_vv[i, j] # symmetric
                            j2+=1
                        j1+=1
                    j0+=1
                i2+=1
            i1+=1
        i0+=1
    return cov_vv


def tf_create_cov_matrix(mm_x,mm_y,mm_z,mm_p, mm_t, sptl_const, sptl_pt_const, encoder, sess):
    '''

    '''
    linsp_x = np.linspace(mm_x[0], mm_x[1], sptl_const)
    linsp_y = np.linspace(mm_y[0], mm_y[1], sptl_const)
    linsp_z = np.linspace(mm_z[0], mm_z[1], sptl_const)
    mesh_xyz = np.zeros((linsp_x.shape[0], linsp_y.shape[0], linsp_z.shape[0]))

    linsp_p = np.linspace(mm_p[0], mm_p[1], sptl_pt_const).reshape(-1,1)
    linsp_t = np.linspace(mm_t[0], mm_t[1], sptl_pt_const).reshape(-1,1)
    X, Y = np.meshgrid(linsp_p, linsp_t)
    grid_pt = np.array([X.flatten(), Y.flatten()]).T

    I0 = mesh_xyz.shape[0]
    I1 = mesh_xyz.shape[1]
    I2 = mesh_xyz.shape[2]
    cov_vv = np.zeros((I0*I1*I2, I0*I1*I2))

    i0 = 0
    while i0 < I0:
    # for i0 in range(I0): # take loc i
        i1 = 0
        while i1 < I1:
        # for i1 in range(I1):
            i2 = 0
            while i2 < I2:
            # for i2 in range(I2):
                tracers_loc_i = get_tracers_for_coordloc(i0, i1, i2, grid_pt, encoder, sess)
                print(".")
                j0 = 0
                while j0 < I0:
                # for j0 in range(I0): # take loc j
                    j1 = 0
                    while j1 < I1:
                    # for j1 in range(I1):
                        j2 = 0
                        while j2 < I2:
                        # for j2 in range(I2): # 2 spatial coord indices
                            tracers_loc_j = get_tracers_for_coordloc(j0, j1, j2, grid_pt, encoder, sess)
                            print("_")
                            i = i0 + i1 * I0 + i2 * I0 * I1 # calc index of location i
                            assert (0 <= i)
                            assert (i < I0 * I1 * I2)

                            j = j0 + j1 * I0 + j2 * I0 * I1 # calc index of location j
                            assert (0 <= j)
                            assert (j < I0 * I1 * I2)

                            # get a: i0, i1, i2 -> 300
                            # get b: j0, j1, j2 -> 300 (tracer)
                            cov_vv[i, j] = np.cov(tracers_loc_i, tracers_loc_j, bias=True)[0,1] # i= index of a location(0-399), 300 tracer vector covary with another -> get 1 number
                            cov_vv[j, i] = cov_vv[i, j] # symmetric
                            j2+=1
                        j1+=1
                    j0+=1
                i2+=1
            i1+=1
        i0+=1
        sess.run()
    return cov_vv


def graph_normalization_factors_from_training_data(sample_len, col_len, batch_cnt=1):

    # tf.reset_default_graph()# Save the variables to disk.
    # for f in FEATURES:
    #     with tf.name_scope(f):

    mean_var = tf.Variable(initial_value=tf.zeros(shape=[col_len], dtype=tf.float64),
                       shape=[col_len],
                       dtype=tf.float64,
                       name='mean')
    stdev_var = tf.Variable(initial_value=tf.ones(shape=[col_len], dtype=tf.float64),
                        shape=[col_len],
                        dtype=tf.float64, name='stdev')

    values = tf.placeholder(shape=[sample_len],
                            dtype=tf.float64, name="input_values")
    mean = tf.reduce_mean(values, axis=[0])
    mean_var2 = mean_var.assign([mean])
    centered = tf.subtract(values, mean_var2)
    stdev = tf.math.reduce_std(centered, axis=[0])
    stdev_var2 = stdev_var.assign([stdev])
    normal = tf.divide(centered, stdev_var2)

    return values, normal, mean_var, stdev_var


def graph_normalize_with_mean_stdev(mean, stdev, col):
    centered = tf.Variable(tf.subtract(col, mean), name="centered")
    normal = tf.Variable(tf.divide(centered, stdev), name="normal")
    return normal


def slice_grid_xyz(i0, i1, i2, linsp_x, linsp_y, linsp_z):
    # i0 is the index we want from linsp_x
    x_ = tf.slice(linsp_x, begin=[i0],size=[1],name="x_")
    y_ = tf.slice(linsp_y, begin=[i1], size=[1], name="y_")
    z_ = tf.slice(linsp_z, begin=[i2], size=[1], name="z_")
    xyz_ = tf.stack([x_, y_, z_], name="xyz_")
    xyz_ = tf.reshape(xyz_, shape=[-1])
    # xyz = tf.Variable([x_, y_, z_])
    return xyz_


def py_get_coord_idxs(sel_idx, xyz_idxs):

    sel_coord = np.reshape(xyz_idxs[sel_idx[0], :] ,[1,3])
    i = 1
    while i < 7:
        coord_i = sel_idx[i]
        r_i = xyz_idxs[coord_i, :]
        sel_coord = np.vstack((sel_coord, r_i))  # append 3 coord to sel

        i+=1
    return sel_coord
    pass

def denormalize_coord(sel_norm_coord):
    stdev_var = 0.0007434639347162126 * 3000000
    mean_var = 0.0018159087825037148

    for i in range(sel_norm_coord.shape[0]):
       for j in range(3):
           sel_norm_coord[i,j] = sel_norm_coord[i,j] * stdev_var
           sel_norm_coord[i,j] = sel_norm_coord[i,j] + mean_var

    return sel_norm_coord