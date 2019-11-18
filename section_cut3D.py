import numpy as np
import tensorflow as tf

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()
import matplotlib.pyplot as plt

from toy_models import gp_plot_sinusoid3D as gpsin

#%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
#%config InlineBackend.figure_format = 'png'
print(tf.__version__)

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

RANGE_PTS=[-2, 2]
NUM_PTS=100
NUM_FEATURES=3 # dimensions
BEGIN=np.array([-2,-2,0])
END=np.array([2,1,0])
EPSILON=1.
TEST_FN_PARAM = np.array([3 * np.pi, np.pi])
NUM_TRAINING_POINTS = 180
RANDOM = False
SIN_PT_GEN = True
# Suppose we have 2d points and corresponding sin values. -> prev plot.
# A line in XY plane (from BEGIN to END, say) will hold our section cut.
# To generate plt.x coordinates along the line, calculate the selected point's
#  projection to the line and then, the distance of the projected point from BEGIN (BEGIN is a point on the line too).
# To represent the sin fn along the line, plot the plt.x, sinusoid(X) relationship, where X is the original 2d (higher)
#  dimension input point
# This is independent of the input space's dimensionality. (wiki)
# The purpose of the graph is to see (visually check) that the fitting was successful, so the
#  generated predictions by the GP are not so far away from the training points and the confidence intervals are not too
#  noisy either.
# Also would work to display the fitted kernel parameters, where the kernel param dimensionality is 4 for
#  the 2d case already.

# TODO
# How can we take a sample along the XY line. Conditioning?
# PLOT x = how far(length of projection) are the points to a BEGIN point to projection

#########################################################
# SECTION CUT STRATEGY
# we have:
#   x,y,z = original x axis
#   3d pts XYZ (num, XYZ)
#   2d line (num, XY)
#   sel_pts (num, XY)
#   sel_line_pts (num, X'Y')
#   BEGIN = coordinate along wall to take section
#   END = coordinate along another wall
#   u = BEGIN
#   v = END
#   line = axis along section cut, defined by BEGIN and END

#   linedist_x = x'' = distance from BEGIN along line axis
#   sel_line_x_pts (num, X''0) # change of reference.
#   seldist_pts = sel_line_x_pts (num, XYZ X'' ) #TODO add another dimension
#
#   plot original 3d pts XYZ along linedist_x = X'', y = sinusoid(x)

# NEED:
#   l_pts = points along line to plot.
#   sinusoid(l_pts)
#########################################################

def generate_random_points(num_pts, range_val):
    pts = np.random.uniform(range_val[0], range_val[1], size=(num_pts, 3)) # (100,3)
    return pts

def create_line(u, v):
    line = u-v # == [begin[0]-end[0], begin[1]-end[1], begin[2]-end[2]]
    return line

## 3. find points epsilon distance to this plane
## IN: all points, plane, epsilon
## OUT: selected points
def distance(p, a, n): #p=point, a=unit vector to BEGIN ,n=line vector
    t = (a-p) - np.dot((a-p), n)*n
    dist = np.sqrt(np.dot(t, t))
    return dist

def project_3Dpts_to_2Dline(pts, u):
    prjd_pts = np.array([u[0], u[1]])
    for p in pts:
        p = np.array([p[0], p[1]])
        prjd_pts = np.vstack((prjd_pts ,p))
    return prjd_pts

def capture_close_pts(pts,u=BEGIN, v=END, eps=EPSILON):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#A_vector_projection_proof
    a = u # start vector
    l = create_line(u, v) # line
    n = l / np.sqrt(np.dot(l, l)) # unit length vector
    sel_pts = np.array([]) # points in epsilon distance
    for p in pts:
        p_distance = distance(p, a, n)
        if p_distance < eps:
            sel_pts = np.concatenate((sel_pts,p))
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
            ext_pt = np.array([pt[0] ,pt[1] ,pt[2], x_[i]])
        if pts.shape[1] == 2:
            ext_pt = np.array([pt[0], pt[1], x_[i]])
        if pts.shape[1] == 1:
            ext_pt = np.array([pt[0], x_[i]])
        ext_pts = np.vstack((ext_pts, ext_pt))
        i += 1
    return ext_pts

def del_last_dimension(pts):
    original_dim = pts.shape[1]
    reduced_dim = original_dim - 1
    reduced_pts = np.zeros((0, reduced_dim))
    for pt in pts:
        if pts.shape[1] == 3:
            reduced_pt= np.array([pt[0] ,pt[1]])
        if pts.shape[1] == 2:
            reduced_pt= np.array([pt[0]])
        reduced_pts = np.vstack((reduced_pts, reduced_pt))
    return reduced_pts


def project_to_line_coordinates(pts, u=BEGIN, v=END): # 3D to 2D coords on line
    proj_coords = np.zeros((0, 3))
    for x in pts:
        # alpha = x - u - v
        cos_alpha = np.dot((v - u), (x - u)) / (np.linalg.norm(x - u) * np.linalg.norm(v - u))
        # d =  np.linalg.norm(x-u) * cos_alpha
        d = np.dot((v - u), (x - u)) / np.linalg.norm(v - u)
        projpt = u + d * (v - u) / np.linalg.norm(v - u)
        # print("ppt.shape : ", ppt)
        # print("cos_alpha.shape : ", cos_alpha)
        # print("d.shape : ", d)
        # print("proj_pts.shape : ", proj_pts)
        proj_coords = np.vstack((proj_coords, projpt))
    # proj_coords.reshape(0,2)
    return proj_coords # 3D coords on the line

# def create_line_coordinate_x(pts, u=BEGIN, v=END): # 2D coords to 1D new x_ axis
#     x_ = np.zeros((0, 2)) # 1D
#     l = create_line(u, v)
#
#     for i in pts:
#         print(i)
#         print(u)
#         d = np.array((v-i))
#         print("d :", d)
#         # d_= np.zeros((0, p))
#         d_ = np.array((d[0],d[1]))
#         print("d_ :", d_)
#
#         x_ = np.vstack((x_, d_))
#         print(x_)
#     return x_ # 1D coords x_ new axis

def create_line_coordinate_x(pts, u=BEGIN, v=END): # 2D coords to 1D new x_ axis
    x_ = np.zeros((0, 1)) # 1D

    for i in pts:
        print(i)
        print(u)
        d = np.array((v-i))
        dist = np.sqrt(d[0]**2 + d[1]**2)
        print("d :", d)
        # d_= np.zeros((0, p))
        # d_ = np.array((d[0],d[1]))
        # print("d_ :", d_)

        x_ = np.vstack((x_, dist))
        print(x_)
    return x_ # 1D coords x_ new axis



def extend_pts_with_line_coord(pts):
    # take 3d pts XYZ, create X'Y' on line
    # projected dimension_x() # calcuate new x_ from X'Y'
    # return extended dim

    proj_pts_xy = project_to_line_coordinates(pts) # 3D input, 2D output
    x_coords = create_line_coordinate_x(proj_pts_xy) # 2D input, 1D output

    e_pts = add_dimension(pts, x_coords) # old pts & new x_ coord! # 3D input, 4D output
    print(e_pts)
    return e_pts

if RANDOM ==True:
    pts_rand = generate_random_points(NUM_PTS, RANGE_PTS)  # 3d pointcloud
    pts_rand[:,2] *=0.5 # transform length of 3rd scale.
    sel_pts_rand = capture_close_pts(pts_rand, BEGIN, END)

obs_idx_pts, obs = gpsin.generate_2d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=0.001)

obs = obs.reshape(-1,1)
print(obs_idx_pts.shape, obs.shape)
print("----------")
pts = add_dimension(obs_idx_pts, obs)
print(pts)
# pts =
sel_pts = capture_close_pts(pts, BEGIN, END)
ext_sel_pts = extend_pts_with_line_coord(sel_pts)

print("sel_pts.shape : ", sel_pts.shape)
print("ext_sel_pts.shape : ", ext_sel_pts.shape)
print("sel_pts : ", sel_pts)
print("ext_sel_pts[:,3]: ", ext_sel_pts[:,3])

# print(ext_sel_pts)
# add projected x'' dimension to sel_pts
# print(xy_)
# print(xy_.shape)

'''
## PLOTS ##
'''

def plot_XYline_pointdistance(pts, sel_pts): # epsilon pts colour # sin true curve, GP and confidence intervals
    # line on XY plane and all points in space
    plt.figure(figsize=(12, 7))
    plt.scatter(pts[:,0], pts[:,1])
    plt.scatter(sel_pts[:,0], sel_pts[:,1])
    plt.scatter(BEGIN[0], BEGIN[1])
    plt.scatter(END[0], END[1])
    plt.plot([BEGIN[0],END[0]], [BEGIN[1],END[1]], ls="--", c=".4")
    plt.show()

plot_XYline_pointdistance(pts, sel_pts)

def plot_alongline(sel_pts, u=BEGIN, v=END):
    plt.figure(figsize=(12, 7))
    projsel_pts = project_to_line_coordinates(sel_pts)
    plt.scatter(projsel_pts[:,0], projsel_pts[:,1], color='orange')
    plt.plot([u[0],v[0]], [u[1],v[1]], ls="--", c=".4")
    plt.show()

plot_alongline(sel_pts)

s = np.linspace(-2,2,120)
print("s : ", s)
s=s.reshape(-1, 1)
print("s : ", s)

#TODO
# x should be from 3d sinusoid data, not random gen one!

def plot_new_x_pts(x,  u=BEGIN, v=END):
    # x = section line
    # s = y values output from sin fn
    plt.figure(figsize=(12, 7))
    # plt.scatter(x[:,0], x[:,1], color='orange')
    # plt.scatter(x[:,0], sinusoid(x), color='orange')
    # length = x.shape[0]
    # s = s[:length]
    # s = gpsin.sinusoid(s)
    # print("s : ", s)
    # ordered list of the x axis, will not link well with original coordinate ordering
    # z = np.sort(x[:,3])
    # print("z : ", z)
    # pass
    # plt.scatter(z, s, color='orange')
    # PLOT sinusoid
    num = x.shape[0]
    l = np.linspace(u, v, num=10000)
    print("l : ", l)
    e = np.zeros((0, 1))
    l_len = l.shape[0]-1
    l = del_last_dimension(l)
    # for k in range(l_len):
    e = gpsin.sinusoid(l)
    e = e.reshape(-1,1)
        # l_ = np.sin(x[k,0])*np.sin(x[k,1])
    # e = np.vstack((e, l_))
        # print("_")
    # print("l_ : ", l_)
    plt.scatter(x[:,3], x[:,2], color='orange') #observations
    plt.scatter(l[:,0],e, color='red', s=0.1) #sin cut
    plt.show()
plot_new_x_pts(ext_sel_pts,BEGIN, END)


def plot_sin_and_sel_pts_at_section(x_new, pts,x_original):
    plt.figure(figsize=(12, 4))
    plt.scatter(x_new[:,0], x_new[:,1], color='orange')
    # plt.plot([u[0],v[0]], [u[1],v[1]], ls="--", c=".4")
    # print("x_new : ", x_new)
    # print("x_original : ", x_original)
    y = np.array(gpsin.sinusoid(x_new))
    plt.scatter(pts[:,0], y,label='True fn')
    x_ = np.linspace(0, 60, 500)
    sin_y= np.sin(x_)
    plt.plot(x_, sin_y)
    # plt.scatter(observation_index_points_[:, axis_k], observations_,label='Observations')
    # for i in range(num_samples):
    #     plt.plot(predictive_index_points_[:, axis_k], samples_[i, axis_k, :].T, c='r', alpha=.1,
    #              label='Posterior Sample' if i == 0 else None)
    plt.show()

# gpsin.plot_samples2D(12,4,1, \
#                      gpsin.predictive_index_points_,
#                      gpsin.observation_index_points_,
#                      gpsin.observations_)



# plot_sin_and_sel_pts_at_section(projsel_line_x_pts ,pts, x_original)
## GP FIT AND PLOT ##
#... GP along selected lines ...


# def plot_samples2D(figx, figy, axis_k):
#     plt.figure(figsize=(figx, figy))
#     plt.plot(predictive_index_points_[:, axis_k], gp_plot_sinusoid3D.sinusoid(predictive_index_points_),label='True fn')
#     plt.scatter(observation_index_points_[:, axis_k], observations_,label='Observations')
#     for i in range(num_samples):
#         plt.plot(predictive_index_points_[:, axis_k], samples_[i, axis_k, :].T, c='r', alpha=.1,
#                  label='Posterior Sample' if i == 0 else None)
#     leg = plt.legend(loc='upper right')
#     for lh in leg.legendHandles:
#         lh.set_alpha(1)
#     plt.xlabel(r"Index points ($\mathbb{R}^1$)")
#     plt.ylabel("Observation space")
#     # plt.show()
