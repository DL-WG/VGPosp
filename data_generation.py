import numpy as np
import tensorflow as tf
from matplotlib import cm

import matplotlib.pyplot as plt
import os
#%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'

###########################################################
# FILE IMPORTS
import gp_functions as gpf


###########################################################
# FUNCTIONS
#TODO
##csv
# pd.Dataframe(("Time": time, ... ))


def generate_2D_data(num_train_pts, coord_range):
    """Generate noisy sinusoidal observations at a random set of points.
    Returns:
       obs_idx_pts: XY (num, 2)
    """
    idx_pts = np.random.uniform(0., 1., (num_train_pts, 2)) # XY
    assert(idx_pts.shape == (num_train_pts, 2))
    idx_pts = idx_pts.astype(np.float64)
    x_scale = coord_range[0][1] - coord_range[0][0]
    idx_pts[:, 0] *= x_scale
    idx_pts[:, 0] += coord_range[0][0]
    y_scale = coord_range[1][1] - coord_range[1][0]
    idx_pts[:, 1] *= y_scale
    idx_pts[:, 1] += coord_range[1][0]
    return idx_pts


def generate_1D_data(num_train_pts, coord_range):
    idx_pts = np.random.uniform(0., 1., (num_train_pts))  # X
    assert (idx_pts.shape[0] == (num_train_pts))
    idx_pts = idx_pts.astype(np.float64)

    x_scale = coord_range[1] - coord_range[0]
    idx_pts *= x_scale
    idx_pts += coord_range[0]

    return idx_pts


def generate_noisy_2Dsin_data(num_train_pts, obs_noise_variance, coord_range):
    '''
    returns: idx_pts:XY (num, 2), obs: Z (num, 1)
    '''
    idx_pts = generate_2D_data(num_train_pts, coord_range)
    noise = random_noise(0, obs_noise_variance, num_train_pts)
    assert (noise.shape[0] == (num_train_pts))
    obs = (gpf.sinusoid(idx_pts) + noise)  # y = f(x) + noise
    return idx_pts, obs


def generate_noisy_1Dsin_data(num_train_pts, obs_noise_variance, coord_range):
    '''
    returns: idx_pts:XY (num, 2), obs: Z (num, 1)
    '''
    idx_pts = generate_1D_data(num_train_pts, coord_range)
    noise = random_noise(0, obs_noise_variance, num_train_pts)
    assert (noise.shape[0] == (num_train_pts))
    sin_vals = gpf.sinusoid(idx_pts)
    obs = sin_vals + noise  # y = f(x) + noise
    return idx_pts, obs


def random_noise(loc, var, num):
    return np.random.normal(loc=loc,
                            scale=np.sqrt(var),
                            size=(num ))


def generate_random_points(num_pts, range_val):
    pts = np.random.uniform(range_val[0], range_val[1], size=(num_pts, 2)) # (100,3)
    return pts


def create_line(u, v):
    line = u-v # == [begin[0]-end[0], begin[1]-end[1], begin[2]-end[2]]
    return line


def create_random_cov(n):
    matrix_rand = np.random.uniform(0,1,n**2).reshape(-1,n)
    cov_vv = np.dot(matrix_rand, matrix_rand.T)
    return cov_vv


def generate_6d_polinomials():
    col_o = np.random.uniform(size=(1200, 1))
    col_o2 = col_o ** 2
    col_o3 = col_o ** 3
    col_o4 = col_o ** 4
    col_o5 = col_o ** 5
    col_o6 = col_o ** 6
    train_dataset = np.zeros((0, 6))
    for i in range(col_o.shape[0]):
        row_ = np.array([col_o[i], col_o2[i], col_o3[i], col_o4[i], col_o5[i], col_o6[i]]).reshape(1, -1)
        train_dataset = np.vstack((train_dataset, row_))
    return train_dataset


def generate_4d_polinomials():
    col_o = np.random.uniform(size=(1200, 1))
    col_o2 = col_o ** 2
    col_o3 = col_o ** 3
    col_o4 = col_o ** 4
    train_dataset = np.zeros((0, 4))
    for i in range(col_o.shape[0]):
        row_ = np.array([col_o[i], col_o2[i], col_o3[i], col_o4[i]]).reshape(1, -1)
        train_dataset = np.vstack((train_dataset, row_))
    return train_dataset