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

# import imports as im
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime
# import time
#
# import gp_functions as gpf
# import data_generation as dg
# import plots as plts
# import data_readers as drs
# import main_GP_fit

def shapes_1():
    tf.reset_default_graph()

    x_start = tf.constant(-3.)
    x_stop = tf.constant(3.)
    x_num = tf.constant(5)
    y_start = tf.constant(-3.)
    y_stop = tf.constant(3.)
    y_num = tf.constant(5)
    x = tf.linspace(x_start, x_stop, x_num) # [1, 2, 3] , [4, 5, 6]
    y = tf.linspace(y_start, y_stop, y_num)
    X, Y = tf.meshgrid(x, y)
    # X_, Y_ = tf.expand_dims(X, -1), tf.expand_dims(Y, -1)  # dim is [y_num, x_num, 1]; [ [[1,2,3],[1,2,3],[1,2,3]], [[4,4,4],[5,5,5],[6,6,6]] ]
    # st = tf.stack([X, Y])    # [ [[1,2,3],[1,2,3],[1,2,3]], [[4,4,4],[5,5,5],[6,6,6]] ]
    st2 = tf.stack([X, Y], axis=2)   #  [[[1,4],[1,5],[1,6]],[[2,4],[2,5],[2,6]], ...]    # valami ilyen kell, a parok st[0,0] == [1, 4]
    vec = tf.reshape(st2, [2, -1])


    init_op = tf.global_variables_initializer()
    with tf.Session as sess:
        [vec_v] = sess.run([vec])
        print(vec_v)


def stack_1():
    p, t = tf.linspace(1., 3., 3), tf.linspace(4., 7., 4)  # [1, 2, 3] , [4, 5, 6]
    P, T = tf.meshgrid(p, t)

    st2 = tf.stack([P, T], axis=2)   #  [[[1,4],[1,5],[1,6]],[[2,4],[2,5],[2,6]], ...]    # valami ilyen kell, a parok st[0,0] == [1, 4]
    vec_grid_pt = tf.reshape(st2, [-1, 2])

    xyz = tf.constant([-3., -2.99, -2.98])

    fx = tf.fill(dims=[vec_grid_pt.shape[0], 1], value=xyz[0])
    fy = tf.fill(dims=[vec_grid_pt.shape[0], 1], value=xyz[1])
    fz = tf.fill(dims=[vec_grid_pt.shape[0], 1], value=xyz[2])
    p_ = tf.slice(vec_grid_pt, [0, 0], [vec_grid_pt.shape[0], 1])
    t_ = tf.slice(vec_grid_pt, [0, 1], [vec_grid_pt.shape[0], 1])

    fx_ = tf.reshape(fx, shape=[-1], name="fx")
    fy_ = tf.reshape(fy, shape=[-1], name="fy")
    fz_ = tf.reshape(fz, shape=[-1], name="fz")
    fp_ = tf.reshape(p_, shape=[-1], name="fp_")
    ft_ = tf.reshape(t_, shape=[-1], name="ft_")

    fxyzpt = tf.stack([fx_, fy_, fz_, fp_, ft_], axis=1)

    with tf.Session() as sess:
        # [vec_v] = sess.run([vec])
        print('P = \n', P.eval())
        print('T = \n', T.eval())

        print('stack([X,Y], axis=2) = \n', st2.eval())
        print('vec = \n', vec_grid_pt.eval())

        print('fxyzpt = ')
        print(fxyzpt.eval())

        pass

def TEST_shapes_1():
    stack_1()
    # shapes_1()

if __name__ == '__main__':
    TEST_shapes_1()
    pass


