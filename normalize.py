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

import imports as im
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

import gp_functions as gpf
import data_generation as dg
import plots as plts
import data_readers as drs
import main_GP_fit

def normalization_factors_from_training_data(df, FEATURES,
                                             chkpt_name = 'normalize/normalize.ckpt'):

    # tf.reset_default_graph()# Save the variables to disk.

    # for f in FEATURES:
    #     with tf.name_scope(f):

    values = tf.placeholder(shape=[df.shape[0], len(FEATURES)],
                            dtype=tf.float32)
    mean = tf.reduce_mean(values, axis=[0])
    centered = values - mean
    stdev = tf.math.reduce_std(centered, axis=[0])
    normal = centered / stdev

    mean_var = tf.Variable(initial_value=mean,
                       shape=[len(FEATURES)],
                       dtype=tf.float32,
                       name='mean')
    stdev_var = tf.Variable(initial_value=stdev,
                        shape=[len(FEATURES)],
                        dtype=tf.float32, name='stdev')


    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver({"mean": mean_var, "stdev": stdev_var})

    with tf.Session() as sess:
        sess.run(init_op,
                 feed_dict={values: df.loc[:, FEATURES]})

        mean_v, stdev_v, normal_v = sess.run(
            [mean, stdev, normal],
            feed_dict={values: df.loc[:, FEATURES]}
        )

        # checking
        print('mean: ', mean_v)
        print('stdev: ', stdev_v)
        i = 0
        for f in FEATURES:
            m = np.mean(df[f])
            v = np.std(df[f])
            m_ = np.mean(normal_v[:, i])  # expect 0.
            v_ = np.std(normal_v[:, i])  # expect 1.
            print('normalized mean, stdev for ', f, m, v, m_, v_)
            i += 1

        save_path = saver.save(sess, chkpt_name)
        print("Model saved in path: %s" % save_path)
    pass


def load_saved_mean_stdev_to_see_their_values(df, FEATURES,
                                              chkpt_name = 'normalize/normalize.ckpt'):
    tf.reset_default_graph()
    mean_var = tf.Variable(initial_value=tf.zeros(shape=[len(FEATURES)]),
                       shape=[len(FEATURES)],
                       dtype=tf.float32,
                       name='mean')
    stdev_var = tf.Variable(initial_value=tf.ones(shape=[len(FEATURES)]),
                        shape=[len(FEATURES)],
                        dtype=tf.float32, name='stdev')

    init_op = tf.global_variables_initializer()

    # Add ops to save and restore only 'mean_var' using the name 'mean'
    # .. and 'stdev_var' using the name 'stdev'
    saver = tf.train.Saver({"mean": mean_var, "stdev": stdev_var})

    with tf.Session() as sess:
        sess.run(init_op)

        print('before loading')
        print('mean_var', mean_var.eval())  # .eval() is a shorthand for eval = sess.run(mean_var)
        print('stdev_var', stdev_var.eval())

        saver.restore(sess, chkpt_name)  # loading only affects mean_var and stdev_var (if there are other ops)
        print('after loading')
        print('mean_var', mean_var.eval())  # .eval() is a shorthand for eval = sess.run(mean_var)
        print('stdev_var', stdev_var.eval())

    pass

def use_saved_mean_stdev_to_normalize(df, FEATURES,
                                      chkpt_name = 'normalize/normalize.ckpt'):
    tf.reset_default_graph()
    mean_var = tf.Variable(initial_value=tf.zeros(shape=[len(FEATURES)]),
                       shape=[len(FEATURES)],
                       dtype=tf.float32,
                       name='mean')
    stdev_var = tf.Variable(initial_value=tf.ones(shape=[len(FEATURES)]),
                        shape=[len(FEATURES)],
                        dtype=tf.float32, name='stdev')

    values = tf.placeholder(shape=[df.shape[0], len(FEATURES)],
                            dtype=tf.float32)
    centered = values - mean_var
    normal = centered / stdev_var

    init_op = tf.global_variables_initializer()

    # Add ops to save and restore only 'mean_var' using the name 'mean'
    # .. and 'stdev_var' using the name 'stdev'
    saver = tf.train.Saver({"mean": mean_var, "stdev": stdev_var})

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, chkpt_name)  # loading only affects mean_var and stdev_var (if there are other ops)

        normal_v = sess.run(
            [normal],
            feed_dict={values: df.loc[:, FEATURES]}
        )

        return normal_v
        pass

def TEST_normalize():
    df = pd.read_csv('roomselection_1800.csv', encoding='utf-8', engine='c')
    FEATURES = ['Points:0', 'Points:1', 'Points:2', 'Temperature', 'Pressure', 'Tracer']


    normalization_factors_from_training_data(df, FEATURES)

    load_saved_mean_stdev_to_see_their_values(df, FEATURES)

    [normal_v] = use_saved_mean_stdev_to_normalize(df, FEATURES)

    # checking
    print('--- checking: ---')
    i = 0
    for f in FEATURES:
        m = np.mean(df[f])
        v = np.std(df[f])
        m_ = np.mean(normal_v[:, i])  # expect 0.
        v_ = np.std(normal_v[:, i])  # expect 1.
        print('normalized mean, stdev for ', f, m, v, m_, v_)
    i += 1

if __name__ == '__main__':
    TEST_normalize()
    pass
