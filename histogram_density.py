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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import sys

import data_readers as dr

LOGDIR = "./log_dir_/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S') + "_histogram"


def test_one_update_on_constant_input():

    k = tf.placeholder(tf.float32)

    # hist = tf.histogram_fixed_width(values, value_range, nbins=10)

    # tracer_h = tf.placeholder(tf.float32)
    # tf.summary.histogram("tracer", tracer_h)

    temp_h = tf.placeholder(tf.float32)
    tf.summary.histogram("temp", temp_h)

    summaries = tf.summary.merge_all()

    sess = tf.Session()
    writer = tf.summary.FileWriter(LOGDIR)

    # ROWS
    r1 = np.random.uniform(-5, -4, 2000)
    r2 = np.random.uniform(0,1,2000)
    r3 = np.random.uniform(3,4,2000)
    r4 = np.random.uniform(-1, 7,2000)
    matrix_longrow = np.stack([r1, r2, r3, r4])

    # COLUMNS
    r1_ = np.reshape(r1, [-1, 1])
    r2_ = np.reshape(r2, [-1, 1])
    r3_ = np.reshape(r3, [-1, 1])
    r4_ = np.reshape(r4, [-1, 1])
    matrix = np.column_stack([r1_, r2_, r3_, r4_])


    dim, tracer, time, temperature, pressure, velocity, density = dr.load_pvtk_to_arrays([0, 1])

    i = 0
    while i < 700:
        dim_, tracer_, time_, temperature_, pressure_, velocity_, density_ = dr.load_pvtk_to_arrays([i,i+1])
        # dim = np.stack([dim, dim_])
        tracer = np.hstack([tracer, tracer_])
        # time = np.hstack([time, time_])
        temperature = np.hstack([temperature, temperature_])
        # pressure = np.hstack([pressure, pressure_])
        # velocity = np.hstack([velocity, velocity_])
        # density= np.hstack([density, density_])
        i+= 100

    # N = tracer.shape[0]
    # step = 0
    # while step < N:
    #     xyz_row_tr = tracer[step, :]
    #     summ_tr = sess.run(summaries, feed_dict={tracer_h: xyz_row_tr})
    #     writer.add_summary(summ_tr, global_step=step)
    #     step += 100

    N = temperature.shape[0]
    step = 0
    while step < N:
        xyz_row_temp = temperature[step, :]
        summ_temp = sess.run(summaries, feed_dict={temp_h: xyz_row_temp})
        writer.add_summary(summ_temp, global_step=step)
        step += 20


def fixed_row_histogram():
    k = tf.placeholder(tf.float32)
    shape_s = 20
    t = tf.random_uniform([shape_s, 40],  maxval=k*1)
    # t = tf.Variable([])
    i = 0
    hist = tf.constant(0, shape=[0, shape_s], dtype=tf.int32)
    cond = lambda i, _: i < shape_s
    def loop_body(i, hist):
        h = tf.histogram_fixed_width(t[i, :], [0.0, 10.0], nbins=shape_s)
        return i + 1, tf.concat([hist, tf.expand_dims(h, 0)], axis=0)

    i, hist = tf.while_loop(
        cond, loop_body, [i, hist],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, shape_s])])

    tf.summary.histogram("hist", hist)

    poisson = tf.random_poisson(shape=[1000], lam=k)
    tf.summary.histogram("poisson", poisson)

    # sess = tf.InteractiveSession()
    # print(hist.eval())

    summaries = tf.summary.merge_all()

    # Setup a session and summary writer
    sess = tf.Session()
    writer = tf.summary.FileWriter(LOGDIR)

    cols = np.arange(0., 100.,10)
    # Setup a loop and write the summaries to disk
    N = 20
    for step in range(N):
        k_val = step / float(N)
        summ = sess.run(summaries, feed_dict={k: k_val})
        writer.add_summary(summ, global_step=step)




def fixed_row_histogram_2(matrix_, vals, nbins, t_num_rows):
    t_histo_rows = [
        tf.histogram_fixed_width(tf.gather(matrix_, [row]), vals, nbins)
        for row in range(t_num_rows)]
    t_histo = tf.pack(t_histo_rows, axis=0)
    return t_histo


def test__histogram_distributions():
    k = tf.placeholder(tf.float32)

    # Make a normal distribution, with a shifting mean
    mean_moving_normal = tf.random_normal(shape=[1000], mean=(5 * k), stddev=1)
    # Record that distribution into a histogram summary
    tf.summary.histogram("normal/moving_mean", mean_moving_normal)

    # Make a normal distribution with shrinking variance
    variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1 - (k))
    # Record that distribution too
    tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)

    # Let's combine both of those distributions into one dataset
    normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
    # We add another histogram summary to record the combined distribution
    tf.summary.histogram("normal/bimodal", normal_combined)

    # Add a gamma distribution
    gamma = tf.random_gamma(shape=[1000], alpha=k)
    tf.summary.histogram("gamma", gamma)

    # And a poisson distribution
    poisson = tf.random_poisson(shape=[1000], lam=k)
    tf.summary.histogram("poisson", poisson)

    # And a uniform distribution
    uniform = tf.random_uniform(shape=[1000], maxval=k * 10)
    tf.summary.histogram("uniform", uniform)

    # Finally, combine everything together!
    all_distributions = [mean_moving_normal, variance_shrinking_normal,
                         gamma, poisson, uniform]
    all_combined = tf.concat(all_distributions, 0)
    tf.summary.histogram("all_combined", all_combined)

    summaries = tf.summary.merge_all()

    # Setup a session and summary writer
    sess = tf.Session()
    writer = tf.summary.FileWriter(LOGDIR)

    # Setup a loop and write the summaries to disk
    N = 5
    for step in range(N):
        k_val = step / float(N)
        summ = sess.run(summaries, feed_dict={k: k_val})
        writer.add_summary(summ, global_step=step)


def test__histogram_of_random_matrix():
    rmatrix = tf.constant(np.random.normal(0, 3, size=(11, 11)))

    hist = tf.summary.histogram('rmatrix', rmatrix)
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    # saver = tf.train.Saver()  # all the object in the graph mapped

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # after the init
    # allmodel_saved_path = saver.save(sess, './saved_variable')
    # print('model saved in {}'.format(allmodel_saved_path))
    writer.add_graph(sess.graph)

    h = sess.run(summ)
    print(h)

if __name__ == '__main__':
    # test__histogram_of_random_matrix()
    # test__histogram_distributions()
    # fixed_row_histogram()
    test_one_update_on_constant_input()
    pass