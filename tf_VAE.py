from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import tensorflow as tf

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from matplotlib import cm

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()
from mpl_toolkits.mplot3d import Axes3D


import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk
from tensorboard.plugins import projector
import tensorflow_datasets as tfds
tf.keras.backend.get_session().run(tf.global_variables_initializer())
tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

import gp_functions as gpf
import plots as plts
import data_generation as dg
# import data_readers as drs

import os
import time
#%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
#%config InlineBackend.figure_format = 'png'

#========================================================
# Constants
#========================================================
LOGDIR = "./log_dir_vae/vae/"
SELECT_ROW_NUM = 8000
ENCODED_SIZE = 1  # latent variable z has encoded_size dimensions
BATCH_SIZE = 100 # num of training
CSV = 'roomselection_1800.csv'
N_LATENT = 1
EPOCH = 20


EAGER_EXEC_ENABLES = False
GENERATE_DATA = False

if EAGER_EXEC_ENABLES == True:
    tf.enable_eager_execution()   # iter() require eager execution
    assert tf.executing_eagerly()

#========================================================
# Data
#========================================================


'''0. Generate Dataset
'''
if GENERATE_DATA == True:
    train_dataset = dg.generate_4d_polinomials()
    INPUT_SHAPE = train_dataset.shape[1]
    test_dataset = dg.generate_4d_polinomials()
    trainA = gpf.create_dataframe_from_4d_dataset(train_dataset)
    plts.pairplots(trainA)


'''0. Load dataset
'''
dim1 = 'Points:0'
dim2 = 'Points:1'
# dim3 = 'Points:2'
dim3 = 'Tracer'
dim4 = 'Temperature'

# Select a fraction of the csv, indexed randomly, halved for training and test data
xyzt_idx, train_dataset, test_dataset = gpf.load_randomize_select_train_test(CSV, SELECT_ROW_NUM, dim1, dim2, dim3, dim4)

INPUT_SHAPE = train_dataset.shape[1]
plts.pairplots(xyzt_idx)

print("-----------")

#========================================================
# Code
#========================================================
'''1. select and simplify model
'''
tf.reset_default_graph()

'''Define model input and output placeholders
'''
with tf.name_scope("initplaceholders"):
    X_in = tf.placeholder(dtype=tf.float32, shape=[train_dataset.shape[1]], name='X_in')
    X_out = tf.placeholder(dtype=tf.float32, shape=[train_dataset.shape[1]], name='X_out')
    Y_flat = tf.reshape(X_out, shape=[-1, 4])
    rate = tf.placeholder(dtype=tf.float32, shape=(), name='rate')

# def prior(encoded_size, name="prior_distribution"):
# with tf.name_scope("prior"):
#     prior= tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size),scale=1),
#                                       reinterpreted_batch_ndims=1)
        # return prior


# def multivariatenormal(prior, decoded_size, name="multivarnorm_distribution"):
# with tf.name_scope("multivariate_norm"):
#     multivarnorm = tfpl.MultivariateNormalTriL(decoded_size,
#         activity_regularizer=tfpl.KLDivergenceRegularizer(prior))
        # return multivarnorm


reshaped_dim = [-1, ENCODED_SIZE]
inputs_decoder = ENCODED_SIZE

# Dense : output = activation(dot(input, kernel) + bias)
# def encoder(X_in, rate):
activation = tf.nn.leaky_relu
with tf.name_scope("encoder_sequence"):
    e_in = tf.reshape(X_in, shape=[-1,INPUT_SHAPE])
    e1 = tf.layers.dense(e_in, units=N_LATENT, activation=activation)
    e2 = tf.nn.dropout(e1, rate)
    e3 = tf.layers.flatten(e2)
    # e4 = prior(ENCODED_SIZE, "prior")

    with tf.name_scope("prior"):
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(ENCODED_SIZE), scale=1),
                                reinterpreted_batch_ndims=1)
    with tf.name_scope("multivariate_norm"):
        multivarnorm = tfpl.MultivariateNormalTriL(ENCODED_SIZE,
                                                   activity_regularizer=tfpl.KLDivergenceRegularizer(prior))

    mn = tf.layers.dense(e3, units=N_LATENT)
    stdev = 0.5 * tf.layers.dense(e3, units=N_LATENT)
    epsilon = tf.random_normal(tf.stack([tf.shape(e3)[0], N_LATENT]))

    e_z = mn + tf.multiply(epsilon, tf.exp(stdev))
    # return z, mn, stdev

DECODED_SIZE = [-1,4]

# def decoder(sampled_z, rate):
# activation = tf.nn.leaky_relu
with tf.name_scope("decoder_sequence"):

    d_in = tf.layers.dense(e_z, units=inputs_decoder, activation=activation, name="d_in")
    d1 = tf.layers.dense(d_in, units=inputs_decoder*4, activation=activation, name="d1")
    d2 = tf.layers.flatten(d1,name="d2")
    d3 = tf.layers.dense(d2, units=4, activation=tf.nn.sigmoid, name="d3")
    # d4 = tf.layers.dense(tfpl.IndependentBernoulli(tfd.Bernoulli.logits), units=4, name="d4")
    d_out = tfp.trainable_distributions.bernoulli(d3, name="d_out")
    # d_out = tf.reshape(d4, shape=[1, 4], name="d_out")
    # return reconstruction


# sampled, mn, stdev = encoder(X_in, rate)
# dec_out = decoder(sampled, rate)
# with tf.name_scope("optimize_step"):
    # out_shaped = tf.reshape(d_out, [-1, 4])
    # img_loss = tf.reduce_sum(tf.squared_difference(out_shaped, Y_flat), 1)
    # latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * stdev - tf.square(mn) - tf.exp(2.0 * stdev), 1)
    # loss = tf.reduce_mean(img_loss + latent_loss)
    # optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)


'''Tf graph writer, saver
'''
summ = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
saver = tf.train.Saver()

'''2. Train model
'''
batch = train_dataset[0,:]
sess.run(d2, feed_dict={X_in: batch, X_out: batch, rate: 0.8})
print("====================")
# sess.run(optimizer, feed_dict={X_in: batch, X_out: batch, rate: 0.8})


# for e in range(EPOCH):
#     for i in range(4000):
#         batch = np.zeros((0, 4))
#         batch = train_dataset[i,:]
#         sess.run(optimizer, feed_dict={X_in: batch, X_out: batch, rate: 0.8})
#
#         # Sample, evaluate effectiveness
#         if not i % 200:
#             ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec_out, img_loss, latent_loss, mn, stdev],
#                                                    feed_dict={X_in: batch, X_out: batch, rate: 1.0})
#
#     print("Epoch: ",e," Loss:" ,ls," Mean:", np.mean(i_ls), np.mean(d_ls))
#
# graph = tf.get_default_graph()
# print(graph.get_operations())
