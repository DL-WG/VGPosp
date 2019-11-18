from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()
from mpl_toolkits.mplot3d import Axes3D

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk
from tensorboard.plugins import projector
# import tensorflow_datasets as tfds
tf.keras.backend.get_session().run(tf.global_variables_initializer())
tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
from matplotlib import cm

import gp_functions as gpf
import plots as plts
import data_generation as dg
import data_readers as drs

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
LOGDIR = "./log_dir_keras_VAE/"
SELECT_ROW_NUM = 8000
CSV = 'roomselection_1800.csv'
CHECKPOINT_PATH = "./vae_training/model_t.ckpt"

#LARGE SCALE TRAINING. current best loss: 1.014
# CHECKPOINT_PATH = "./vae_training/model.ckpt"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
SAVED_MODEL_DIR = "./vae_model/"
BATCH_SIZE = 400 # num of training
EPOCH = 500
encoded_size = 1  # latent variable z has encoded_size dimensions

# Python
mean_xyztptr = np.array([225.1191196136701, 66.54242329123329, 7.3222030832095095, 15.175489533803864, 1.0791075677934623, 0.0018159087825037148])
stdev_xyztptr = np.array([2.1742510933755383, 1.7179175136851061, 0.962321422680363, 4.146289019353405, 0.18341642616992837, 0.0007434639347162126])


'''0. Generate Dataset
'''
GENERATE_DATA = False
# if GENERATE_DATA == True:
#     train_dataset = dg.generate_4d_polinomials()
#     INPUT_SHAPE = train_dataset.shape[1]
#     test_dataset = dg.generate_4d_polinomials()
#     trainA = gpf.create_dataframe_from_4d_dataset(train_dataset)
#     plts.pairplots(trainA)


#========================================================
# Data in np and pd
#========================================================
# Select a fraction of the csv, indexed randomly, halved for training and test data

dim1 = 'Points:0'
dim2 = 'Points:1'
dim3 = 'Points:2'
dim4 = 'Temperature'
dim5 = 'Pressure'
dim6 = 'Tracer'
FEATURES = [dim1, dim2, dim3, dim4, dim5, dim6]
df = pd.read_csv(CSV, encoding='utf-8', engine='c')

xyztptr_idx_df = gpf.randomize_df(df, SELECT_ROW_NUM, dim1, dim2, dim3, dim4, dim5, dim6)
xyztpt_idx = xyztptr_idx_df.to_numpy()

xyztptr_norm = (xyztpt_idx - mean_xyztptr) / stdev_xyztptr

train_dataset, test_dataset = gpf.load_randomize_select_train_test(xyztptr_norm)

xyztp_train, tr_train = gpf.separate_to_encode_dataset_and_tracer_dataset(train_dataset) # we dont need tracer for VAE
xyztp_test, tr_test =  gpf.separate_to_encode_dataset_and_tracer_dataset(test_dataset)

INPUT_SHAPE = xyztp_train.shape[1]# we dont want to encode tracer, that is observation

h = pd.DataFrame(xyztp_train)
plts.pairplots(h)

#========================================================
# Create Graph
#========================================================

tf.reset_default_graph()
tf.keras.backend.clear_session()
sess = gpf.reset_session()

vae, prior, encoder, decoder = gpf.create_model(encoded_size, INPUT_SHAPE)


#========================================================
# Instanciation
#========================================================
saver = tf.train.Saver()
summ = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())  # after the init
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)


#========================================================
# Graph calls
#========================================================

# Train model for fix num of epochs, return History object, record of training loss values
# Output of TFP layer should be a distribution object
history = vae.fit(  xyztp_train, # train data
                    xyztp_train, # target data
                    epochs=EPOCH,
                    steps_per_epoch=2)

# tf.summary.scalar("VAE loss", history.history)

plts.plot_history(12, 4, history)

save_path = saver.save(sess, CHECKPOINT_PATH)
vae.summary()

# tf.saved_model.save(vae, SAVED_MODEL_DIR)
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()

score = vae.evaluate(xyztp_train, xyztp_test[:xyztp_train.shape[0],:]) # should it be test - test
print(score)
# Generate output prediction for input samples
# output_pred = vae.predict(pred_testset)
# print("output_pred", output_pred)


# Input: np array,
testnote = np.array(xyztp_test[:4000])
testnote = testnote.reshape(-1, INPUT_SHAPE)

# sample
xhat = vae(testnote)
xhat_m = xhat.mean()
xhat_s = xhat.sample()
print("xhat_m.shape", xhat_m.shape)
print("xhat_s.shape", xhat_s.shape)
print("sample instead of prediction--------------")

pass


'''3. Sample, evaluate effectiveness
'''
# We'll just examine ten random digits.



# if EAGER_EXEC_ENABLES == True:
#     assert tf.executing_eagerly()
#
#     # Returns scalar, the loss value & metrics values for the model in test mode.
#     test_loss = vae.evaluate(test_dataset,
#                  test_dataset,
#                  )
#
#     pred_against = test_dataset[1,:]
#
#     pred_against_ = np.array([])
#     for i in range(pred_against.shape[0]):
#         pred_against_ = np.append(pred_against_, pred_against[i])
#     # pred_against = pred_against.reshape(1,6)
#     vae.summary()
#     print("pred_against.shape", pred_against.shape)
#     # Returns a np array of predictions
#     pred = vae.predict(x=pred_against_,)
#     error = pred - pred_against
#     z = next(iter(test_dataset))[0]

    # z = vae.predict(x=x)
    # z = encoder.predict(x=x)
    # sess = tf.Session()
    # z = sess.run([encoder(x)])

