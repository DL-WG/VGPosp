###########################################
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
# import sys

# import imports as im
import gp_functions as gpf
# import data_generation as dg
# import plots as plts
# import data_readers as drs
# import main_GP_fit
# import placement_algorithm2 as alg2
# import normalize_delete as nm
# from tensorflow_core import training.saving.saveable_object_util as sou
# import tensorflow_core.python.training.saving.saveable_object_util as sou

# import tensorflow_core.python.training.saver as saver_util
from tensorflow.python.training import saver as saver_util

from tensorflow.python.training import checkpoint_management

from tensorflow.python.tools import inspect_checkpoint as chkp
# print all tensors in checkpoint file
CHECKPOINT_PATH = "./vae_training/model.ckpt"
chkp.print_tensors_in_checkpoint_file(CHECKPOINT_PATH, tensor_name='', all_tensors=True)


ENCODED_SIZE = 1
LOGDIR = "./log_dir_/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S') + "_gp"

vae, prior, encoder, decoder = gpf.create_model(ENCODED_SIZE, 5)
vae.summary()

######################################################################
# GRAPH SAVER ########################################################
print("LOGDIR", LOGDIR)
# for i, var in enumerate(saver._var_list):
#     print('Var {}: {}'.format(i, var))
summ = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGDIR)

saveable_names =\
    ['VAE_/decoder_/d_dense_/bias',
     'VAE_/decoder_/d_dense_/kernel',
     'VAE_/decoder_/d_dense_10/bias',
     'VAE_/decoder_/d_dense_10/kernel',
     'VAE_/encoder_/e_dense_/bias',
     'VAE_/encoder_/e_dense_/kernel',
     'VAE_/encoder_/e_dense_10/bias',
     'VAE_/encoder_/e_dense_10/kernel',
     'VAE_/encoder_/e_mvn_dense_/bias',
     'VAE_/encoder_/e_mvn_dense_/kernel']

# saveables = {op.name: op for op in sou.saveable_objects_for_op(encoder, 'VAE_/encoder_')}
# saveables = {name: tf.get_variable(name=name) for name in saveable_names}


# saver = tf.train.Saver(saveables)
"""
                        ['VAE_/decoder_/d_dense_/bias',
                        'VAE_/decoder_/d_dense_/kernel',
                        'VAE_/decoder_/d_dense_10/bias',
                        'VAE_/decoder_/d_dense_10/kernel',
                        'VAE_/encoder_/e_dense_/bias',
                        'VAE_/encoder_/e_dense_/kernel',
                        'VAE_/encoder_/e_dense_10/bias',
                        'VAE_/encoder_/e_dense_10/kernel',
                        'VAE_/encoder_/e_mvn_dense_/bias',
                        'VAE_/encoder_/e_mvn_dense_/kernel']
"""
# saver = saver_util.saver_from_object_based_checkpoint(CHECKPOINT_PATH)  # all
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())  # after the init
# allmodel_saved_path = saver.save(sess, './saved_variable')
# print('model saved in {}'.format(allmodel_saved_path))
writer.add_graph(sess.graph)

######################################################################
# SESS RUNS ##########################################################

for _ in range(2):
    e_test = encoder(tf.reshape(tf.constant([1., 1., 1., 1., 1.]), [-1, 5]))
    assert isinstance(e_test, tfd.Distribution)
    e_test_s = e_test.mean()
    print('encoder([1,1,1,1,1]) before restore', sess.run(e_test_s))

print(CHECKPOINT_PATH)
# TODO what are these checkpoints
# checkpoint = tf.train.Checkpoint()
# checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
saver.restore(sess, CHECKPOINT_PATH)

for _ in range(2):
    e_test = encoder(tf.reshape(tf.constant([1., 1., 1., 1., 1.]), [-1, 5]))
    assert isinstance(e_test, tfd.Distribution)
    e_test_s = e_test.mean()
    print('encoder([1,1,1,1,1]) after restore', sess.run(e_test_s))

pass
