import numpy as np
import tensorflow as tf
import data_readers as drs

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from matplotlib import cm

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk
from tensorboard.plugins import projector
# import tensorflow_datasets as tfds
import plots as plts
import datetime

tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

import os
import time


def mix():
    # Create a mixture of two Gaussians:
    tfd = tfp.distributions
    mix = 0.3
    bimix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[mix, 1.-mix]),
      components=[
        tfd.Normal(loc=-1., scale=0.1),
        tfd.Normal(loc=+1., scale=0.5),
    ])

    # Plot the PDF.
    import matplotlib.pyplot as plt
    with tf.Session() as sess:
        x = tf.linspace(-2., 3., int(1e4))
        x_ = sess.run(x)
        prob = sess.run(bimix_gauss.prob(x))
        plt.plot(x_, prob)
        plt.show()
        pass


if __name__ == '__main__':
    mix()
    pass
