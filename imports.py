import numpy as np
import tensorflow as tf
from matplotlib import cm

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()
from mpl_toolkits.mplot3d import Axes3D
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk
import matplotlib.pyplot as plt
from tensorboard.plugins import projector
import os
#%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
#%config InlineBackend.figure_format = 'png'
print(tf.__version__)

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()