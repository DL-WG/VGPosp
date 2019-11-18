import numpy as np
import tensorflow as tf

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
import tensorflow_datasets as tfds

tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers


import os
import time
#%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
#%config InlineBackend.figure_format = 'png'

tf.enable_eager_execution()   # iter() require eager execution
assert tf.executing_eagerly()

'''
Load dataset
'''

datasets, datasets_info = tfds.load(name='mnist',
                                    with_info=True,
                                    as_supervised=False)


def _preprocess(sample):
  image = tf.cast(sample['image'], tf.float32) / 255.  # Scale to unit interval.
  image = image < tf.random.uniform(tf.shape(image))   # Randomly binarize.
  return image, image

train_dataset = (datasets['train']
                 .map(_preprocess)
                 .batch(256)
                 .prefetch(tf.data.experimental.AUTOTUNE)
                 .shuffle(int(10e3)))
eval_dataset = (datasets['test']
                .map(_preprocess)
                .batch(256)
                .prefetch(tf.data.experimental.AUTOTUNE))


'''
Simplify model
'''
input_shape = datasets_info.features['image'].shape

encoded_size = 16 # latent variable z has encoded_size dimensions
base_depth = 32

print("--------------")

# indep gaussian distribution, no learned parameters
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)

print("--------------")

# encoder = normal Keras Sequential model, output passed to MultivatiateNormalTril()
#   which splits activations from final Dense() layer into that is
#   needed to spec mean and lower triangular cov matrix
# Add KL div between encoder and prior to loss.
# full covariance Gaussian distribution
# mean and covariance matricies parameterized by output of NN
encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
    tfkl.Conv2D(base_depth, 5, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(base_depth, 5, strides=2,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(2 * base_depth, 5, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(2 * base_depth, 5, strides=2,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(4 * encoded_size, 7, strides=1,
                padding='valid', activation=tf.nn.leaky_relu),
    tfkl.Flatten(),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
               activation=None),
    tfpl.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
])

decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[encoded_size]),
    tfkl.Reshape([1, 1, encoded_size]),
    tfkl.Conv2DTranspose(2 * base_depth, 7, strides=1,
                         padding='valid', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(2 * base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(2 * base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(filters=1, kernel_size=5, strides=1,
                padding='same', activation=None),
    tfkl.Flatten(),
    tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
])

print("--------------")


# output defined as composition of encoder and decoder
#   only need to specify the reconstruction loss: first term of ELBO
vae = tfk.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]))

'''
Do Inference
'''
# original input: x
# output of model: rv_x (is random variable x)
# loss function is negative log likelihood of
#   the data given the model.
# negloglik = lambda x, rv_x: -rv_x.log_prob(x)

def calculate_neg_log_likelihood(x, rv_x):
    return -rv_x.log_prob(x)


vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=calculate_neg_log_likelihood)

# output of TFP layer should be a distribution object
vae.fit(train_dataset,
        epochs=1,
        validation_data=eval_dataset,
        steps_per_epoch=2,  # 240 is OK: just made it smaller experimentally .. so it does not run out of data
        validation_steps=1
        )

'''
Sample
'''
# We'll just examine ten random digits.

assert tf.executing_eagerly()
x = next(iter(eval_dataset))[0][:10]
xhat = vae(x)
assert isinstance(xhat, tfd.Distribution)
pass