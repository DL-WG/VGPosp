from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

'''
input xyz and other feature vectors
datatypes : float
calc byte repr loss so we know what compression size we need
    (6, 10000) - 3 float , 3 spatial , 10000 sample 
    10 factor compression (smaller than work to vec)
    encoder step -> aim 1 factor output, compressed state : (1, 10000)
    
    decoder should be able to redo (6, 10000) - learning loss curve 
    measure loss according to: 60% repr is ok already -> measure it by saying the values are epsilon distance 
        from the original 
'''

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
idx = np.random.randint(0, 10000, 2)

# Grayscale examples of the data we will be working with
f, ax = plt.subplots(1, 2, figsize=(10, 10));
ax[0].imshow(train_images[idx[0]].reshape(14, -1), cmap='gray');
ax[0].set(title='Digit Value: {}'.format(np.argmax(train_images[idx[0]])));
ax[1].imshow(test_images[idx[1]].reshape(28, 28), cmap='gray');
ax[1].set(title='Digit Value: {}'.format(np.argmax(test_images[idx[1]])));
plt.show()
print("1--------")

'''
test_images (10000, 28, 28), size: 784000
idx (2,)
MNIST_dim = 784 = 28*28
batch_size = 100
num_iterations = 10000 

? hidden_dim = 500
? latent_dim_vec = [2,5,10,20]

'''


MNIST_dim = test_images.shape[1]**2 # (img_x * img_y)
batch_size = 100
num_iterations = 10000
hidden_dim = 500
latent_dim_vec = [2,5,10,20]

print("2--------")



######################################################




def load_dataset(ds):
    dset = ds.reshape(ds.shape[0], 28, 28, 1).astype('float32')
    return dset

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_data = train_images
test_data = test_images

train_images = load_dataset(train_data)
test_images = load_dataset(test_data)

def normalize_between_zero_and_one(images):
    c = 255.
    return images/c

train_data = normalize_between_zero_and_one(train_images)
test_data= normalize_between_zero_and_one(test_images)

# Normalizing the images to the range of [0., 1.]
# train_images /= 255.
# test_images /= 255.

# Binarization
train_data[train_data >= .5] = 1.
train_data[train_data < .5] = 0.
test_data[test_data >= .5] = 1.
test_data[test_data< .5] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data).shuffle(TEST_BUF).batch(BATCH_SIZE)
pass
class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )

  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
  with tf.GradientTape() as tape:
      loss = compute_loss(model, x)
  return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
  optimizer.apply_gradients(zip(gradients, variables))

epochs = 10 # 100
latent_dim = 50
num_examples_to_generate = 2

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_input):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    gradients, loss = compute_gradients(model, train_x)
    apply_gradients(optimizer, gradients, model.trainable_variables)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))
    generate_and_save_images(
        model, epoch, random_vector_for_generation)

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

plt.imshow(display_image(epochs))
plt.axis('off')# Display images

anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info >= (6,2,0,''):
  display.Image(filename=anim_file)

try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download(anim_file)