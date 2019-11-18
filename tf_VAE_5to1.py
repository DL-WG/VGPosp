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
tfkb = tf.keras.backend
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

# tf.enable_eager_execution()   # iter() require eager execution
# assert tf.executing_eagerly()

'''
Load dataset
'''
LOGDIR = "./log_dir_VAE_/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S') + "_vae"
CSV_GP = './main_datasets_/Dset_xyz_ave_small.csv'  # 'roomselection_1800.csv'

# train_dataset = np.random.uniform(size=(1200, 6))
FRACTION_OF_DATASET = 0.1
ds = drs.load_csv(CSV_GP)

ds_ = ds.loc[:, ['Points:0', 'Points:1', 'Points:2', 'Pressure_ave', 'Temperature_ave']]

ds_idxs = np.linspace(0, ds.shape[0] - 1, ds.shape[0]).T
np.random.shuffle(ds_idxs)
train_cnt = int(ds.shape[0] * FRACTION_OF_DATASET)

train_idxs = np.array(ds_idxs[:train_cnt])
train_dataset = np.array( ds_.loc[train_idxs, :] )

test_idxs = np.array(ds_idxs[train_cnt:2*train_cnt])
test_dataset = np.array( ds_.loc[test_idxs, :] )

# mean = np.average(train_dataset, axis=0)
# central = train_dataset - mean
# stdev = np.std(central, axis=0)
# normal_dataset = central / stdev
# print('mean: ', mean)
# print('stdev: ', stdev)

mean = [225.11911961, 66.54242329, 7.32220308, 1.07910757, 15.17548953]
stdev = [2.17425109, 1.71791751, 0.96232142, 0.18341643, 4.14628902]

central_train = train_dataset - mean
normal_train = central_train / stdev

central_test = test_dataset - mean
normal_test = central_test / stdev

mean_check = np.average(normal_train, axis=0)
stddev_check = np.std(normal_train, axis=0)
print('mean_check  : ', mean_check)
print('stddev_check: ', stddev_check)

input_shape = train_dataset.shape[1]

'''
Simplify model
'''
# input_shape = datasets_info.features['image'].shape

encoded_size = 5  # latent variable z has encoded_size dimensions

print("--------------")

with tf.name_scope('VAE'):
    # indep gaussian distribution, no learned parameters
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                            reinterpreted_batch_ndims=1)

    print("--------------")


    tf_mean = tf.constant([225.11911961, 66.54242329, 7.32220308, 1.07910757, 15.17548953])
    tf_stdev = tf.constant([2.17425109, 1.71791751, 0.96232142, 0.18341643, 4.14628902])


    # encoder = normal Keras Sequential model, output passed to MultivatiateNormalTril()
    #   which splits activations from final Dense() layer into that is
    #   needed to spec mean and lower triangular cov matrix
    # Add KL div between encoder and prior to loss.
    # full covariance Gaussian distribution
    # mean and covariance matricies parameterized by output of NN

    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=input_shape),
        tfkl.Dense(15, activation=tf.nn.leaky_relu),
        tfkl.Dense(encoded_size, activation=tf.nn.leaky_relu),
    ])

    tf.summary.histogram('encoded_dense_outputs', encoder.outputs)

    encoder_head = tfk.Sequential([
        tfkl.InputLayer(input_shape=encoded_size),

        tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
                   activation=None),

        tfpl.MultivariateNormalTriL(
            encoded_size,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
        ),
    ])

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[encoded_size]),
        tfkl.Dense(15, activation=tf.nn.leaky_relu),
        tfkl.Dense(input_shape, activation=tf.nn.leaky_relu),
    ])

    tf.summary.histogram('decoded_dense_outputs', decoder.outputs)

    # emp = tfd.Empirical(normal_train[np.newaxis, ...], event_ndims=1)

    decoder_head = tfk.Sequential([
        tfkl.InputLayer(input_shape=[input_shape]),
        # emp,
        # tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
        tfkl.Dense(tfpl.IndependentNormal.params_size(input_shape)),
        tfpl.IndependentNormal(input_shape)
    ])

    print("--------------")

    # output defined as composition of encoder and decoder
    #   only need to specify the reconstruction loss: first term of ELBO
    vae = tfk.Model(inputs=encoder.inputs,
                    outputs=decoder_head(decoder(
                                            encoder_head(encoder.outputs[0])
                    )))

    epoch_print_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=lambda epoch, logs: tf.print('epoch', epoch)
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)  # before init


    '''
    Do Inference
    '''
    # original input: x
    # output of model: rv_x (is random variable x)
    # loss function is negative log likelihood of
    #   the data given the model.
    # negloglik = lambda x, rv_x: -rv_x.log_prob(x)


    def calculate_neg_log_likelihood(x, rv_x):
        # return tf.losses.log_loss(x, rv_x)
        return -rv_x.log_prob(x)


    def calculate_log_mse(x, rv_x):
        return tf.log(tf.losses.mean_squared_error(x, rv_x))


    def custom_metrics_max_diff(x, x_hat):
        return tf.reduce_max(x-x_hat)


    # n_z = 2
    # inputs = Input(shape=(784,))
    # h_q = tfkl.Dense(512, activation='relu')(inputs)

    # mu = tfkl.Dense(n_z, activation='linear')(h_q)
    # log_sigma = tfkl.Dense(n_z, activation='linear')(h_q)

    # mu = tfk.Model(inputs=encoder.inputs,
    #                outputs=encoder.outputs[0])
    #
    # log_sigma = tfk.Model(inputs=encoder.inputs,
    #                       outputs=encoder.outputs[0])

    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        # recon = tfkb.sum(tfkb.binary_crossentropy(y_pred, y_true), axis=1)
        diff = tf.reshape(y_true - y_pred, [-1, 1])
        reconstr = tf.math.reduce_mean(tf.matmul(diff, diff, transpose_b=True))
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        # kl = 0.5 * tfkb.sum(tfkb.exp(log_sigma) + tfkb.square(mu) - 1. - log_sigma, axis=1)

        # KL loss is being added as a regularizer loss (by the term above):
        # activity_regularizer=tfpl.KLDivergenceRegularizer(prior)

        return reconstr  # + kl


    # Build the evidence lower bound (ELBO) or the negative loss
    # kl = tf.reduce_sum(tfd.kl_divergence(q_z, p_z), 1)
    # expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), [1, 2, 3])
    # elbo = tf.reduce_sum(expected_log_likelihood - kl, 0)
    # optimizer_ = tf.train.RMSPropOptimizer(learning_rate=0.001)
    # train_op = optimizer_.minimize(-elbo)


    vae.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=vae_loss,
                metrics=['mse', 'acc', custom_metrics_max_diff]
                )

    summ = tf.summary.merge_all()
    saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())  # after the init

writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

# saver.restore(sess, LOGDIR + '/headless_train.ckpt')

# output of TFP layer should be a distribution object
history = vae.fit(
        x=normal_train,
        y=normal_train,
        epochs=10000,
        batch_size=normal_train.shape[0],
        steps_per_epoch=200,  # 3*400 = 1200

        validation_data=(normal_test, normal_test),
        validation_steps=1,

        callbacks=[
            tensorboard_callback
        ]
        )
        #, distribute=strategy)

plts.plot_history(12, 4, history)

save_path = saver.save(sess, LOGDIR+'/headless_train.ckpt')
vae.summary()

# # with tf.Session() as sess:
# #     sess.run(init)
#     '''
#     Sample
#     '''
#     # We'll just examine ten random digits.
#
#     # assert tf.executing_eagerly()
#     # x = np.random.uniform(size=(333, input_shape))
#
#     # z = vae.predict(x=x)
#     # z = encoder.predict(x=x)
#
#     # sess = tf.Session()
#     # enc = encoder(x)
#     # enc_ = sess.run(enc)
#
#     # z = sess.run([tf.get_variable(shape=[x.shape[0], encoded_size], name="dense_1")],
#     #              feed_dict={tf.get_variable(shape=x.shape, name="input_1"): x})
#
#     # plt.hist(enc_[:, 0], bins=30)
#     # plt.title("Histogram of encoder_for_roomselection_1800(random.uniform)")
#
#     # plt.hist(z, bins=10)
#     # plt.title("Histogram of encoder(x)")
#
#     # plt.show()

pass