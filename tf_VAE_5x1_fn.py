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
#%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
#%config InlineBackend.figure_format = 'png'

# tf.enable_eager_execution()   # iter() require eager execution
# assert tf.executing_eagerly()


def get_data(data_cnt = 1200,
             split_ratio = 0.5,
             file_name = 'roomselection_1800.csv'):
    '''
    Load dataset
    '''

    # train_dataset = np.random.uniform(size=(1200, 6))

    ds = drs.load_csv(file_name)
    ds_ = ds.loc[:, ['Points:0', 'Points:1', 'Points:2', 'Pressure', 'Temperature']]

    ds_idxs = np.linspace(0, ds.shape[0] - 1, ds.shape[0]).T
    np.random.shuffle(ds_idxs)

    if data_cnt is not None:
        # data_cnt = 1200
        pass
    else:
        data_cnt = ds.shape[0]

    train_cnt = int(data_cnt * split_ratio)


    train_idxs = np.array(ds_idxs[:train_cnt])
    train_dataset = np.array( ds_.loc[train_idxs, :] )

    test_idxs = np.array(ds_idxs[train_cnt:data_cnt])
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

    # print('mean_check  : ', mean_check)
    # print('stddev_check: ', stddev_check)
    assert np.allclose(mean_check, 0., atol=0.1)
    assert np.allclose(stddev_check, 1., atol=0.1)

    return normal_train, normal_test


def make_model(encoded_size = 1,
               input_shape = 5,
               dense_layers_dims=[]):
    '''
    Simplify model
    '''
    # input_shape = datasets_info.features['image'].shape
    # input_shape = train_dataset.shape[1]
    # encoded_size = 1  # latent variable z has encoded_size dimensions

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
            *[tfkl.Dense(dim, activation=tf.nn.leaky_relu) for dim in dense_layers_dims],
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

        if 0<len(dense_layers_dims):
            dense_layers_dims.reverse()
            dense_layers_rdims = dense_layers_dims
        else:
            dense_layers_rdims = []

        decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=[encoded_size]),
            *[tfkl.Dense(dim, activation=tf.nn.leaky_relu) for dim in dense_layers_rdims],
            tfkl.Dense(input_shape, activation=tf.nn.leaky_relu),
        ])

        tf.summary.histogram('decoded_dense_outputs', decoder.outputs)

        decoder_head = tfk.Sequential([
            tfkl.InputLayer(input_shape=[input_shape]),
            tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
        ])

        print("--------------")


        # output defined as composition of encoder and decoder
        #   only need to specify the reconstruction loss: first term of ELBO
        vae = tfk.Model(inputs=encoder.inputs,
                        # outputs=decoder_head(decoder(
                        outputs=decoder_head(decoder(
                                 encoder_head(encoder.outputs[0])
                        )))

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


        def calculate_log_mse(x, rv_x):
            return tf.log(tf.losses.mean_squared_error(x, rv_x))


        def custom_metrics_max_diff(x, x_hat):
            return tf.reduce_max(x-x_hat)


        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=calculate_neg_log_likelihood,
                    metrics=['mse', 'acc', custom_metrics_max_diff]
                    )

        summ = tf.summary.merge_all()
        saver = tf.train.Saver()

        return vae, summ, saver


def vae_fit(vae,
            summ,
            saver,
            normal_train,
            normal_test,
            epochs=10000,
            steps_per_epoch=200,
            logdir=None,
            tag=''):

    if logdir is None:
        sep = '-' if 0<len(tag) else ''
        logdir = 'vae-5to1/' + tag + sep + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    epoch_print_callback = tf.keras.callbacks.LambdaCallback(on_epoch_begin=
                                                             lambda epoch, logs: tf.print('epoch', epoch))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)  # before init

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # after the init

        writer = tf.summary.FileWriter(logdir)
        writer.add_graph(sess.graph)

        # saver.restore(sess, logdir + '/headless_train.ckpt')

        # output of TFP layer should be a distribution object
        history = vae.fit(
                x=normal_train,
                y=normal_train,
                epochs=epochs,
                batch_size=normal_train.shape[0],
                steps_per_epoch=steps_per_epoch,

                validation_data=(normal_test, normal_test),
                validation_steps=1,

                callbacks=[
                    tensorboard_callback
                ]
                )
                #, distribute=strategy)

        save_path = saver.save(sess, logdir+'/headless_train.ckpt')
        vae.summary()

        return history


def train_experiment():
    normal_train, normal_test = get_data(data_cnt=1200,
                                         split_ratio=0.5,
                                         file_name='roomselection_1800.csv')

    vae, summ, saver = make_model(encoded_size=3,
                                  input_shape=5,
                                  dense_layers_dims=[])

    history = vae_fit(vae, summ, saver,
                      normal_train, normal_test,
                      10000,
                      200,
                      tag='5x3_bern_b')

    plts.plot_history(12, 4, history)


if __name__ == '__main__':
    train_experiment()
    pass
