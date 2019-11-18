# This example fits a logistic regression loss.
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
LOGDIR = "./log_dir_regression/"

# Create fictitious training data.
from ipython_genutils.py3compat import xrange

dtype = np.float32
n = 3000    # number of samples
x_size = 4  # size of single x
# def make_training_data():
np.random.seed(142)
x = np.random.randn(n, x_size).astype(dtype)
w = np.random.randn(x_size).astype(dtype)
b = np.random.randn(1).astype(dtype)
true_logits = np.tensordot(x, w, axes=[[-1], [-1]]) + b
noise = np.random.logistic(size=n).astype(dtype)
y = dtype(true_logits + noise > 0.)
  # return y, x
# y, x = make_training_data()

# Build TF graph for fitting Bernoulli maximum likelihood estimator.
with tf.name_scope("bernoulli_"):
  bernoulli = tfp.trainable_distributions.bernoulli(x)
with tf.name_scope("loss_"):
  loss = -tf.reduce_mean(bernoulli.log_prob(y), name="reduce_mean_")
with tf.name_scope("train_op_"):
  train_op = tf.train.AdamOptimizer(learning_rate=2.**-5).minimize(loss)
with tf.name_scope("mse_"):
  mse = tf.reduce_mean(tf.squared_difference(y, bernoulli.mean()))
with tf.name_scope("init_op_"):
  init_op = tf.global_variables_initializer()

# Run graph 1000 times.
with tf.name_scope("train_100_"):
  num_steps = 2000
  loss_ = np.zeros(num_steps)   # Style: `_` to indicate sess.run result.
  mse_ = np.zeros(num_steps)

with tf.Session() as sess:
  sess.run(init_op)
  for it in xrange(loss_.size):
    _, loss_[it], mse_[it] = sess.run([train_op, loss, mse])
    if it % 200 == 0 or it == loss_.size - 1:
      print("iteration:{}  loss:{}  mse:{}".format(it, loss_[it], mse_[it]))

# sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
saver = tf.train.Saver()




# ==> iteration:0    loss:0.635675370693  mse:0.222526371479
#     iteration:200  loss:0.440077394247  mse:0.143687799573
#     iteration:400  loss:0.440077394247  mse:0.143687844276
#     iteration:600  loss:0.440077394247  mse:0.143687844276
#     iteration:800  loss:0.440077424049  mse:0.143687844276
#     iteration:999  loss:0.440077424049  mse:0.143687844276


# import tensorflow as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions
#
# import plots as plts
# import gp_functions as gpf
# import data_generation as dg
#
# train_dataset = dg.generate_4d_polinomials()
# INPUT_SHAPE = train_dataset.shape[1]
# test_dataset = dg.generate_4d_polinomials()
# trainA = gpf.create_dataframe_from_4d_dataset(train_dataset)
# plts.pairplots(trainA)
#
# negloglik = lambda y, p_y: -p_y.log_prob(y)
#
# # Build model.
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(1),
#   tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
# ])
#
# # Do inference.
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
# model.fit(train_dataset, train_dataset, epochs=500, verbose=False)
#
# # Make predictions.
# yhat = model(x_tst)
#
# print("1-----------------------")
#
#
# # # Build model.
# # model = tfk.Sequential([
# #   tf.keras.layers.Dense(1 + 1),
# #   tfp.layers.DistributionLambda(
# #       lambda t: tfd.Normal(loc=t[..., :1],
# #                            scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
# # ])
# #
# # # Do inference.
# # model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
# # model.fit(x, y, epochs=500, verbose=False)
# #
# # # Make predictions.
# # yhat = model(x_tst)
# #
# # mean = yhat.mean()
# # stddev = yhat.stddev()
# # mean_plus_2_stddev = mean - 2. * stddev
# # mean_minus_2_stddev = mean + 2. * stddev
# #
# #
# #
# # # Build model.
# # model = tf.keras.Sequential([
# #   tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable),
# #   tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
# # ])
# #
# # # Do inference.
# # model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
# # model.fit(x, y, epochs=500, verbose=False)
# #
# # # Make predictions.
# # yhats = [model(x_tst) for i in range(100)]
# #
# #
# # # Build model.
# # model = tf.keras.Sequential([
# #   tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable),
# #   tfp.layers.DistributionLambda(
# #       lambda t: tfd.Normal(loc=t[..., :1],
# #                            scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))),
# # ])
# #
# # # Do inference.
# # model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
# # model.fit(x, y, epochs=500, verbose=False);
# #
# # # Make predictions.
# # yhats = [model(x_tst) for _ in range(100)]
# #
# #
# # num_inducing_points = 40
# # model = tf.keras.Sequential([
# #     tf.keras.layers.InputLayer(input_shape=[1], dtype=x.dtype),
# #     tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
# #     tfp.layers.VariationalGaussianProcess(
# #         num_inducing_points=num_inducing_points,
# #         kernel_provider=RBFKernelFn(dtype=x.dtype),
# #         event_shape=[1],
# #         inducing_index_points_initializer=tf.constant_initializer(
# #             np.linspace(*x_range, num=num_inducing_points,
# #                         dtype=x.dtype)[..., np.newaxis]),
# #         unconstrained_observation_noise_variance_initializer=(
# #             tf.constant_initializer(
# #                 np.log(np.expm1(1.)).astype(x.dtype))),
# #     ),
# # ])
# #
# # # Do inference.
# # batch_size = 32
# # loss = lambda y, rv_y: rv_y.variational_loss(
# #     y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0])
# # model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=loss)
# # model.fit(x, y, batch_size=batch_size, epochs=1000, verbose=False)
# #
# # # Make predictions.
# # yhats = [model(x_tst) for _ in range(100)]
