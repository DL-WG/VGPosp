import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.keras import backend as k
from tensorflow.python.framework import ops
ops.reset_default_graph()

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

import numpy as np




LOGDIR = "./log_dir_graph_tests/"

a_in = 5
b_in = 15


    # a = tf.placeholder(dtype=float,name="a")
    # b = tf.placeholder(dtype=float,name="b")
    # prod = tf.multiply(a, b, name="Multiply")
    # sum = tf.add(a, b, name="Add")
    # result = tf.divide(prod, sum, name="Divide")
    #
    # prod_ = sess.run(prod, feed_dict={a:a_in, b : b_in})
    # sum_ = sess.run(sum, feed_dict={a:a_in, b : b_in})
    # result_ = sess.run(result, feed_dict={a:a_in, b : b_in})

    # creating model 1 and model 2 -- the "previously existing models"
# creating model 3, joining the models
from keras import backend as K
graph = tf.Graph()
K.set_session(tf.Session(graph=graph))
with graph.as_default():
    sess = tf.Session()

    encoded_size = 1
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                            reinterpreted_batch_ndims=1)


    with tf.name_scope("encoder"):
        m1_in = tf.placeholder(dtype=float, shape=20)
        m1 = tfk.Sequential(None, name="en_")
        m1.add(tfkl.Dense(20, input_shape=(50,), name="1_"))
        m1.add(tfkl.Dense(30, name="2_"))
        m1.add(tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
                                  activation=None, name="3_"))
        m1.add(tfpl.MultivariateNormalTriL(encoded_size,activity_regularizer=tfpl.KLDivergenceRegularizer(prior),name="4_"))
        m1.add(tfkl.Dense(30, name="5_"))

    with tf.name_scope("decoder"):

        m2 = tfk.Sequential(None, name="de_")
        m2.add(tfkl.Dense(5, input_shape=(30,), name="6_"))
        m2.add(tfkl.Dense(11, name="7_"))


    with tf.name_scope("model"):

    #     o = m2(m1.outputs)
    #     m3 = tfk.Model(m1.inputs, o, name="model")
    # with tf.name_scope("model"):
        m3 = tfk.Model(inputs=m1.inputs, outputs=m2(m1.outputs))
    # out2 = m2(m1.outputs)
    # m3 = tfk.Model(m1.inputs, out2)

# checking out the results
    m3.summary()

    # layers in model 3
    print("\nthe main model:")
    for i in m3.layers:
        print(i.name)

    # layers inside the last layer of model 3
    # print("\ninside the submodel:")
    # for i in m3.layers[-1].layers:
    #     print(i.name)


    # print(prod_, sum_, result_)
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)


# saver = tf.train.Saver()

