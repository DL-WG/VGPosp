import numpy as np
import pandas as pd
import tensorflow as tf

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

tf.Module._TF_MODULE_IGNORED_PROPERTIES = set()

tfk = tf.keras
tfkb = tf.keras.backend


#=======================================================
# Graph
#=======================================================
def get_cell_id(i_0, i_1, i_2, I0, I1, I2, SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z): #  j_0, j_1, j_2, i):
    # calculate bucket number and return it.

    cell_x_i = i_0 * (1 / I0) // (1 / SPLIT_cell_x)  # tf.floordiv(tf.math.divide(i_0, I0), tf.math.divide(1, SPLIT_bkt_x))
    cell_y_i = i_1 / I1 // (1 / SPLIT_cell_y)
    cell_z_i = i_2 / I2 // (1 / SPLIT_cell_z)
    cell_id = cell_x_i + (cell_y_i * SPLIT_cell_x) + (cell_z_i * SPLIT_cell_x * SPLIT_cell_y)

    return cell_id


def get_tf_cell_id(i_0, i_1, i_2, I0, I1, I2, SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z): #  j_0, j_1, j_2, i):
    # calculate bucket number and return it.

    cell_x_i = i_0 * (1 / I0) // (1 / SPLIT_cell_x)  # tf.floordiv(tf.math.divide(i_0, I0), tf.math.divide(1, SPLIT_bkt_x))
    cell_y_i = i_1 / I1 // (1 / SPLIT_cell_y)
    cell_z_i = i_2 / I2 // (1 / SPLIT_cell_z)
    cell_id = cell_x_i + (cell_y_i * SPLIT_cell_x) + (cell_z_i * SPLIT_cell_x * SPLIT_cell_y)

    return cell_id


def sample_dataset(dataset_shape, cell_id):
    # takes a constant to make shapes of placeholders: timesteps_in_cell, cell_id and dataset

    cell_ts_pl = tf.placeholder(shape=(), dtype=np.float64, name='cell_timesteps')
    dataset_pl = tf.placeholder(shape=dataset_shape, dtype=np.float64, name='dataset')
    cell_selection = tf.slice(dataset_pl, begin=[tf.cast(cell_id * cell_ts_pl, tf.int64), 0],
                              size=[cell_ts_pl, dataset_pl.shape[1]],
                              name="cell_ts_pl")

    op1 = tf.Assert(tf.equal(cell_selection[0, 0], cell_id), [cell_selection[0, :], cell_ts_pl, cell_id])
    op2 = tf.Assert(tf.equal(cell_selection[-1, 0], cell_id), [cell_selection[0, :], cell_ts_pl, cell_id])

    # Extracts x[0, 1:2, :] == [[[ 3.,  4.]]]
    # cell_selection = dataset_pl[[cell_id*cell_ts_pl :  ], [2,3]]
    # cell_selection = tf.slice(dataset_pl, begin=[cell_id*cell_ts_pl, 2], size=[cell_ts_pl, 2])
    # cell_ts_pl = tf.shape(tf.squeeze(cell_ts_pl, [1]))
    # idx = tf.fill(dims=tf.cast(cell_ts_pl, tf.int64), value=2)
    # cell_selection = tf.gather_nd(dataset_pl, tf.stack([tf.range(cell_ts_pl), idx], axis=0))
    with tf.control_dependencies([op1, op2]):
        return cell_ts_pl, dataset_pl, cell_selection


#=======================================================
# Tests
#=======================================================
def test_cell_id_getter():
    i_0 = 3
    i_1 = 0
    i_2 = 0
    I0 = 4
    I1 = 4
    I2 = 1
    SPLIT_cell_x = 4
    SPLIT_cell_y = 4
    SPLIT_cell_z = 1
    #
    cell_id_ = get_tf_cell_id(i_0, i_1, i_2, I0, I1, I2, SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z)
    return cell_id_


def test_tf_cell_id_getter():
    i_0 = tf.constant(3, tf.float64)
    i_1 = tf.constant(0, tf.float64)
    i_2 = tf.constant(0, tf.float64)
    I0 = tf.constant(4, tf.float64)
    I1 = tf.constant(4, tf.float64)
    I2 = tf.constant(1, tf.float64)
    SPLIT_cell_x = tf.constant(4, tf.float64)
    SPLIT_cell_y = tf.constant(4, tf.float64)
    SPLIT_cell_z = tf.constant(1, tf.float64)
    #
    cell_id_ = get_cell_id(i_0, i_1, i_2, I0, I1, I2, SPLIT_cell_x, SPLIT_cell_y, SPLIT_cell_z)
    return cell_id_



def test_sample_dataset_cell(dataset, cell_id, timesteps_in_cell):

    cell_selection_ = sample_dataset(dataset, cell_id, timesteps_in_cell)
    return cell_selection_


if __name__ == '__main__':

    SET_ID = 2
    df1 = pd.read_csv('./dataset_2_cov/dataset_2_cov_' + str(SET_ID) + '.csv')
    df1.drop(columns=['Unnamed: 0'], inplace=True)
    TIMESTEPS_IN_CELL =250

    cell_id = test_tf_cell_id_getter()  # test returns cell_id = 3


    cell_ts_pl, dataset_pl, cell_selection = sample_dataset(df1.shape, cell_id)

    with tf.Session() as sess:
        tfkb.set_session(sess)  # keras backend joined
        sess.run(tf.global_variables_initializer())  # after the init
        cell_id_ = sess.run(cell_id)
        cell_selection_ = sess.run(cell_selection,
                                   feed_dict={cell_ts_pl: TIMESTEPS_IN_CELL,
                                              dataset_pl: df1.to_numpy()})

    pass