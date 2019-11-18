import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import numpy as np
import collections
import placement_algorithm2 as alg2
import tensorflow_probability as tfp
import sys
import snippets
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt


def load_cov_vv(file_name='cov_vv.csv'):

    df = pd.read_csv(file_name, encoding='utf-8', engine='c')
    cov_vv = np.array(df.iloc[:, 1:])

    # times = list(df['times'])
    return cov_vv


def save_cov_vv(cov_vv, file_name='cov_vv.csv'):

    df = pd.DataFrame(cov_vv)
    df.to_csv(file_name)
    pass


def test_save_cov_vv():

    dim = 11
    U = np.random.normal(1, 1, size=(dim, dim))
    U_UT = np.dot(U, U.T)
    cov_vv_np = U_UT  + 0.001*np.eye(dim)  # Sigma_pos_definite + diagonal(1)

    save_cov_vv(cov_vv_np, 'test_cov_vv.csv')
    cov_vv_back = load_cov_vv('test_cov_vv.csv')

    assert np.allclose(cov_vv_back, cov_vv_np)

    pass


if __name__ == '__main__':
    test_save_cov_vv()
    pass
