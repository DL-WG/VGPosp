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


def graph_algo2_time_by_cov_vv_size():

    df = pd.read_csv('algo2_times_vs_dims.csv', encoding='utf-8', engine='c')

    times = list(df['times'])
    dims = list(df['dims'])
    # model: times = b * dim^a
    # => log(times) = a * log(dim) + log(b)

    ax2 = df.plot.scatter('dims', 'times')
    plt.title('algo2 times vs. dims')
    plt.show()
    pass

    ax2_ = plt.scatter(np.log(dims), np.log(times))
    plt.title('algo2 ln(times) vs. ln(dims)')
    plt.show()
    pass

if __name__ == '__main__':
    graph_algo2_time_by_cov_vv_size()
    pass
