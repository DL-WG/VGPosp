import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers
from tensorflow_probability import positive_semidefinite_kernels as tfkern

import numpy as np
# import gp_functions as gpf
# import plots as plts
# import data_generation as dg
import matplotlib.pyplot as plt
import time

import gp_functions as gpf
import snippets

def sparse_argmax_cache_linear(cache, A, V):
    # y_st = -1  # index of coordinate
    # delta_st = -1

    V_minus_A = tf.sets.difference(V, A)
    delta_y_s = tf.reshape(tf.gather_nd(
                            tf.reshape(cache, [-1, 1]),
                            tf.reshape(V_minus_A.indices[:, 0], [-1, 1])
                           ), [-1])
    delta_st = tf.reduce_max(delta_y_s)  # max
    val_idx = tf.where(tf.equal(delta_y_s, delta_st)) # index of max
    # # val_col = 0
    # # index_col = 1
    idx_st = val_idx[0, 0]  # get index of delta_st
    y_st = V_minus_A.values[idx_st]

    # y_st = snippets.tf_print2(y_st,
    #             [cache,
    #              V_minus_A.values,
    #              delta_y_s,
    #              delta_st,
    #              val_idx, idx_st,
    #              y_st],
    #             'sparse_argmax_cache_linear')

    # now we have considered all y-s in the cache, and the winner is y_st
    return y_st


def argmax_cache_linear(cache, A, V):
    y_st = -1  # index of coordinate
    delta_st = -1
    for y in V:
        if y in A:  # if y in A then don't consider: continue
            continue

        delta_y = cache[y]

        if delta_st < delta_y:
            delta_st = delta_y  # update largest delta yet, delta_star
            y_st = y  # update index

    # now we have considered all y-s in the cache, and the winner is y_st
    return y_st


def tf_argmax_cache_linear(cache, A, V):
    y_st = tf.Variable([-1])  # index of coordinate
    delta_st = tf.Variable([-1])

    delta_y = tf.Variable([], name="delta_y")
    tf.constant(0)
    tf.size(V)
    # cond_a = ()
    # def body_a():
    #     return
    # while_loop()  # from 0 to num of V. Each iter take V[y] and check if in A with map_fn and reduce_any
    #
    # elems = tf.constant([1, 2, 3, 4, 5, 6])
    # y = tf.constant(3)
    # eq = tf.map_fn(lambda x, y: x == y, elems)  # returns tf.Tensor([False, False, True, False, False, False])
    # is_eq = tf.reduce_any(eq)
    #
    #     tf.case([(tf.less(delta_st, delta_y), f1], default=f2)  # if delta_st < delta_y:
    #
    # assign_op = delta_y.assign(tf.slice(cache, [y], size=1))

    for y in V:
        if y in A:  # if y in A then don't consider: continue
            continue

        delta_y = cache[y]
        if delta_st < delta_y:
            delta_st = delta_y  # update largest delta yet, delta_star
            y_st = y  # update index


    # now we have considered all y-s in the cache, and the winner is y_st
    return y_st


def argmax_(A, A_bar, V, cov_vv):
    y_st = -1  # index of coordinate
    delta_st = -1

    for y in V:  # improved in algorithm 2
        if y in A:  # if y in A then dont consider: continue
            continue  # skip delta
        else:  # y is not in A, consider, calc delta.
            nom = nominator(y, A, cov_vv)
            denom = denominator(y, A_bar, cov_vv)
            # print('nom=', nom, 'denom=', denom)
            if np.abs(denom) < 1e-8 or np.abs(nom) < 1e-8:
                delta_y = 0
            else:
                delta_y = nom / denom

        if delta_st < delta_y:
            delta_st = delta_y  # update largest delta yet, delta_star
            y_st = y  # update index

    return y_st, delta_st


def placement_algorithm_1(cov_vv, k):
    '''
    assume U is empty
    assume V = S, indexes where there may be sensors [0 .. cov_vv.shape[0])
    A - selection of placements eventually lngth = k
    y = len(V) + len(S) -> MOVE selections from V to S
    '''
    A = [] # selected_indexes
    V = np.linspace(0, cov_vv.shape[0]-1, cov_vv.shape[0], dtype=np.int64)
    A_bar = [] # complementer set to A.
    for i in V: # iterate indices where we can have sensors
        A_bar.append(i)

    while len(A) < k:  # "for j=1 to k"
        y_st, _ = argmax_(A, A_bar, V, cov_vv)  # choose y* mutual information which is the highest of the nom/denom in the formula given
        A.append(y_st)  # add largest MI's corresponding index
        A_bar.remove(y_st)  # remove it from complementer set
    return A  # A contains k elements.


#=========================================================
# Algorithm 2
#=========================================================
def placement_algorithm_2(cov_vv, k):
    '''
    assume U is empty
    assume V = S
    A - selection of placements eventually lngth = k
    y = len(V) + len(S) -> MOVE selections from V to S
    '''
    # I.
    A = [] # selected_indexes
    V = np.linspace(0, cov_vv.shape[0]-1, cov_vv.shape[0], dtype=np.int64)
    A_bar = [] # complementer set to A.
    delta_cached = []
    delta_cached_is_uptodate = []
    INF = 1e1000

    # II.
    for i in V: # iterate indices where we can have sensors
        A_bar.append(i)
        delta_cached.append(INF)
        delta_cached_is_uptodate.append(0)

    # III.
    while len(A) < k: # "for j=1 to k"
        y_st = -1  # index of coordinate
        delta_st = -1
        # "S \ A do current_y <- false"

        # IV.  set_all_delta_cach_to_false
        for i in range(len(delta_cached_is_uptodate)):  # in each round we make decision on updated value
            delta_cached_is_uptodate[i] = False

        # V.
        while True:
            y_st = argmax_cache_linear(delta_cached, A, V)  # re-implement with priority-queue

            # VI.
            if delta_cached_is_uptodate[y_st]:
                print('y*=', y_st)
                break

            # VII.  then_update_delta_y
            else:
                delta_y = 0
                nom = nominator(y_st, A, cov_vv)
                denom = denominator(y_st, A_bar, cov_vv)

                # VIII.  if_denom_is_near_zero + if_denom_is_near_zero_check
                if np.abs(denom) < 1e-8 or np.abs(nom) < 1e-8:  # here: delta_y = 0
                    pass

                # IX.
                else:
                    delta_y = nom / denom

                print('delta_y=', delta_y, 'y_st=', y_st)

                delta_cached[y_st] = delta_y
                delta_cached_is_uptodate[y_st] = True

        # X.
        A.append(y_st)  # add largest MI's corresponding index .. found so far,
                        # .. which is the first updated value in this round exceeding
                        # .. all the rest of not yet updated previous round values (that is "good enough")
        A_bar.remove(y_st)  # remove it from complementer set

        # XI.
        # plot3D(delta_cached, 12, 12)
        # print("plot, current A is:", A)
    return A  # A contains k elements.


#=========================================================
# Plot cache
#=========================================================
def plot3D(cache, figX, figY, xlabel=None, ylabel=None, zlabel=None):
    fig = plt.figure(figsize=(figX, figY))
    axFig = fig.gca(projection='3d')
    # axisX, axisY = np.meshgrid(calc_axisZ(edges), calc_axisZ(edges), sparse=False)  # 20x20
    # surf = axFig.plot_surface(axisX, axisY, H2, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)  # Surface
    axFig.set_xlabel(xlabel)  # Axis labels
    axFig.set_ylabel(ylabel)
    axFig.set_zlabel(zlabel)
    plt.show()


def tf_placement_algorithm_2(cov_vv, k):
    # build graph here
    # input cov
    # instanciate
    # sess.run
    # array as output
    '''
    assume U is empty
    assume V = S
    A - selection of placements eventually lngth = k
    y = len(V) + len(S) -> MOVE selections from V to S
    '''

    A = tf.Variable([], name="A_selected")
    A_bar = tf.Variable([], name="A_bar_unselected")
    INF = tf.constant([100000], name="inf")
    V = tf.cast(tf.linspace(0., tf.cast(cov_vv.shape[0]-1, dtype=tf.float32), cov_vv.shape[0]), dtype=tf.int64)

    delta_cached = tf.Variable([], name="delta_cached")
    delta_cached_is_uptodate = tf.Variable([], name="delta_cached_is_uptodate")

    i = tf.constant(0)
    cond_v = lambda i : tf.less(i, input)
    def body_v(i):
        tf.stack(A_bar, i)
        tf.stack(delta_cached, INF)
        tf.stack(delta_cached_is_uptodate, 0)
        return [tf.add(i, 1)]
    loop_v = tf.while_loop(cond_v, body_v, [i])

    k = tf.constant(5)
    len_A = tf.constant(tf.size(A))
    cond_k = lambda k, A : tf.less(len_A, k)
    def body_k(len_A):
        y_st = tf.Variable([-1], name="y_st")
        delta_st = tf.Variable([-1], name="delta_st")
        # "S \ A do current_y <- false"

        d = tf.constant(0)
        cond_d = lambda d : tf.less(d, tf.size(delta_cached_is_uptodate))
        def body_d():
            indexes = tf.cast([d], dtype=tf.int32, name='indexes')
            update = tf.constant(0, name="update")
            tf.scatter_nd_update(delta_cached_is_uptodate, [indexes], [update])  # [[i0, 0]], [3.+i0+0])
            return [tf.add(d, 1)]
        loop_d = tf.while_loop(cond_d, body_d, [d])

        y = tf.Variable([1])
        cond_y = lambda y : tf.math.equal(1, y)  # while true(1)
        def body_y():
            y_st = tf_argmax_cache_linear(delta_cached, A, V) # TODO

            slice = delta_cached_is_uptodate[y_st]
            f1 = lambda slice : tf.constant(0)
            f2  = lambda : tf.constant(1)
            d_true = tf.case()

            return

        loop_y = tf.while_loop(cond_y, body_y, [y])

        return [tf.add(len_A, 1)]

    loop_k = tf.while_loop(cond_k, body_k, [len_A])


###########################################################
    while len(A) < k: # "for j=1 to k"
        y_st = -1  # index of coordinate
        delta_st = -1
        for i in range(len(delta_cached_is_uptodate)):  # in each round we make decision on updated value
            delta_cached_is_uptodate[i] = False
###########################################################
        while True:
            y_st = argmax_cache_linear(delta_cached, A, V)  # re-implement with priority-queue

            if delta_cached_is_uptodate[y_st]:
                break
            else:
                delta_y = 0
                nom = nominator(y_st, A, cov_vv)
                denom = denominator(y_st, A_bar, cov_vv)
                if np.abs(denom) < 1e-8 or np.abs(nom) < 1e-8:
                    # delta_y = 0
                    pass
                else:
                    delta_y = nom / denom

                print('delta_y=', delta_y)

                delta_cached[y_st] = delta_y
                delta_cached_is_uptodate[y_st] = True

        A.append(y_st)  # add largest MI's corresponding index .. found so far,
                        # .. which is the first updated value in this round exceeding
                        # .. all the rest of not yet updated previous round values (that is "good enough")
        A_bar.remove(y_st)  # remove it from complementer set
    return A  # A contains k elements.


def tf_nominator(y, Ain, cov_vv):  # A may comes in
    A = tf.sets.union(Ain, y)

    # sigm_yy = tf.slice(cov_vv, [y.indices[:, 0][0], y.indices[:, 0][0]], [1, 1])
    y_idx = y.indices[:, 0][0]
    sigm_yy = tf.reshape(tf.gather_nd(cov_vv, [ [y_idx, y_idx] ]), [1, 1])

    cov_yA_row = tf.reshape(tf.slice(cov_vv, [y_idx, 0], [1, cov_vv.shape[1]])[0], [1, -1])
    A_idxs_ = A.indices[:, 0]
    A_idxs = tf.stack([tf.zeros_like(A_idxs_), A_idxs_], axis=1)
    cov_yA = tf.gather_nd(cov_yA_row, [ A_idxs ] )[0]

    A_idxs_d = tf.expand_dims(A_idxs_, axis=1)
    cov_AA_rows = tf.gather_nd(cov_vv, A_idxs_d)
    cov_AA_rowsT = tf.transpose(cov_AA_rows)
    cov_AAT = tf.gather_nd(cov_AA_rowsT, A_idxs_d)
    cov_AA = tf.transpose(cov_AAT)


    cov_Ay_col = tf.slice(cov_vv, [0, tf.reshape(y_idx, [])], [cov_vv.shape[0], 1])[:, 0]
    cov_Ay = tf.gather(cov_Ay_col, [ A_idxs_ ])[0]

    pinv = tfp.math.pinv(cov_AA)
    cov_yAT = tf.transpose(cov_yA)
    mul1 = tf.tensordot(cov_yAT, pinv, [[0], [0]])
    nom = sigm_yy - tf.tensordot(mul1, cov_Ay, [[0], [0]])

    return nom


def tf_denominator(y, A_hat, cov_vv):
    A_hat_ = tf.sets.difference(A_hat, y)
    return tf_nominator(y, A_hat_, cov_vv)


def nominator(y, A, cov_vv):
    A_ = list(A.copy())
    # A_.append(int(y))
    # s = slice(1, 3)
    sigm_yy = make_slice(cov_vv, [y], [y])

    if 0 == len(A):
        retv = sigm_yy
    else:
        cov_yA = make_slice(cov_vv, [y], A_)
        cov_AA = make_slice(cov_vv, A_, A_)
        cov_Ay = make_slice(cov_vv, A_, [y])
        inv_cov_AA = call_pinv(cov_AA)
        dot_yA_iAA = np.dot(cov_yA, inv_cov_AA)
        dot_yAiAA_Ay = np.dot(dot_yA_iAA, cov_Ay)
        retv = sigm_yy - dot_yAiAA_Ay
    # print('np_nom', retv)
    return retv


def make_slice(cov_vv, y, A):
    cov_yA = np.zeros(shape=[len(y), len(A)])
    for i in range(cov_yA.shape[0]):
        for j in range(cov_yA.shape[1]):
            cov_yA[i,j] = cov_vv[y[i], A[j]]
    return cov_yA


def call_pinv(a):
    assert a.shape[0] == a.shape[1]
    if a.shape[0] == 1:
        return 1 / a
    else:
        r = np.linalg.pinv(a)
        return r


def denominator(y, A_hat, cov_vv):
    A_hat_ = list(A_hat.copy())
    if y in A_hat_:
        A_hat_.remove(int(y))

    return nominator(y, A_hat_, cov_vv)


def calculate_running_time_algorithm_1(cov_vv, k):
    strt = time.time()
    pl = gpf.placement_algorithm_1(cov_vv, k)
    print("placement_array : ",pl)
    end = time.time()
    return (end - strt)


def calculate_running_time_algorithm_2(cov_vv, k):
    strt = time.time()
    # pl = tfplacement_algorithm_2(cov_vv, k)
    pl = placement_algorithm_2(cov_vv, k)
    print("placement_array : ",pl)
    end = time.time()
    return (end - strt)


def plts_plot_algorithm_scale(figx, figy, n_array, time_array):
    plt.figure(figsize=(figx, figy))
    length = time_array.shape[0]
    for i in range(length):
        plt.scatter(n_array[i], time_array[i], color='grey', s=40)  # observations
    plt.show()


def dg_create_random_cov(n):
    matrix_rand = np.random.uniform(0,1,n**2).reshape(-1,n)
    cov_vv = np.dot(matrix_rand, matrix_rand.T)
    return cov_vv


def test_algorithm_2():
    """algorithm 1 improved with priority queue
    gpf.placement_algorithm_2

    we time the algorithm for each increase in the size of the cov input
    """
    k = 5
    num_of_cov_sizes = 40
    n_sizes = np.linspace(10, num_of_cov_sizes, num_of_cov_sizes - 10, dtype=int)

    # n_array = np.array([10,20,30,40,50])
    n_array = np.array([])
    for i in n_sizes:
        n_array = np.append(n_array, i)

    times_array = np.array([])
    for n in n_array:
        n = int(n)
        cov = dg_create_random_cov(n)  # ndarray
        r_time = calculate_running_time_algorithm_2(cov, k)  # Algorithm 2
        times_array = np.append(times_array, r_time)

    plts_plot_algorithm_scale(12, 4, n_array, times_array)
    pass


def cov_vv_4x4():
    cov_vv = np.array(
        [[1.10, 0.31, 0.33, 0.27],
         [0.31, 1.01, 0.30, 0.27],
         [0.33, 0.30, 0.97, 0.33],
         [0.27, 0.27, 0.33, 1.2]])
    return cov_vv


def test_placement_algorithm_1():
    cov_vv = cov_vv_4x4()
    A = placement_algorithm_1(cov_vv, 4)
    print('algo1 A: ', A)
    pass


def test_placement_algorithm_2():
    cov_vv = cov_vv_4x4()
    A = placement_algorithm_2(cov_vv, 4)
    print('algo2 A: ', A)
    pass


if __name__ == '__main__':
    test_placement_algorithm_1()
    test_placement_algorithm_2()
    test_algorithm_2()

