import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import numpy as np
import collections
import placement_algorithm2 as alg2
import tensorflow_probability as tfp
import sys
sys.path.insert(0, '..')
sys.path.insert(0, 'main_tests')

import snippets
import time
import datetime
import pandas as pd
import snippets_save
# import main_architecture_2_sampledistribution as arch2samp
import main_tests.alg2_split10x10x10_50x50x10_tests as a2t

# (loop_V) II. --------------------------------------------------------
def loop_1(begin, end, A_bar):

    cond_1 = lambda i_, N_, _: tf.less(i_, N_)
    def body_1(i_, N_, A_bar_):

        i0 = tf.SparseTensor(indices=[[i_, 0]], values=[tf.cast(i_, dtype=tf.int64)], dense_shape=(N_, 1))
        A_bar_ = tf.sets.union(A_bar_, i0)

        with tf.control_dependencies([A_bar_.values]):
            i_ret = tf.add(1, i_)

        return i_ret, N_, A_bar_

    # lvd = {'A_bar': A_bar}
    [loop, _, A_bar] = tf.while_loop(cond_1, body_1, loop_vars=[begin, end, A_bar])
    # loop = snippets.tf_print2(loop, [A_bar.values], 'v: ')
    return loop, A_bar


# XX, tf.function test --------------------------------------------------------
def loop_0(begin, end, A_bar):  # loop_V

    cond_1 = lambda i_, N_, _: tf.less(i_, N_)
    def body_1(i_, N_, A_bar_):

        i0 = tf.SparseTensor(indices=[[i_, 0]], values=[tf.cast(i_, dtype=tf.int64)], dense_shape=(N_, 1))
        A_bar_ = tf.sets.union(A_bar_, i0)

        inner_loop, A_bar_ = tf.function(loop_1)(0, end, A_bar_)

        with tf.control_dependencies([A_bar_.values, inner_loop]):
            i_ret = tf.add(1, i_)

        return i_ret, N_, A_bar_

    # lvd = {'A_bar': A_bar}
    [loop, _, A_bar] = tf.while_loop(cond_1, body_1, loop_vars=[begin, end, A_bar])
    # loop = snippets.tf_print2(loop, [A_bar.values], 'v: ')
    return loop, A_bar


#  I. --------------------------------------------------------
def while_param_propagation_test(cov_vv):  # loop_v

    N = cov_vv.shape[0]
    # A = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))
    A_bar = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))

    # cond_1 = lambda i_, N_, _: tf.less(i_, N_)
    # def body_1(i_, N_, A_bar_):
    #
    #     i0 = tf.SparseTensor(indices=[[i_, 0]], values=[tf.cast(i_, dtype=tf.int64)], dense_shape=(N, 1))
    #     A_bar_ = tf.sets.union(A_bar_, i0)
    #
    #     with tf.control_dependencies([A_bar_.values]):
    #         i_ret = tf.add(1, i_)
    #
    #     return i_ret, N_, A_bar_
    #
    # loop, _, A_bar = tf.while_loop(cond_1, body_1, loop_vars=[0, N, A_bar])

    loop, A_bar = tf.function(loop_0)(0, N.value, A_bar)  # loop_V
    # loop = snippets.tf_print2(loop, [A_bar.values], 'v: ')
    return loop, A_bar


# (body_A) IV. --------------------------------------------------------
def set_all_delta_cache_to_false(delta_cached_uptodate):
    '''
    for i in range(len(delta_cached_is_uptodate)):  # in each round we make decision on updated value
        delta_cached_is_uptodate[i] = False
    '''
    op = tf.assign(delta_cached_uptodate, tf.fill(tf.shape(delta_cached_uptodate), False))
    return op, delta_cached_uptodate


# TEST IV. ========================================================
def test__set_all_delta_cache_to_false(cov_vv):

    INF = tf.constant(1e1000, dtype=tf.float64)
    N = cov_vv.shape[0]

    delta_cached = tf.Variable(tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)),
                               dtype=tf.float64,
                               shape=[N, 1])
    delta_cached_uptodate = tf.Variable(tf.fill([N, 1], True), shape=[N, 1])  # needs to be a bool value

    delta_cached_setfalse, delta_cached_uptodate = set_all_delta_cache_to_false(delta_cached_uptodate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('delta_cached_uptodate: ', sess.run(delta_cached_uptodate))
        print('set_all_delta_cache_to_false: ', sess.run(delta_cached_setfalse))
    pass


def sparse_argmax_cache_linear(cache, A, V):
    # y_st = -1  # index of coordinate
    # delta_st = -1

    V_minus_A = tf.sets.difference(V, A)
    delta_y_s = tf.reshape(tf.gather_nd(
                            tf.reshape(cache, [-1, 1]),
                            tf.reshape(V_minus_A.indices[:, 0], [-1, 1])
                           ), [-1])
    delta_st = tf.reduce_max(delta_y_s)  # max
    val_idx = tf.where(tf.equal(cache, delta_st))
    # # val_col = 0
    # # index_col = 1
    y_st = val_idx[0, 0]  # get index of delta_st

    # now we have considered all y-s in the cache, and the winner is y_st
    return y_st


def tf_nominator(y, Ain, cov_vv):  # A may comes in
    # A = Ain  # tf.sets.union(Ain, y)
    A = tf.sets.difference(Ain, y)  # make sure that y is not in A

    # sigm_yy = tf.slice(cov_vv, [y.indices[:, 0][0], y.indices[:, 0][0]], [1, 1])
    y_idx = y.indices[:, 0][0]
    sigm_yy = tf.reshape(tf.gather_nd(cov_vv, [ [y_idx, y_idx] ]), [1, 1])

    def if_A_not_empty(y, A, cov_vv, y_idx, sigm_yy):

        cov_yA_row = tf.reshape(tf.slice(cov_vv, [y_idx, 0], [1, cov_vv.shape[1]])[0], [1, -1])
        A_idxs_ = A.indices[:, 0]
        A_idxs = tf.stack([tf.zeros_like(A_idxs_), A_idxs_], axis=1)
        cov_yA = tf.gather_nd(cov_yA_row, [ A_idxs ] )[0]

        A_idxs_d = tf.expand_dims(A_idxs_, axis=1)
        cov_AA_rows = tf.gather_nd(cov_vv, A_idxs_d)
        cov_AA_rowsT = tf.transpose(cov_AA_rows)
        cov_AAT = tf.gather_nd(cov_AA_rowsT, A_idxs_d)
        cov_AA = tf.transpose(cov_AAT)

        # numerical stability of pinv is helped by cov_AA += 1e-5 * eye
        # eye matrix has parameter n.
        cov_AA = tf.linalg.set_diag(cov_AA,
                                    tf.linalg.tensor_diag_part(cov_AA) +
                                    tf.constant(1e-6, dtype=tf.float64))

        cov_Ay_col = tf.slice(cov_vv, [0, tf.reshape(y_idx, [])], [cov_vv.shape[0], 1])[:, 0]
        cov_Ay = tf.gather(cov_Ay_col, [ A_idxs_ ])[0]

        # pinv, Danger of numerical representation loss. When n >> , then det cov_ij -> 0
        pinv = tfp.math.pinv(cov_AA)
        cov_yAT = tf.transpose(cov_yA)
        mul1 = tf.tensordot(cov_yAT, pinv, [[0], [0]])
        mul2 = tf.tensordot(mul1, cov_Ay, [[0], [0]])
        nom = sigm_yy - mul2

        # nom_pr = tf.print('\ny=', y,
        #                   # '\nAin=', Ain,
        #                   # '\ny_idx=', y_idx,
        #                   # '\nsigm_yy=', sigm_yy,
        #                   #
        #                   # '\ncov_yA_row=', cov_yA_row,
        #                   # '\nA_idxs_=', A_idxs_,
        #                   # '\nA_idxs=', A_idxs,
        #                   # '\ncov_yA=', cov_yA,
        #                   # '\ncov_yAT=', cov_yAT,
        #                   #
        #                   # '\nA_idxs_d=', A_idxs_d,
        #                   # '\ncov_AA_rows=', cov_AA_rows,
        #                   # '\ncov_AA_rowsT=', cov_AA_rowsT,
        #                   # '\ncov_AAT=', cov_AAT,
        #                   # '\ncov_AA=', cov_AA,
        #                   #
        #                   # '\ncov_Ay_col=', cov_Ay_col,
        #                   # '\ncov_Ay=', cov_Ay,
        #                   #
        #                   # '\npinv=', pinv,
        #                   # '\nmul1=', mul1,
        #                   # '\nmul2=', mul2,
        #                   #
        #                   # '\nnom=', nom
        #                   # '\ncov_vv=', cov_vv
        #                   )
        # with tf.control_dependencies([nom_pr]):
        return nom

    nom_ = tf.cond(tf.equal(0, tf.size(A.values)),
                   lambda: sigm_yy,
                   lambda: if_A_not_empty(y, A, cov_vv, y_idx, sigm_yy))

    # nom_pr2 = tf.print(['nom=', nom_],
    #                    output_stream=sys.stdout,
    #                    name='nom_pr2')
    # with tf.control_dependencies([nom_pr2]):
    return nom_


def tf_denominator(y, A_hat, cov_vv):
    A_hat_ = tf.sets.difference(A_hat, y)
    return tf_nominator(y, A_hat_, cov_vv)


# TEST V. ========================================================
def test__y_st_argmax_cache_linear(cov_vv, index=2, value=3., A_vals=[3, 4, 5, 6]):

    N = cov_vv.shape[0]
    default = 0.11

    # delta_cached size is :
    delta_cached = tf.Variable(tf.multiply(default, tf.ones([N, 1], dtype=tf.float64)),
                               dtype=tf.float64,
                               shape=[N, 1])
    set_delta_cached = tf.scatter_nd_update(delta_cached, [[index, 0]], [value])

    values01 = tf.cast(tf.linspace(0., tf.cast(cov_vv.shape[0]-1, tf.float32), cov_vv.shape[0]), tf.int64)  # values
    values02 = tf.cast(tf.linspace(0., 0., cov_vv.shape[0]), tf.int64)  # values
    indxes = tf.stack([values01, values02], axis=1)
    V = tf.SparseTensor(indices=indxes, values=values01, dense_shape=[N, 1])

    A = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))

    for i in A_vals:
        val = tf.SparseTensor(indices=[[i, 0]], values=[tf.cast(i, dtype=tf.int64)], dense_shape=[N, 1])
        A = tf.sets.union(A, val)

    with tf.control_dependencies([set_delta_cached]):
        y_st = tf.reshape(sparse_argmax_cache_linear(delta_cached, A, V), [])
        # y_st = alg2.sparse_argmax_cache_linear(delta_cached, A, V):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('V: ', sess.run(V.values))
        print(' - ')
        print('A: ', sess.run(A.values))
        print(' - ')
        # print('V_minus_A: ', sess.run(V_minus_A.values))
        # print('delta_y_s: ', sess.run(delta_y_s))
        # print('delta_st: ', sess.run(delta_st))
        # print('val_idx: ', sess.run(val_idx))
        print('y_st: ', sess.run(y_st))

    pass


# VI. --------------------------------------------------------
def if_delta_cached_is_uptodate(delta_cached, y_st):
    # [y] = tf.reshape(y_st.values[0], [-1])
    y = y_st.values[0]
    delta_cached = tf.reshape(delta_cached, [-1])
    s = tf.reshape(tf.slice(delta_cached, [y], [1]), [])
    false_fn = lambda : False
    true_fn = lambda: True

    bool_uptodate = tf.reshape(tf.case([(tf.equal(s, True), true_fn)], default=false_fn), [])
    return bool_uptodate


# TEST VI. ========================================================
def test__if_delta_cached_is_uptodate(cov_vv, cache_vals=True):

    INF = tf.constant(1e36)
    N = cov_vv.shape[0]
    delta_y = 0.1

    i = tf.cast(0, dtype=tf.int64)
    y_st = tf.SparseTensor(indices=[[i, 0]], values=[i], dense_shape=[N, 1])

    delta_cached_uptodate = tf.Variable(tf.fill([N, 1], cache_vals), shape=[N, 1])
    op2 = tf.scatter_nd_update(delta_cached_uptodate, [[y_st.values[0], 0]], [cache_vals])  # deltdelta_cached_uptodate[y_st] = True

    with tf.control_dependencies([op2]):
        bool_uptodate = if_delta_cached_is_uptodate(delta_cached_uptodate, y_st)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('idcnu_0: ', sess.run(bool_uptodate))
    pass
    # delta_cached = tf.Variable(tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)), dtype=tf.float64, shape=[N, 1])
    # i = tf.cast(0, dtype=tf.int64)
    # y_st = tf.SparseTensor(indices=[[i, 0]], values=[i], dense_shape=[N, 1])
    #
    # bool_uptodate = if_delta_cached_is_uptodate(delta_cached, y_st)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print('idcnu_0: ', sess.run(bool_uptodate))
    #
    # pass


# (loop_F) VII. --------------------------------------------------------
def if_delta_cached_is_not_uptodate(y_st, A, A_bar, cov_vv,
                                    delta_cached,
                                    delta_cached_uptodate,
                                    go):
    """
    Update the cache (the else branch here ...)

            if delta_cached_is_uptodate[y_st]:
                break
            else:
                delta_y = 0
       *        nom = nominator(y_st, A, cov_vv)
       *        denom = denominator(y_st, A_bar, cov_vv)
                if np.abs(denom) < 1e-8 or np.abs(nom) < 1e-8:
                    # delta_y = 0
                    pass
                else:
                    delta_y = nom / denom

       *        delta_cached[y_st] = delta_y
       *        delta_cached_is_uptodate[y_st] = True
    """
    # cov_vv = snippets.tf_print2(cov_vv_, [A.values, A_bar.values], 'if_delta_cached_is_not_uptodate: A, A_bar=\n')

    tf_nominator_ = tf.function(tf_nominator)
    tf_denominator_ = tf.function(tf_denominator)

    nom = tf_nominator_(y_st, A, cov_vv)
    denom = tf_denominator_(y_st, A_bar, cov_vv)

    # nom_ = snippets.tf_print2(nom, [nom, denom], 'if_delta_cached_is_not_uptodate: nom, denom=\n')

    denom_is_near_zero = if_denom_is_near_zero(nom, denom)

    delta_y = then_update_delta_y(
        denom_is_near_zero,  # True : don't do anything
        go,                  # False : don't do anything
        nom,
        denom)

    # if (!go) : we have delta_y == 0 and going to update the cache anyway
    # => let delta_y stay!

    # delta_y_pr2 = tf.print(['delta_y=', delta_y, 'y_st.values[0]', y_st.values[0]])
    # with tf.control_dependencies([delta_y_pr2]):
    # pr_op0 = tf.print(['nom=', nom, 'denom=', denom, 'delta_y=', delta_y],
    #                   output_stream=sys.stdout,
    #                   name='print_delta_y')

    with tf.control_dependencies([delta_y]):
        N = delta_cached.shape[0]

    delta_cached_v = tf.Variable(lambda: tf.zeros([N, 1], dtype=tf.float64),
                             dtype=tf.float64,
                             shape=[N, 1])
    delta_cached_uptodate_v = tf.Variable(lambda: tf.fill([N, 1], False), shape=[N, 1])

    op01 = tf.assign(delta_cached_v, delta_cached)                    # delta_cached[y_st] = delta_y
    op02 = tf.assign(delta_cached_uptodate_v, delta_cached_uptodate)  # delta_cached_is_uptodate[y_st] = True

    with tf.control_dependencies([cov_vv, delta_y, y_st.values, op01, op02]):



        # => let delta_y stay!
        op11 = tf.case([(go,
                       lambda: tf.scatter_nd_update(op01, [[y_st.values[0], 0]], [delta_y]))],  # delta_cached[y_st] = delta_y
                       default=lambda: op01)
        #
        op12 = tf.case([(go,
                       lambda: tf.scatter_nd_update(op02, [[y_st.values[0], 0]], [tf.cast(True, tf.bool)]))],
                       default=lambda: op02)
                                   # [tf.cast([True], tf.float32)])  # delta_cached_is_uptodate[y_st] = True

        # this should be the point when delta_cached gets updated EVEN FOR THE CURRENT SELECTED y_st
        # pr_op = tf.print(['op11=', op11])

    with tf.control_dependencies([op11, op12]):
        # return delta_cached_v, delta_cached_uptodate_v, cov_vv
        return op11, op12


# TEST VII. ========================================================
def test__if_delta_cached_is_not_uptodate(cov_vv, updated, go):

    INF = tf.constant(1e10, dtype=tf.float64)
    N = cov_vv.shape[0]
    A = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))
    A_bar = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))

    values01 = tf.cast(tf.linspace(0., tf.cast(cov_vv.shape[0]-1, tf.float32), cov_vv.shape[0]), tf.int64) # values
    values02 = tf.cast(tf.linspace(0., 0., cov_vv.shape[0]), tf.int64)  # values
    indxes = tf.stack([values01, values02], axis=1)
    V = tf.SparseTensor(indxes, values01, [cov_vv.shape[0], 1])

    v = 3
    y_st = tf.SparseTensor(indices=[[v, 0]], values=[tf.cast(v, dtype=tf.int64)], dense_shape=(N, 1))
    # A = tf.sets.union(A, y_st)
    A_bar = tf.sets.difference(V, A)

    delta_cached = tf.Variable(tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)),
                               dtype=tf.float64,
                               shape=[N, 1])
    delta_cached_uptodate = tf.Variable(tf.fill([N, 1], updated), shape=[N, 1])

    # cov_vv = snippets.tf_print2(cov_vv, [y_st.values, A.values, A_bar.values], 'test: A, A_bar=\n')

    # if_delta_cached_is_not_uptodate_ = tf.function(if_delta_cached_is_not_uptodate)

    # delta_cached2, delta_cached_uptodate2, cov_vv2 = if_delta_cached_is_not_uptodate(
    delta_cached2, delta_cached_uptodate2 = if_delta_cached_is_not_uptodate(
        y_st, A, A_bar, cov_vv,
        delta_cached,
        delta_cached_uptodate, go=tf.constant(go))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('test__if_delta_cached_is_not_uptodate')
        writer.add_graph(sess.graph)

        # TODO: here cov_vv2 is forcing if_delta_cached_is_not_uptodate to be called at all
        # maybe other than the Variables inside will be needed
        # print('idcnu_1: \n', sess.run([cov_vv2, delta_cached2, delta_cached_uptodate2]))
        print('idcnu_-: \n', sess.run([delta_cached, delta_cached_uptodate]))
        print('idcnu_1: \n', sess.run([delta_cached2, delta_cached_uptodate2]))
        print('^^^ updated=', updated, 'go=', go, 'expected to update cache[3] if not updated and go\n')

    pass


# (body_E) VIII. --------------------------------------------------------
def if_denom_is_near_zero_check(nom, denom):
    """
    figure out if denom is near zero, therefore
    delta_y shall not be calculated

                if np.abs(denom) < 1e-8 or np.abs(nom) < 1e-8:
                    # delta_y = 0
                    pass
                else: ...
    """
    nom = tf.cast(nom, dtype=tf.float64)
    denom = tf.cast(denom, dtype=tf.float64)

    small = tf.constant(1e-7, dtype=tf.float64)

    denom_is_small = tf.less(tf.abs(denom), small)  # if np.abs(denom) < 1e-8
    nom_is_zero = tf.less(tf.abs(nom), small)

    denom_is_near_zero = tf.logical_or(
        denom_is_small,
        nom_is_zero
    )
    return denom_is_near_zero, denom_is_small, nom_is_zero


# (body_E) VIII. --------------------------------------------------------
def if_denom_is_near_zero(nom, denom):
    """
    figure out if denom is near zero, therefore
    delta_y shall not be calculated

            if np.abs(denom) < 1e-8 or np.abs(nom) < 1e-8 :
                # delta_y = 0
                pass
            else: ...
    """
    nom = tf.cast(nom, dtype=tf.float64)
    denom = tf.cast(denom, dtype=tf.float64)

    small = tf.constant(1e-7, dtype=tf.float64)

    denom_is_near_zero = tf.logical_or(
        tf.less(tf.abs(denom), small),  # if np.abs(denom) < 1e-8
        tf.less(tf.abs(nom), small)
    )

    # true_if_fn_ = lambda: True
    # else_fn_ = lambda: False

    # denom_is_near_zero = tf.case([(
    #     tf.reshape(
    #         tf.logical_or(
    #             tf.less(tf.abs(denom), small),  # if np.abs(denom) < 1e-100
    #             tf.equal(nom,                   # or nom == 0:
    #                      tf.cast(0., dtype=tf.float64))
    #             ),
    #     []), true_if_fn_)], default=else_fn_)

    return denom_is_near_zero  # True / False


# TEST VIII. ========================================================
def test__if_denom_is_near_zero(nom, denom):

    denom_is_near_zero = tf.function(if_denom_is_near_zero)(nom, denom)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(' denom_is_near_zero: ', sess.run(denom_is_near_zero), nom, denom)

    pass


# IX. --------------------------------------------------------
def assign_delta_y(nom_, denom_ ):
    delta_y_ = tf.cast(tf.divide(tf.reshape(nom_, []), tf.reshape(denom_, [])), dtype=tf.float64)
    # delta_y = snippets.tf_print2(delta_y_, [delta_y_], 'tf.print delta_y_F: ')

    return delta_y_

# (loop_F) IX. --------------------------------------------------------
def then_update_delta_y(denom_is_near_zero, true_, nom, denom):
    """
    if denom is not zero, update delta_y

            delta_cached[y_st] = delta_y
            delta_cached_ is_uptodate[y_st] = True

    """
    delta_y = tf.Variable(0., dtype=tf.float64, name="delta_y")  # delta_y = 0

    cond_F = lambda denom_is_near_zero, true__, _, nom_, denom_: \
        tf.logical_and(
            tf.logical_not(denom_is_near_zero), true__)

    def body_F(denom_is_near_zero, true__, _, nom_, denom_):

        # delta_y_F_ = tf.cast(tf.divide(tf.reshape(nom_, []), tf.reshape(denom_, [])),dtype=tf.float64)
        # delta_y_F = snippets.tf_print2(delta_y_F_, [delta_y_F_], 'tf.print delta_y_F: ')
        delta_y_F = tf.function(assign_delta_y)(nom_, denom_)
        run_only_once = tf.constant(False)

        return [False, run_only_once, delta_y_F, nom_, denom_]  # is this ok ?

    [cd_F, true_F, delta_y, nom_F, denom_F] = tf.while_loop(
        cond_F, body_F,
        loop_vars=[tf.reshape(denom_is_near_zero, []), true_, delta_y, nom, denom],
        name='while_in_then_update_delta_y')

    return delta_y


# TEST IX. ========================================================
def test__then_update_delta_y(dnz, go):

    # delta_y = 0
    denom_is_near_zero = tf.constant(dnz)
    true_ = tf.constant(go)
    nom = tf.constant(0.1)
    denom = tf.constant(0.001)

    delta_y_ = then_update_delta_y(  # TODO tf.function
        denom_is_near_zero,  # True : don't do anything
        true_,               # False : don't do anything
        nom,
        denom)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(' - ')
        print(' denom_is_near_zero: ', sess.run(denom_is_near_zero))
        print(' true_: ', sess.run(true_))
        print(' delta_y_: ', sess.run(delta_y_))  # expect 100.
    pass


# X. --------------------------------------------------------
def append_to_A_remove_from_A_bar(A, A_bar, y_st):

    A_ = tf.sets.union(A, y_st)
    A_bar_ = tf.sets.difference(A_bar, y_st)

    return A_, A_bar_
    pass


# TEST X. ========================================================
def test__append_to_A_remove_from_A_bar(cov_vv):

    N = cov_vv.shape[0]
    A = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))
    A_bar = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))
    # y_st = tf.SparseTensor(indices=[[1, 0]], values=[tf.cast(1, dtype=tf.int64)], dense_shape=(N, 1))
    i = tf.cast(tf.constant(0), dtype=tf.int64)
    y_st = tf.SparseTensor(indices=[[i, 0]], values=[i], dense_shape=[N, 1])

    A_bar = tf.sets.union(A_bar, y_st)

    A_, A_bar_ = append_to_A_remove_from_A_bar(A, A_bar, y_st)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(' A_: ', sess.run(A_.values))
        print(' A_bar_: ', sess.run(A_bar_.values))
    pass


# V.
def while_true_outside(A_A, A_bar_A, V, delta_cached_A, delta_cached_uptodate_A, cov_vv_A):

    cond_D = lambda true, _, A_D, A_bar_D, delta_cached_D, delta_cached_uptodate_D, cov_vv_D: true  # while True:
    def body_D(true, y_st_D, A_D, A_bar_D, delta_cached_D, delta_cached_uptodate_D, cov_vv_D):

        true, y_st_D, A_D, A_bar_D, delta_cached_D, \
        delta_cached_uptodate_D = while_true_inside(
            true, A_D, A_bar_D, V, delta_cached_D, delta_cached_uptodate_D, cov_vv_D)

        # delta_cached_D = snippets.tf_print2(delta_cached_D,
        #                                     [y_st_D.values, A_D.values, A_bar_D.values,
        #                                      delta_cached_D, delta_cached_uptodate_D],
        #                                     'while_true_outside inside body_D : ')

        return [true, y_st_D, A_D, A_bar_D, delta_cached_D, delta_cached_uptodate_D, cov_vv_D]

    N = delta_cached_A.shape[0]
    y_st_ = tf.SparseTensor(indices=[[0, 0]], values=[tf.cast(0, dtype=tf.int64)], dense_shape=(N, 1))

    # delta_cached_A = snippets.tf_print2(delta_cached_A,
    #             [A_A.values, A_bar_A.values,
    #              delta_cached_A, delta_cached_uptodate_A],
    #             'while_true_outside before body_D : ')

    [loop_D, y_st_D_, A_D_, A_bar_D_, delta_cached_D_, delta_cached_uptodate_D_, _] = tf.while_loop(
            cond_D, body_D,
            loop_vars=[True, y_st_, A_A, A_bar_A, delta_cached_A, delta_cached_uptodate_A, cov_vv_A],
            name='whD')

    return y_st_D_, A_D_, A_bar_D_, delta_cached_D_, delta_cached_uptodate_D_


# V.b
def while_true_inside(true, A_D, A_bar_D, V, delta_cached_D, delta_cached_uptodate_D, cov_vv_D):

    N = delta_cached_D.shape[0]

    y_st_ = alg2.sparse_argmax_cache_linear(delta_cached_D, A_D, V)
    y_st_D = tf.SparseTensor(indices=[[y_st_, 0]], values=[tf.cast(y_st_, dtype=tf.int64)], dense_shape=(N, 1))

    # VI.
    bool_uptodate = if_delta_cached_is_uptodate(delta_cached_uptodate_D, y_st_D)
    true_ = tf.logical_not(bool_uptodate)

    # true_ = snippets.tf_print2(true_,
    #                 [true, y_st_D.values, A_D.values, A_bar_D.values,
    #                  delta_cached_D, delta_cached_uptodate_D],
    #                 'while_true_inside: ')

    # VII.  then_update_delta_y
    delta_cached_D_, delta_cached_uptodate_D_ = if_delta_cached_is_not_uptodate(
                                    y_st_D, A_D, A_bar_D, cov_vv_D,
                                    delta_cached_D,
                                    delta_cached_uptodate_D,
                                    go=true_)

    # delta_cached_D_ = snippets.tf_print2(delta_cached_D_,
    #                 [delta_cached_D_, delta_cached_uptodate_D_],
    #                 'while_true_inside, delta_cached_D_: ')

    return true_, y_st_D, A_D, A_bar_D, delta_cached_D_, delta_cached_uptodate_D_


def delta_cached_uptodate_empty(N):
    delta_cached_uptodate = tf.Variable(lambda: tf.fill([N, 1], False), shape=[N, 1])
    return delta_cached_uptodate

#===============================================================
# Placement algorithm 2
#===============================================================
def sparse_placement_algorithm_2(cov_vv, k, COVER_spatial):
    # V.
    '''
    assume U is empty
    assume V = S
    A - selection of placements eventually lngth = k
    y = len(V) + len(S) -> MOVE selections from V to S
    '''
    # I.
    # A = [] # selected_indexes
    with tf.name_scope("tf-placement-2"):
        INF = tf.constant(1e8, dtype=tf.float64)
        N = cov_vv.shape[0]
        assert_op = tf.Assert(tf.equal(N, COVER_spatial[0]*COVER_spatial[1]*COVER_spatial[2]), [N])  # 625
        with tf.control_dependencies([assert_op]):
            A = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))
        # V = np.linspace(0, cov_vv.shape[0]-1, cov_vv.shape[0], dtype=np.int64)
        values01 = tf.cast(tf.linspace(0., tf.cast(cov_vv.shape[0]-1, tf.float32), cov_vv.shape[0]), tf.int64)  # values
        values02 = tf.cast(tf.linspace(0., 0., cov_vv.shape[0]), tf.int64)  # values
        indxes = tf.stack([values01, values02], axis=1)
        V = tf.SparseTensor(indices=indxes, values=values01, dense_shape=[N, 1])

        # calc all delta_cached_iters
        delta_cached_iters = tf.Variable(tf.zeros([N, k], dtype=tf.float64), dtype=tf.float64, shape=[N, k])

        # II.
        A_bar = V  # complementer set to A.

        # delta_cached = [[INF, INF, ..]]
        delta_cached = tf.Variable(tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)),
                                   dtype=tf.float64,
                                   shape=[N, 1])
        A_selection_and_delta = tf.Variable(tf.zeros([k, 2], dtype=tf.float64),
                                   dtype=tf.float64,
                                   shape=[k, 2])
        # ===========================================================================
        # while len(A) < k:
        cond_A = lambda len_A, A_A, A_bar_A, delta_cached_A, cov_vv_A, dci, A_sel_n_delta: tf.less(len_A, k)  # dci is delta_cached_iters
        def body_A(len_A, A_A, A_bar_A, delta_cached_A, cov_vv_A, dci, A_sel_n_delta):

            # IV.  set_all_delta_cach_to_false
            # in each round we make decision on updated value
            # delta_cached_is_uptodate = [[False, False, ..]]
            delta_cached_uptodate_A = delta_cached_uptodate_empty(N)

            # delta_cached_uptodate = tf.Variable(tf.fill([N, 1], cache_vals), shape=[N, 1])
            # op2 = tf.scatter_nd_update(delta_cached_uptodate, [[y_st.values[0], 0]],
            #                            [cache_vals])  # deltdelta_cached_uptodate[y_st] = True
            # op = tf.s
            # delta_cached_uptodate_A = tf.Variable(tf.fill([N, 1], False), shape=[N, 1])

            # V.
            # while True:
            y_st_A, A_A, A_bar_A, delta_cached_A, delta_cached_uptodate_A \
                = while_true_outside(
                    A_A, A_bar_A, V,
                    delta_cached_A,
                    delta_cached_uptodate_A,
                    cov_vv_A)

            # pr_yst_op = tf.print(['y_st_A=', y_st_A.values[0], 'A_A=', A_A.values, A_A.indices,
            #                       '\ndelta_cached_A: \n', delta_cached_A])
            # with tf.control_dependencies([pr_yst_op]):
            #     pr_yst_op2 = tf.print(['before: ', 'A_A=', A_A.values, 'A_bar_A=', A_bar_A.values])

            # X.
            A_A, A_bar_A = append_to_A_remove_from_A_bar(A_A, A_bar_A, y_st_A)

            #     pr_yst_op3 = tf.print(['after: ', 'A_A=', A_A.values, 'A_bar_A=', A_bar_A.values,
            #                            # '\ndci: \n', dci,
            #                            '\ndelta_cached_iters: \n', delta_cached_iters])
            # with tf.control_dependencies([pr_yst_op2, pr_yst_op3]):

            len_A = tf.size(A_A.values)

            # t_i_ = tracers_loc_i - tf.fill(dims=tracers_loc_i.shape, value=tr_mean)
            indx_stack = tf.cast(tf.expand_dims(tf.linspace(0., tf.cast(N-1, dtype=tf.float32), N), axis=1), dtype=tf.int32)
            len_A_fill = tf.fill(dims=indx_stack.shape, value=len_A-1)
            indx_stack_ = tf.reshape(tf.stack([indx_stack, len_A_fill], axis=1), [-1, 2])

            # -------------------------------------------------
            # 1.
            A_sel_and_delta_var = tf.Variable(lambda: tf.zeros([k, 2], dtype=tf.float64),
                                              dtype=tf.float64, shape=[k, 2])
            # 2.
            A_sel_and_delta_before = tf.assign(A_sel_and_delta_var, A_sel_n_delta)

            # 3.
            A_sel_and_delta_after_ = tf.scatter_nd_update(A_sel_and_delta_before, [[len_A-1, 0]], [tf.cast(y_st_A.values[0], dtype=tf.float64)])  # update y_st
            A_sel_and_delta_after = tf.scatter_nd_update(A_sel_and_delta_after_, [[len_A-1, 1]], delta_cached_A[y_st_A.values[0]])  # update delta

            # -------------------------------------------------
            # 1. create a new Var and fill with its previous value, then add the current update
            dci_var = tf.Variable(lambda: tf.zeros([N, k], dtype=tf.float64), dtype=tf.float64, shape=[N, k])

            # 2. ... fill with its previous value, then add the current update
            dci_set_before = tf.assign(dci_var, dci)

            # 3. ... add the current update
            dci_set_after = tf.scatter_nd_update(dci_set_before, indx_stack_, tf.reshape(delta_cached_A, [-1]))

            pr_op = tf.print(['y_st=', y_st_A.values[0], 'delta=', delta_cached_A[y_st_A.values[0], 0]])

            # -------------------------------------------------
            # 1. delta_cached update interleaved with A_ordered_set update
            delta_cached_A_cpy = tf.Variable(lambda: tf.zeros([N, 1], dtype=tf.float64),
                                             dtype=tf.float64,
                                             shape=[N, 1])
            # 2.
            delta_cached_A_z = tf.assign(delta_cached_A_cpy, delta_cached_A)

            # 3.
            with tf.control_dependencies([dci_set_after
                                             # , pr_op
                                          ]):
                # we have saved delta_cached_A, but the value for y_st_A will never change again
                # We may zero it now - after delta_cache is separate from A selections.
                op_z = tf.scatter_nd_update(delta_cached_A_z, [[y_st_A.values[0], 0]], [0.])  # delta_cached[y_st] = 0
                pr_op_z = tf.print(['y_st=', y_st_A.values[0], 'delta_z=', delta_cached_A_z[y_st_A.values[0], 0]])

            with tf.control_dependencies([op_z, pr_op_z]):
                len_A_ret = len_A


            return [len_A_ret, A_A, A_bar_A, op_z, cov_vv_A, dci_set_after, A_sel_and_delta_after]


        # delta_cached = snippets.tf_print2(delta_cached, [tf.size(A.values), A.values, A_bar.values, delta_cached], 'starting with: ')
        # delta_cached_iters -> delta_cached_iters
        # A_selection_and_delta -> A_selection_and_delta_
        [loop_A, A, A_bar, delta_cached, _, delta_cached_iters_, A_selection_and_delta_] = tf.while_loop(
                cond_A, body_A,
                loop_vars=[tf.size(A.values), A, A_bar, delta_cached, cov_vv, delta_cached_iters, A_selection_and_delta],
                name='whA')

        # A_A is modelling a SET, thus working like a SET: does not care about the order
        #  => We loose the order of the selection in A_A

        # pr_yst_op = tf.print(['A=', A.values, A.indices, '\ndelta_cached_iters=', delta_cached_iters])

        with tf.control_dependencies([loop_A, A.values, delta_cached, delta_cached_iters_, A_selection_and_delta_]):

            # delta_cached_iters_tensor = tf.SparseTensor(indices=?, values=delta_cached_iters, dense_shape=[N, 1])
            return A, loop_A, delta_cached_iters_, A_selection_and_delta_  # A shall contain k elements.


# TEST ALL ========================================================
def test__algorithm_2(cov_vv):

    # I., II.
    k = 5
    INF = tf.constant(1e1000)
    N = cov_vv.shape[0]
    A = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))
    A_bar = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))
    values01 = tf.cast(tf.linspace(0., tf.cast(cov_vv.shape[0] - 1, tf.float32), cov_vv.shape[0]), tf.int64)  # values
    values02 = tf.cast(tf.linspace(0., 0., cov_vv.shape[0]), tf.int64)  # values
    indxes = tf.stack([values01, values02], axis=1)
    V = tf.SparseTensor(indices=indxes, values=values01, dense_shape=[N, 1])
    delta_cached = tf.Variable(tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)),
                               dtype=tf.float64)  #, dense_shape=[N, 1])
    delta_cached_uptodate = tf.Variable(tf.fill([N, 1], True)) #, dense_shape=[N, 1])  # needs to be a bool value

    # III.  while len(A) < k:
    cond_III = lambda delta_cached_, delta_cached_uptodate_, A_, A_bar_, V_ : tf.less(A.shape[0], k)
    def body_III(delta_cached_, delta_cached_uptodate_, A_, A_bar_, V_ ):
        y_st = tf.SparseTensor(indices=[[1, 0]], values=[-1], dense_shape=[N, 1])
        delta_st = tf.Variable([-1], name="delta_st")

        # IV.
        delta_cached_setfalse_, delta_cached_uptodate_ = set_all_delta_cache_to_false(delta_cached_uptodate)

        #V.  while(True):
        cond_V = lambda delta_cached__, delta_cached_uptodate__, A__, A_bar__, V__, y_st__ : tf.less(A.shape[0], k)
        def body_V(delta_cached__, delta_cached_uptodate__, A__, A_bar__, V__, y_st__):

            # VI.
            y_st = tf.SparseTensor(indices=[[1, 0]],values=tf.reshape(sparse_argmax_cache_linear(delta_cached, A, V), [1]), dense_shape=[N, 1]) # TODO if delta_cached has anything?

            # with tf.control_dependencies([y_st]):
            bool_uptodate = if_delta_cached_is_uptodate(delta_cached_uptodate, y_st)

            # VII. incl VIII, IX.
            delta_cached__, delta_cached_uptodate__ = if_delta_cached_is_not_uptodate(y_st, A__, A_bar__,
                                                                                    cov_vv,
                                                                                    delta_cached__,
                                                                                    delta_cached_uptodate__,
                                                                                    go=False)

            return delta_cached__, delta_cached_uptodate__, A__, A_bar__, V__, y_st  # return V. -> TODO while true logic
        [op_V] = tf.while_loop(cond_V, body_V, loop_vars=[delta_cached_, delta_cached_uptodate_, A_, A_bar_, V_, y_st])

        with tf.control_dependencies([op_V]):
        # X.
            append_to_A_remove_from_A_bar(A, A_bar, y_st)

        return delta_cached_, delta_cached_uptodate_, A_, A_bar_, V_ # return III.
    [op_III] = tf.while_loop(cond_III, body_III, loop_vars=[delta_cached, delta_cached_uptodate, A, A_bar, V])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(' A: ', sess.run(A.values))

    pass


def placement_algorithm_2_by_cov_vv_size_inner(cov_vv_np):

    cov_vv = tf.constant(cov_vv_np)
    A, len_A, _, _ = sparse_placement_algorithm_2(cov_vv, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [A_values, len_A_] = sess.run([A.values, len_A])

    return A_values, len_A_


def placement_algorithm_2_test_by_cov_vv_size(begin, end, step):

    print(' ')
    dim = begin
    dims = []
    times = []
    while dim < end:

        U = np.random.normal(0, 3, size=(dim, dim))
        UT_U = np.dot(U.T, U)
        cov_vv_np = UT_U + np.eye(dim)     # Sigma_pos_definite + diagonal(1)

        start_time = datetime.datetime.now()
        A_values, len_A = placement_algorithm_2_by_cov_vv_size_inner(cov_vv_np)
        end_time = datetime.datetime.now()

        diff_time = end_time - start_time
        diff_time_str = str(diff_time)
        print('placement: ', A_values, len_A, dim, diff_time_str)

        times.append(diff_time.total_seconds())
        dims.append(dim)

        dim += step
        df = pd.DataFrame(data={'dims': dims, 'times': times})
        df.to_csv('algo2_times_vs_dims.csv')

    pass


def snippets_a2_tests(file_name='./main_datasets_/cov_vv_small.csv'):

    # placement_algorithm_2_test_by_cov_vv_size(11, 10000*11+1, 11)
    # return

    # dim = 11
    # U = np.random.normal(1, 1, size=(dim, dim))
    # U_UT = np.dot(U, U.T)
    # cov_vv_np = U_UT  + 0.001*np.eye(dim)  # Sigma_pos_definite + diagonal(1)
    # cov_vv = tf.constant(cov_vv_np)

    #
    # When running main, I save a 'real' cov_vv as cov_vv.csv.
    # Saved one of them manually as cov_vv_saved.csv for this test
    #
    if 0 < len(file_name):
        cov_vv_np = snippets_save.load_cov_vv(file_name)[:40, :40]
    else:
        cov_vv_np = alg2.cov_vv_4x4()

    cov_vv = tf.constant(cov_vv_np)


    # print('Algorithm2')
    # test__algorithm_2()

    if False:
        print(' ')
        print('tf_nominator_test:')
        snippets.tf_nominator_test(cov_vv)

        A, len_A, _, _ = sparse_placement_algorithm_2(cov_vv, 5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(' ')
            [A_values_, A_len_] = sess.run([A.values, len_A])
            print('placement: ', A_values_, A_len_)

        A_np = alg2.placement_algorithm_2(cov_vv_np, 5)
        print('placement np: ', A_np, len(A_np))
        assert np.allclose(A_np, A_values_)

        print(' ')
        print('V.')
        test__y_st_argmax_cache_linear(cov_vv, 2, 3.)
        test__y_st_argmax_cache_linear(cov_vv, 10, 3.)
        test__y_st_argmax_cache_linear(cov_vv, 0, 3.)

        test__y_st_argmax_cache_linear(cov_vv, 3, 3.)
        test__y_st_argmax_cache_linear(cov_vv, 4, 3.)
        test__y_st_argmax_cache_linear(cov_vv, 5, 3.)
        test__y_st_argmax_cache_linear(cov_vv, 6, 3.)

        print(' ')
        print('VI.')
        test__if_delta_cached_is_uptodate(cov_vv, True)
        test__if_delta_cached_is_uptodate(cov_vv, False)

    print(' ')
    print('VII.')
    test__if_delta_cached_is_not_uptodate(cov_vv, updated=False, go=True)  # expect action
    test__if_delta_cached_is_not_uptodate(cov_vv, updated=True, go=True)
    test__if_delta_cached_is_not_uptodate(cov_vv, updated=False, go=False)
    test__if_delta_cached_is_not_uptodate(cov_vv, updated=True, go=False)

    print(' ')
    print('XI.')
    test__append_to_A_remove_from_A_bar(cov_vv)

    print(' ')
    print('IV.')
    test__set_all_delta_cache_to_false(cov_vv)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('I./II.')
        print('E.')
        print('if_denom_is_near_zero_check: 0., 0,: ', sess.run([if_denom_is_near_zero_check(0., 0.)]))
        print('if_denom_is_near_zero_check: 1., 0,: ', sess.run([if_denom_is_near_zero_check(1., 0.)]))
        print('if_denom_is_near_zero_check: 0., 1.,: ', sess.run([if_denom_is_near_zero_check(0., 1.)]))
        print('if_denom_is_near_zero_check: 1., 1.,: ', sess.run([if_denom_is_near_zero_check(1., 1.)]))

    print(' ')
    print('VIII.')
    test__if_denom_is_near_zero(nom=0., denom=0.)  # expect True
    test__if_denom_is_near_zero(nom=1., denom=0.)  # expect True
    test__if_denom_is_near_zero(nom=0., denom=1.)  # expect True
    test__if_denom_is_near_zero(nom=1., denom=1e-3)  # expect False
    test__if_denom_is_near_zero(nom=1., denom=1.e-35)  # expect False
    test__if_denom_is_near_zero(nom=1., denom=1.e-40)  # expect True !!
    test__if_denom_is_near_zero(nom=1., denom=1e-101)  # expect True

    print(' ')
    print('IX.')
    print('F. then update delta_y')
    test__then_update_delta_y(dnz=False, go=True)  # expect 100.
    test__then_update_delta_y(dnz=False, go=False)  # 0.
    test__then_update_delta_y(dnz=True, go=True)   # 0.
    test__then_update_delta_y(dnz=True, go=False)  # 0.

    print(' ')
    print('XX.')
    loop, A_bar = while_param_propagation_test(cov_vv)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('while_param_propagation_test loop: ', sess.run(loop))
        print('while_param_propagation_test A_bar: ', sess.run(A_bar.values))

    pass


def algo_2_timing(file_name='./main_datasets_/cov_vv_small.csv'):

    times = []
    for i in range(1, 1000):

        # increase N^2 by a 1000 each steps
        N = np.int32(np.sqrt(i*10000))
        print('N =', N)

        cov_vv_np = snippets_save.load_cov_vv(file_name)[:N, :N]
        cov_vv = tf.constant(cov_vv_np)
        A, len_A, delta_cached_iters, _, _ = sparse_placement_algorithm_2(cov_vv, 8, COVER_spatial)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(' ')
            start_time = datetime.datetime.now()

            [A_values_, A_len_, delta_cached_iters_] = sess.run([A.values, len_A, delta_cached_iters])

            end_time = datetime.datetime.now()
            diff_time = end_time - start_time
            diff_time_str = str(diff_time)
            print('placement: ', A_values_, A_len_, diff_time_str)
            times.append([N, diff_time.total_seconds()])
            pd.DataFrame(times).to_csv('main_datasets_/placement_times.csv')

            # print('delta_cached_iters: \n', delta_cached_iters_)
            pd.DataFrame(delta_cached_iters_).to_csv('main_datasets_/placement_algorithm_cache.csv')

            if False:
                print('\n\nnp:')
                A_np = alg2.placement_algorithm_2(cov_vv_np, 8)
                print('placement np: ', A_np, len(A_np))
                # assert np.allclose(A_np, A_values_)

    pass


def run_sparse_algo_2(flag_use_4x4=False):

    COVER_spatial, file_cov_vv, file_cov_idxs, file_delt_ch_it, _, \
        GEN_LOCAL_kernel, BETA_val = a2t.get_spatial_splits()

    if flag_use_4x4:
        cov_vv_np = alg2.cov_vv_4x4()
    else:
        # COVER_spatial = [10, 10, 1]
        N = COVER_spatial[0]*COVER_spatial[1]*COVER_spatial[2]
        cov_vv_np = snippets_save.load_cov_vv(file_cov_vv)[:N, :N]

    cov_vv = tf.constant(cov_vv_np)

    A, len_A, delta_cached_iters = sparse_placement_algorithm_2(cov_vv, 8, COVER_spatial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(' ')
        [A_values_, A_len_, delta_cached_iters_] = sess.run([A.values, len_A, delta_cached_iters])
        print('placement: ', A_values_, A_len_)
        # print('delta_cached_iters: \n', delta_cached_iters_)
        pd.DataFrame(delta_cached_iters_).to_csv(file_delt_ch_it)

        if False:
            print('\n\nnp:')
            A_np, _, _, _ = alg2.placement_algorithm_2(cov_vv_np, 8)
            print('placement np: ', A_np, len(A_np))
            # assert np.allclose(A_np, A_values_)

    pass


if __name__ == '__main__':
    # snippets_a2_tests('')
    run_sparse_algo_2()
    # algo_2_timing()

    pass


