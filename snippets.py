import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import numpy as np
import collections
import placement_algorithm2 as alg2
import tensorflow_probability as tfp
import sys

def numpy_is_eq(x, y):
    return np.array(x == y).reshape([]).astype(np.int32)


def tf_function_y_in_A(y, A):
    match_array = tf.map_fn(lambda x: tf.cast(tf.equal(x, y), tf.int32), A)
    return tf.reduce_any(tf.cast(match_array, tf.bool))


def tf_function_y_in_A2(y, A):
    return tf.foldl(lambda a, x: tf.cast(tf.logical_or(tf.equal(a, 1), tf.equal(x, y)), tf.int32),
                     A,
                     initializer=tf.constant(0))


def numpy_foldl(a, x, y):
    print('a, x, y: ', a, x, y)
    return np.array(x == y or a == 1).astype(np.int32).reshape([])


def tf_if_y_in_A():

    A = tf.constant([1, 2, 3, 4, 5, 6, 7])
    y = tf.constant(3)

    # match_array = tf.map_fn(lambda x: tf.cast(tf.equal(x, y), tf.int32), A)
    # # match_array = tf.map_fn(lambda x: tf.numpy_function(numpy_is_eq, [x, y], Tout=tf.int32), A)
    # y_in_A = tf.reduce_any(tf.cast(match_array, tf.bool))

    bool_val = tf.function(tf_function_y_in_A)(y, A)

    # foldl_np = tf.foldl(lambda a, x: tf.reshape(tf.numpy_function(numpy_foldl, [a, x, y], Tout=tf.int32), []),
    #                     A,
    #                     initializer=tf.constant(0))
    #
    # foldl = tf.foldl(lambda a, x: tf.cast(tf.logical_or(tf.equal(a, 1), tf.equal(x, y)), tf.int32),
    #                  A,
    #                  initializer=tf.constant(0))

    bool_val2 = tf.function(tf_function_y_in_A2)(y, A)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('A: ', sess.run(A))
        print('y: ', sess.run(y))
        # print('match_array: ', sess.run(match_array))
        # print('y_in_A: ', sess.run(y_in_A))
        print('bool_val: ', sess.run(bool_val))

        # print('foldl_np: ', sess.run(foldl_np))
        # print('foldl: ', sess.run(foldl))
        print('bool_val2: ', sess.run(bool_val2))

        pass


def play_with_sets():
    sp1 = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [3, 0]], values=[7, 8, 9, 10], dense_shape=[24, 1])
    sp1_ = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [3, 0]], values=[6, 12, 9, 10], dense_shape=[24, 1])
    sp2 = tf.SparseTensor(indices=[[20, 0], [21, 0], [22, 0], [23, 0]], values=[7, 302, 303, 10], dense_shape=[24, 1])
    values01 = tf.cast(tf.linspace(0., 10., 11), tf.int64) # values
    values02 = tf.cast(tf.linspace(0., 0., 11), tf.int64)  # values
    indxes = tf.stack([values01, values02], axis=1)

    sparse1 = tf.SparseTensor(indxes, values01, [values02.shape[0], 1])

    assert(sp1.shape == sp2.shape)  # tf.sets op REQ
    sp_12 = tf.sets.union(sp1, sp2)
    sp_11_ = tf.sets.intersection(sp1, sp1_)
    sp_10_ = tf.sets.difference(sp1, sp1_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('sp1.indices: ', sess.run(sp1.indices)[:, 0].flatten())
        print('sp1.values: ', sess.run(sp1.values))

        print('sp2.indices: ', sess.run(sp2.indices)[:, 0].flatten())
        print('sp2.values: ', sess.run(sp2.values))

        print('sp_12.indices: ', sess.run(sp_12.indices)[:, 0].flatten())
        print('sp_12.values: ', sess.run(sp_12.values))

        print('sp_11_.indices: ', sess.run(sp_11_.indices)[:, 0].flatten())
        print('sp_11_.values: ', sess.run(sp_11_.values))

        print('sp_10_.indices: ', sess.run(sp_10_.indices)[:, 0].flatten())
        print('sp_10_.values: ', sess.run(sp_10_.values))

        print('sparse1_.indices: ', sess.run(sparse1.indices)[:, 0].flatten())
        print('sparse1_.values: ', sess.run(sparse1.values))

    pass


##########################################################
# GRAPH PRINT ############################################
def tf_print2(op, tensors, message=None):

    def print_message2(*args):
        str_ = message
        for s_ in [str(x) for x in args]:
            str_ += s_ + ', '
        sys.stdout.write(str_ + '\n')
        return args

    prints = tf.numpy_function(print_message2, [t for t in tensors], [t.dtype for t in tensors])
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op


def tf_placement_algorithm2(cov_vv, k):
    INF = tf.constant(1e1000)
    N = cov_vv.shape[0]
    A = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))
    A_bar = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))

    values01 = tf.cast(tf.linspace(0., tf.cast(cov_vv.shape[0]-1, tf.float32), cov_vv.shape[0]), tf.int64) # values
    values02 = tf.cast(tf.linspace(0., 0., cov_vv.shape[0]), tf.int64)  # values
    indxes = tf.stack([values01, values02], axis=1)
    V = tf.SparseTensor(indxes, values01, [cov_vv.shape[0], 1])

    delta_cached = tf.Variable(tf.multiply(INF, tf.ones([N, 1])), shape=[N, 1])
    delta_cached_uptodate = tf.Variable(tf.zeros([N, 1]), shape=[N, 1])

    zeros = tf.constant(np.zeros([N, 1]))
    indexes = np.arange(N.value)

    # def loop_V(i, A_bar):
    #     i0 = tf.SparseTensor(indices=[[i, 0]], values=[i], dense_shape=(N, 1))
    #     u = tf.sets.union(A_bar, i0)
    #     return u, tf.add(1, i)

    # A_bar2 = tf.function(loop_V)(10, A_bar)
    # v = tf.map_fn(lambda i: tf.function(loop_V)(i, A_bar), V)  # for i in V
    # ===========================================================================
    cond_v = lambda i_, N_, _: tf.less(i_, N_)
    def body_v(i_, N_, A_bar_):

        i0 = tf.SparseTensor(indices=[[i_, 0]], values=[tf.cast(i_, dtype=tf.int64)], dense_shape=(N, 1))
        A_bar_ = tf.sets.union(A_bar_, i0)
        # v = A_bar_.values
        # v = tf_print2(v, [i_, v], 'v: ')
        # with tf.control_dependencies([v]):
        with tf.control_dependencies([A_bar_.values]):
            i_ret = tf.add(1, i_)

        return i_ret, N_, A_bar_
    # lvd = {'A_bar': A_bar}
    loop, _, A_bar = tf.while_loop(cond_v, body_v,loop_vars=[0, N, A_bar])

    loop = tf_print2(loop, [A_bar.values], 'v: ')

    A_dict = {'A': A, 'A_bar': A_bar,
              'delta_cached_uptodate': delta_cached_uptodate,
              'zeros': zeros,
              'indexes': indexes}

    # updt0 = tf.reshape(indexes, [N, 1])
    # op_cache_zero0 = tf.assign(delta_cached_uptodate,
    dc_uptodate = tf.slice(A_dict["delta_cached_uptodate"], y_st.indices[0], [1, 1])
    true_if_fn = lambda : False  # when uptodate, break
    else_fn = lambda : True  # otherwise keep calc, else:
    true = tf.case([(tf.reshape(tf.equal(dc_uptodate, [1]), []), true_if_fn)], default=else_fn)

    # else ------------------------------------------------------
    #                      tf.cast(tf.zeros_like(updt0), tf.float32))

    # ===========================================================================
    cond_A = lambda i_, A_A, A_bar_A, delta_cached_uptodate_A: tf.less(i_, k)  # while len(A) < k: #
    def body_A(i_, A_A, A_bar_A, delta_cached_uptodate_A):

        # while must keep the structure
        y_st = tf.SparseTensor(indices=[[-1, 0]], values=[tf.cast(-1, dtype=tf.int64)], dense_shape=(N, 1))

        # Nf = tf.cast(N, tf.float32)
        updt_A = tf.reshape(indexes, [N, 1])
        op_cache_zero_A = tf.assign(delta_cached_uptodate, tf.cast(tf.zeros_like(updt_A), tf.float32))

        # ===========================================================================
        cond_D = lambda true, _, A_D, A_bar_D, delta_cached_uptodate_D: true  # while True:
        def body_D(true, y_st_D, A_D, A_bar_D, delta_cached_uptodate_D):

            y_st_ = tf.function(alg2.sparse_argmax_cache_linear)(delta_cached, A, V)
            y_st_D = tf.SparseTensor(indices=[[y_st_, 0]], values=[tf.cast(y_st_, dtype=tf.int64)], dense_shape=(N, 1))
            # TODO: re-implement with priority-queue

            # if --------------------------------------------------------
            dc_uptodate = tf.slice(delta_cached_uptodate_D, y_st_D.indices[0], [1, 1])
            true_if_fn = lambda : False  # when uptodate, break
            else_fn = lambda : True  # otherwise keep calc, else:
            true = tf.case([(tf.reshape(tf.equal(dc_uptodate, [1]), []), true_if_fn)], default=else_fn)

            # else ------------------------------------------------------
            # ===========================================================================
            cond_E = lambda dc_, true_, _: tf.reshape(tf.logical_and(true_, tf.not_equal(dc_, [[1.]])), [])  # not up to date so we want to exec else
            def body_E(dc_, true_, y_st_E):  # else branch, returns True if it was not up to date

                delta_y = tf.Variable(0., dtype=tf.float64, name="delta_y")  # delta_y = 0
                nom = alg2.tf_nominator(y_st_E, A_D, cov_vv)    # nom = nominator(y_st, A, cov_vv)
                denom = alg2.tf_nominator(y_st_E, A_bar_D, cov_vv)   # denom = denominator(y_st, A_bar, cov_vv)

                # if --------------------------------------------------------
                small = tf.constant(1e-8, dtype=tf.float64)
                true_if_fn_ = lambda : False
                else_fn_ = lambda : True
                check_denom = tf.case([(tf.reshape(
                                            tf.logical_or(
                                                tf.less(tf.abs(denom), small),   # if np.abs(denom) < 1e-8 or nom == 0:
                                                tf.less(tf.abs(nom), small)), []), true_if_fn_)], default=else_fn_)

                # else ------------------------------------------------------
                # ===========================================================================
                cond_F = lambda check_denom_, true__, _, nom_, denom_: tf.logical_not(true__) # TODO or logical or  # is this enough?
                def body_F(check_denom_, true__, _, nom_, denom_):

                    delta_y_F = tf.divide(tf.reshape(nom_, []), tf.reshape(denom_, []))
                    delta_y_F = tf_print2(delta_y_F, [delta_y_F], 'delta_y_F: ')

                    return [False, true__, delta_y_F, nom_, denom_]  # is this ok ?

                [cd_F, true_, delta_y, _, _] = tf.while_loop(
                    cond_F, body_F,
                    loop_vars=[check_denom, true_, delta_y, nom, denom],
                    name='whF')

                delta_y = tf_print2(delta_y, [delta_y], 'delta_y (out): ')

                # Should op1 and op2 be in body_F ?
                op1 = tf.scatter_nd_update(delta_cached, [y_st_E.values[0]], [delta_y])  # delta_cached[y_st] = delta_y
                op2 = tf.scatter_nd_update(delta_cached_uptodate, [y_st_E.values[0]], [tf.cast([1.], tf.float32)])  # delta_cached_is_uptodate[y_st] = True

                with tf.control_dependencies([y_st_E.values, op1, op2]):
                    return tf.constant([[0.]]), true_, y_st_E  # because we only execute it once

            # l_dict = {"true": true}
            [loop_E_uptod, true, y_st_D] = tf.while_loop(
                cond_E, body_E, loop_vars=[dc_uptodate, true, y_st_D],
                name='whE')
            with tf.control_dependencies([y_st_D.values]):
                return [true, y_st_D, A_D, A_bar_D, delta_cached_uptodate_D]

        with tf.control_dependencies([op_cache_zero_A]):
            [loop_D, y_st_Do, A_A, A_bar_A, delta_cached_uptodate_A] = tf.while_loop(
                cond_D, body_D,
                loop_vars=[True, y_st, A_A, A_bar_A, delta_cached_uptodate_A],
                name='whD')

        # y_st_Do = tf_print2(y_st_Do, [y_st_Do.values, A_A.values], 'loop_A-s y_st, A_: ')
        # y_st_ = tf.SparseTensor(indices=[[y_st_Do, 0]], values=[tf.cast(y_st, dtype=tf.int64)], dense_shape=A_A.shape)
        A_A = tf.sets.union(A_A, y_st_Do)
        A_bar_A = tf.sets.difference(A_bar_A, y_st_Do)

        with tf.control_dependencies([op_cache_zero_A]):  # , y_st.values
            s_ret_ = tf.size(A_A)
            s_ret = tf_print2(s_ret_, [s_ret_], 'loop_A ctrl var: ')

        return [s_ret, A_A, A_bar_A, delta_cached_uptodate_A]

    with tf.control_dependencies([loop]):
        [loop_A, A, A_bar, delta_cached_uptodate] = tf.while_loop(
            cond_A, body_A, loop_vars=[tf.size(A), A, A_bar, delta_cached_uptodate],
            name='whA')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # loop_ = sess.run(loop)
        # print('loop_: ', loop_)

        # a_bar = sess.run(A_bar)
        # print('a_bar.values', a_bar.values)

        [loop_Ao, Ao, A_baro] = sess.run([loop_A, A, A_bar])
        print('loop_Ao: ', loop_Ao)
        print('Ao: ', Ao)
        print('A_baro: ', A_baro)

    return A.values


def tf_nominator_test(cov_vv):
    N = cov_vv.shape[0]
    values01 = tf.cast(tf.linspace(0., tf.cast(cov_vv.shape[0] - 1, tf.float32), cov_vv.shape[0]), tf.int64)  # values
    values02 = tf.cast(tf.linspace(0., 0., cov_vv.shape[0]), tf.int64)  # values
    indxes = tf.cast(tf.stack([values01, values02], axis=1), tf.int64)
    V = tf.SparseTensor(indices=indxes, values=values01, dense_shape=[N, 1])

    y_st = tf.SparseTensor(indices=[[4, 0]], values=tf.cast([4], tf.int64), dense_shape=(N, 1))
    # A = tf.SparseTensor(indices=[[5, 0]], values=tf.cast(, tf.int64)[5], dense_shape=(N, 1))
    Ain = tf.SparseTensor(indices=[[7, 0], [9, 0]], values=tf.cast([7, 9], tf.int64), dense_shape=(N, 1))

    y = y_st

    # A = Ain
    A = tf.sets.union(Ain, y)
    A_bar = tf.sets.difference(V, A)

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
    mul2 = tf.tensordot(mul1, cov_Ay, [[0], [0]])
    nom0 = sigm_yy - mul2

    nom = alg2.tf_nominator(y_st, A, cov_vv)  # nom = nominator(y_st, A, cov_vv)
    denom = alg2.tf_denominator(y_st, A_bar, cov_vv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cov_vv_np = sess.run(cov_vv)

        print('y_st: ', sess.run([y_st.indices, y_st.values]))
        y_st_np = sess.run(y_st.values[0])
        print('y_st_np: ', y_st_np)

        print('Ain: ', sess.run([Ain.indices, Ain.values]))
        A_np = sess.run(A.values)
        A_bar_np = sess.run(A_bar.values)
        print('A: ', sess.run([A.indices, A.values]))
        print('A_np: ', A_np)

        sigma_yy_np = alg2.make_slice(cov_vv_np, [y_st_np], [y_st_np])
        sigm_yy_tf = sess.run(sigm_yy)
        print('sigma_yy_np: ', sigma_yy_np)
        print('sigm_yy_tf: ', sigm_yy_tf)
        assert np.allclose(sigma_yy_np, sigm_yy_tf, atol=1.e-15)

        # print('A_idxs_: ', sess.run(A_idxs_))
        # print('A_idxs_d: ', sess.run(A_idxs_d))

        # print('A_idxs: ', sess.run(A_idxs))
        print('cov_yA_row: ', sess.run(cov_yA_row))
        cov_yA_tf = sess.run(cov_yA)
        print('cov_yA_tf: ', cov_yA_tf)
        cov_yA_np = alg2.make_slice(cov_vv_np, [y_st_np], A_np)
        print('cov_yA_np: ', cov_yA_np)
        assert np.allclose(cov_yA_np, cov_yA_tf, atol=1.e-15)

        # print('cov_AA_rows: ', sess.run(cov_AA_rows))
        cov_AA_tf = sess.run(cov_AA)
        print('cov_AA: ', cov_AA_tf)
        cov_AA_np = alg2.make_slice(cov_vv_np, A_np, A_np)
        assert np.allclose(cov_AA_np, cov_AA_tf, atol=1.e-15)

        # print('cov_Ay_col: ', sess.run(cov_Ay_col))
        cov_Ay_tf = sess.run(cov_Ay)
        print('cov_Ay_tf: ', cov_Ay_tf)
        cov_Ay_np = alg2.make_slice(cov_vv_np, A_np, [y_st_np])
        print('cov_Ay_np.T: ', cov_Ay_np.T)

        # TODO: .T ?
        assert np.allclose(cov_Ay_np.T, cov_Ay_tf, atol=1.e-15)

        pinv_tf = sess.run(pinv)
        pinv_np = np.linalg.pinv(cov_AA_np)
        assert np.allclose(pinv_np, pinv_tf, atol=1.e-15)

        mul1_tf = sess.run(mul1)
        mul1_np = np.dot(cov_yA_np, pinv_np)
        assert np.allclose(mul1_np, mul1_tf, atol=1.e-8)

        mul2_np = np.dot(mul1_np, cov_Ay_np)
        mul2_tf = sess.run(mul2)
        assert np.allclose(mul2_np, mul2_tf, atol=1.e-15)

        print(' ')
        nom0_tf = sess.run(nom0)
        print('nom0_tf: ', nom0_tf)
        nom0_np = sigma_yy_np - mul2_np
        assert np.allclose(nom0_np, nom0_tf, atol=1.e-8)

        nom_tf = sess.run(nom)
        nom_np = alg2.nominator(y_st_np, A_np, cov_vv_np)
        print('nom_tf: ', nom_tf)
        print('nom_np: ', nom_np)
        assert np.allclose(nom_np, nom_tf, atol=1.e-8)

        denom_tf = sess.run(denom)
        denom_np = alg2.denominator(y_st_np, A_bar_np, cov_vv_np)
        print('denom_tf: ', denom_tf)
        print('denom_np: ', denom_np)
        assert np.allclose(denom_np, denom_tf, atol=1.e-5)

        pass

    return nom
    pass


if __name__ == '__main__':
    cov_vv = tf.constant(np.random.normal(0, 3, size=(11, 11)))

    # nom = tf_nominator_test(cov_vv)
    sel_idx = tf_placement_algorithm2(cov_vv, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('sel_idx: ', sess.run(sel_idx))

    # play_with_sets()
    # tf_if_y_in_A()
    pass
