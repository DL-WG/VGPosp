import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import numpy as np
import tensorflow_probability as tfp
import snippets_a2 as sa2
import placement_algorithm2 as pa2
import main_architecture_2_sampledistribution as ma2
import main_tests.alg2_split10x10x10_50x50x10_tests as a2t

import pandas as pd
import sys
# import pprint
# pp = pprint.PrettyPrinter(indent=4, width=160)

#===============================================================
# function
#===============================================================

#---------------------------------------------------------------
def fn():

    pass

# exponential_decay_fn = mapping between index distance and decay values.
#   large beta -> narrow decay -> small cutoff but lots have been cut off
# beta = scalar, relation to cutoff through decay function
# cutoff = distance in index space
# i_0, i_1, i_2 = index
# I0 = number of indices splitting a fixed real distance

#===============================================================
# Placement algorithm 3
#===============================================================
# inputs: cov_vv, k, split, cutoff (scaled to represent spatial distance, but algorithm 3 knows only index distance)
# outputs: delta_cached_iters(plot each delta_cache per k), A(selected_points)

# local kernels (cov_vv)
# new cov that models exponential decay in dependecy with distance.

def sparse_placement_algorithm_3(cov_vv, k, COVER_spatial, cutoff):

    [I0, I1, I2] = COVER_spatial
    ts0 = tf.timestamp()

    # I.
    INF = tf.constant(1e8, dtype=tf.float64)
    N = cov_vv.shape[0]
    assert_op = tf.Assert(tf.equal(N, COVER_spatial[0]*COVER_spatial[1]*COVER_spatial[2]), [N])
    with tf.control_dependencies([assert_op]):
        A = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=np.empty((0), dtype=np.int64), dense_shape=(N, 1))

    # II.2 create A_bar
    values01 = tf.cast(tf.linspace(0., tf.cast(cov_vv.shape[0] - 1, tf.float32), cov_vv.shape[0]), tf.int64)  # values
    values02 = tf.cast(tf.linspace(0., 0., cov_vv.shape[0]), tf.int64)  # values
    indxes = tf.stack([values01, values02], axis=1)
    V = tf.SparseTensor(indices=indxes, values=values01, dense_shape=[N, 1])
    A_bar = V  # complementer set to A.
    len_A = tf.size(A.values)

    # II. foreach y in S do: delta_y <- H(y) - Hhat_epsilon(y | V\y)
    # alg2 part 3. == alg3 part 1. the update of all delta_y only happens once here,
    #   taken out from within the loop j=1 to k in alg2
    delta_cached = tf.Variable(tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)), dtype=tf.float64, shape=[N, 1])

    #
    # the arrangement of body_A is such, that we come in with an up-to-date delta_cache,
    # at the beginning we do the selection for index 0
    # and at the end we save the updated delta_cache (into delta_cached_iters) at index 1,
    # because 0 is already up-to-date
    #
    delta_cached_iters = tf.Variable(tf.zeros([N, k], dtype=tf.float64), dtype=tf.float64, shape=[N, k])

    # Update for 1 k all the the delta_cache
    cond_D = lambda y_D, A_D, A_bar_D, delta_cached : tf.less(y_D, N)
    def body_D(y_D, A_D, A_bar_D, delta_cached_D):

        pr_iD = tf.print(['y_D =', y_D, tf.timestamp() - ts0])

        delta_cached_assign = tf.Variable(lambda : tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)), dtype=tf.float64, shape=[N, 1])
        dci_set_before = tf.assign(delta_cached_assign, delta_cached_D)

        tf_nominator_ = tf.function(sa2.tf_nominator)
        tf_denominator_ = tf.function(sa2.tf_denominator)

        y = tf.SparseTensor(indices=tf.cast([[y_D, 0]], dtype=np.int64),
                            values=tf.cast([y_D], dtype=np.int64),
                            dense_shape=[N, 1])

        nom = tf_nominator_(y, A_D, cov_vv)
        denom = tf_denominator_(y, A_bar_D, cov_vv)

        # nom_ = snippets.tf_print2(nom, [nom, denom], 'if_delta_cached_is_not_uptodate: nom, denom=\n')

        denom_is_near_zero = sa2.if_denom_is_near_zero(nom, denom)

        delta_y = sa2.then_update_delta_y(
            denom_is_near_zero,  # True : don't do anything
            True,  # False : don't do anything
            nom,
            denom)

        update = tf.reshape(delta_y, [], name="update")
        indexes = tf.cast([y_D, 0], dtype=tf.int32, name='indexes')
        dci_set_afterD = tf.scatter_nd_update(dci_set_before, [indexes], [update])

        pr_op0 = tf.print(['y_D=', y_D, 'indexes=', indexes, 'update=', update])
        with tf.control_dependencies([pr_iD, dci_set_afterD
                                      # , pr_op0
                                      ]):
            y_D += 1

        return y_D, A_D, A_bar_D, dci_set_afterD

    # loop_D is last value of y, when cond was still called, but while loop is finished
    [loop_D, _, _, delta_cached0] \
        = tf.while_loop(cond_D, body_D,
            loop_vars=[0, A, A_bar, delta_cached], name='whD')

    # now the delta cache is up to date (first pass)
    # save it to delta_cache_iters

    indx_stack = tf.cast(tf.expand_dims(tf.linspace(0., tf.cast(N - 1, dtype=tf.float32), N), axis=1), dtype=tf.int32)
    len_A_fill = tf.fill(dims=indx_stack.shape, value=0)  # which column to fill
    indx_stack_ = tf.reshape(tf.stack([indx_stack, len_A_fill], axis=1), [-1, 2])
    dci_set_after = tf.scatter_nd_update(delta_cached_iters, [indx_stack_], [tf.reshape(delta_cached0, [-1])])
    print_op = tf.print(['dci_set_after=', dci_set_after], output_stream=sys.stdout, name='print_dci')

    # now the first column of delta_cache_iters is up-to-date

    # return indx_stack_, print_op

    # III. for j=1 to k do ===========================================================================
    cond_A = lambda i, A_A, A_bar_A, delta_cached_A, dci: tf.less(i, k-1)  # while len(A) < k: #
    def body_A(i, A_A, A_bar_A, delta_cached_A, dci):

        # pr_op2 = tf.print(['A_A.values=', A_A.values, 'V.values=', V.values], output_stream=sys.stdout, name='print_dci')
        # with tf.control_dependencies([pr_op2]):

        # IV. y* <- argmax_y delta_y
        y_st_ = pa2.sparse_argmax_cache_linear(delta_cached_A, A_A, V)
        pr_sel = tf.print(['y_st=', y_st_, 'delta=', delta_cached_A[y_st_]])
        y_st_A = tf.SparseTensor(indices=[[y_st_, 0]], values=[tf.cast(y_st_, dtype=tf.int64)], dense_shape=(N, 1))

        # V. A <- A U y*
        intersect_size_A0 = tf.reduce_sum(tf.sets.size(tf.sets.intersection(A_A, y_st_A)))
        itaA0 = tf.Assert(tf.equal(0, intersect_size_A0), [intersect_size_A0, y_st_A.values, A_A.values])

        A_A, A_bar_A = sa2.append_to_A_remove_from_A_bar(A_A, A_bar_A, y_st_A)

        # check that y_st not in A
        intersect_size_A = tf.reduce_sum(tf.sets.size(tf.sets.intersection(A_A, y_st_A)))
        itaA = tf.Assert(tf.equal(1, intersect_size_A), [intersect_size_A, y_st_A.values, A_A.values])

        #
        # Consider that zeroing of the selected point (which we would not touch otherwise),
        # .. like in algo_2, for visualization. (The delta_cached_A is already in dci,
        # so this value will just not poison the next round) => delta_cached_A[y_st_] = 0
        #
        delta_cached_Av = tf.Variable(lambda: tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)),
                                      dtype=tf.float64,
                                      shape=[N, 1])  # create a new var
        delta_cached_Avs = tf.assign(delta_cached_Av, delta_cached_A)  # copy the value over
        update = tf.reshape(0., [], name="update")
        indexes = tf.cast([y_st_, 0], dtype=tf.int32, name='indexes')
        delta_cached_A_clear = tf.scatter_nd_update(delta_cached_Avs, [indexes], [update])  # update & carry the new val
        # check
        pr_cl = tf.print(['i=', i,
                          'delta_cached_A_clear[y_st_] != 0. :', y_st_,
                          delta_cached_A_clear[y_st_, 0]],
                                 output_stream=sys.stdout,
                                 name='print_dci')
        a_cl = tf.Assert(tf.equal(delta_cached_A_clear[y_st_, 0], 0.),
                                  [i,
                                   'delta_cached_A_clear[y_st_] != 0. :',
                                   y_st_,
                                   delta_cached_A_clear[y_st_, 0]])

        #
        # foreach y NEAR (y_st, cutoff) delta_cache[y] = ...
        # VI. foreach y E N(y*; epsilon) do:
        # update cache, only the delta_ys should be updated,
        # where the cov_vv > 0 <=> ones near y* in epsilon distance
        # -> THIS is why complexity of function is reduced from O(N*N3) to O(epsilon*N2)
        #

        # with tf.control_dependencies([pr_cl, a_cl]):
        strde0 = I2 * I1
        strde1 = I2
        strde2 = 1
        i0 = y_st_ // strde0
        i1 = (y_st_ - i0*strde0) // strde1
        i2 = (y_st_ - i0*strde0 - i1*strde1) // strde2
        a0 = tf.Assert(tf.less(i0, I0), ['i=', y_st_, 'i0, I0=', i0, I0, 'i1, I1=', i1, I1, 'i2, I2=', i2, I2])
        a1 = tf.Assert(tf.less(i1, I1), ['i=', y_st_, 'i0, I0=', i0, I0, 'i1, I1=', i1, I1, 'i2, I2=', i2, I2])
        a2 = tf.Assert(tf.less(i2, I2), ['i=', y_st_, 'i0, I0=', i0, I0, 'i1, I1=', i1, I1, 'i2, I2=', i2, I2])
        with tf.control_dependencies([a0, a1, a2,
                                      # pr_cl,
                                      pr_sel,
                                      a_cl, itaA0, itaA]):
            i0_ = i0

        cond0 = lambda j0, J0, A0, Abar0, y0, delta_cached0: tf.less(j0, J0)
        def body0(j0, J0, A0, Abar0, y0, delta_cached0):
            pr_i = tf.print(['j0 =', j0, tf.timestamp() - ts0])

            cond1 = lambda j1, J1, A1, Abar1, y1, delta_cached1: tf.less(j1, J1)
            def body1(j1, J1, A1, Abar1, y1, delta_cached1):

                cond2 = lambda j2, J2, A2, Abar2, y2, delta_cached2: tf.less(j2, J2)
                def body2(j2, J2, A2, Abar2, y2, delta_cached2):

                    #
                    # Here we have j0, j1, j2 -> yj
                    #   which y is to be updated in delta_cached_A;
                    #

                    yj = strde0*j0 + strde1*j1 + strde2*j2
                    yjA = tf.SparseTensor(indices=[[yj, 0]],
                                          values=[tf.cast(yj, dtype=tf.int64)],
                                          dense_shape=(N, 1))

                    delta_cached2v = tf.Variable(lambda: tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)),
                                                  dtype=tf.float64,
                                                  shape=[N, 1])  # create a new var
                    delta_cached2vs_ = tf.assign(delta_cached2v, delta_cached2)  # copy the value over

                    # repeat the clearing step (have done outside) TODO: no effect
                    update2 = tf.reshape(0., [], name="update")
                    indexes2 = tf.cast([yj, 0], dtype=tf.int32, name='indexes')
                    delta_cached2vs = tf.scatter_nd_update(delta_cached2vs_, [indexes2],
                                                           [update2])  # update & carry the new val

                    #
                    # switching delta_cached2vs to hold just the original and
                    #  update the delta_cached2 -> delta_cached2o_ works, but not the other way around:
                    # : it does not write the values in A over
                    #

                    A2m = tf.sets.difference(A2, yjA)
                    Abar2m = tf.sets.difference(Abar2, yjA)
                    _, _, _, delta_cached2o_ = body_D(yj, A2m, Abar2m, delta_cached2)

                    # avoid updating the cache (in the point around which we are updating: not enough)
                    #                          -> any of the points in A
                    intersect_size_y_st = tf.reduce_sum(tf.sets.size(tf.sets.intersection(A2, y2)))
                    ita = tf.Assert(tf.equal(1, intersect_size_y_st), [intersect_size_y_st, y2.values, A2.values])

                    intersect_size_yj = tf.reduce_sum(tf.sets.size(tf.sets.intersection(A2, yjA)))
                    delta_cached2o = tf.case([(tf.less(0, intersect_size_yj),       # yj in A2
                                                  lambda: delta_cached2vs)],        # avoid
                                                default=lambda: delta_cached2o_)    # update

                    pr_j2 = tf.print([#'j0, j1, j2, y_st, yj, its, dci, dct=',
                                      j0, j1, j2,
                                      y2.values[0], yj, intersect_size_yj,
                                      delta_cached2o[yj, 0],
                                      delta_cached2o_[yj, 0]])
                    with tf.control_dependencies([
                        delta_cached2vs,
                        pr_j2,  # TODO: if I print here, the cache update seems to work, otherwise NOT!
                        ita,
                        delta_cached2o]):
                        j2_ = j2+1
                    return j2_, J2, A2, Abar2, y2, delta_cached2o

                [loop2, _, _, _, _, delta_cached1o] = tf.while_loop(
                    cond2, body2,
                    loop_vars=[tf.reduce_max([i2 - cutoff, 0]),
                               tf.reduce_min([i2 + cutoff, I2]),
                               A1, Abar1, y1, delta_cached1],
                    name='wh1')

                # pr_j1 = tf.print(['pr1: j1, delta_cached1o[y_st_, 0]=', j1, delta_cached1o[y_st_, 0]])
                with tf.control_dependencies([loop2  #, pr_j1
                                              ]):
                    j1_ = j1+1
                return j1_, J1, A1, Abar1, y1, delta_cached1o

            [loop1, _, _, _, _, delta_cached0o] = tf.while_loop(
                cond1, body1,
                loop_vars=[tf.reduce_max([i1 - cutoff, 0]),
                           tf.reduce_min([i1 + cutoff, I1]),
                           A0, Abar0, y0, delta_cached0],
                name='wh1')

            # pr_j0 = tf.print(['pr0: j0, delta_cached0o[y_st_, 0]=', j0, delta_cached0o[y_st_, 0]])
            with tf.control_dependencies([pr_i, loop1  #, pr_j0
                                          ]):
                j0_ = j0+1
            return j0_, J0, A0, Abar0, y0, delta_cached0o

        with tf.control_dependencies([a0, a1, a2,
                                      # pr_cl,
                                      a_cl]):
            [loop0, _, _, _, _, delta_cached_Ao] = tf.while_loop(
                        cond0, body0,
                        loop_vars=[tf.reduce_max([i0_-cutoff, 0]),
                                   tf.reduce_min([i0_+cutoff, I0]),
                                   A_A, A_bar_A, y_st_A, delta_cached_A_clear],
                        name='wh0')

        #
        # At this point we have updated delta_cached_Ao, so we are about to select the next y_st
        # Before doing that, we should update delta cached iters (the next selection will be based on that)
        # .. and consider that zeroing of the selected point (which we would not touch otherwise),
        # .. like in algo_2, for visualization.
        #
        # indx_stackA = tf.cast(tf.expand_dims(tf.linspace(0., tf.cast(N - 1, dtype=tf.float32), N), axis=1),
        #                       dtype=tf.int32)
        with tf.control_dependencies([delta_cached_Ao]):

            # it seems it is hard to stop the incoming delta_cached to have a value at y_st_
            # TODO: investigate
            # try again ... delta_cached_Ao[y_st_] = 0
            delta_cached_iv = tf.Variable(lambda: tf.multiply(INF, tf.ones([N, 1], dtype=tf.float64)),
                                          dtype=tf.float64,
                                          shape=[N, 1])  # create a new var
            delta_cached_ivs = tf.assign(delta_cached_iv, delta_cached_Ao)  # copy the value over
            updatei = tf.reshape(0., [], name="update")
            indexesi = tf.cast([y_st_, 0], dtype=tf.int32, name='indexes')
            delta_cached_i_clear = tf.scatter_nd_update(delta_cached_ivs, [indexesi],
                                                        [updatei])  # update & carry the new val
            # ... delta_cached_Ao[y_st_] = 0

            len_A_fillA = tf.fill(dims=indx_stack.shape, value=i+1)  # which column to fill
            indx_stackA_ = tf.reshape(tf.stack([indx_stack, len_A_fillA], axis=1), [-1, 2])
            dci_set_afterA = tf.scatter_nd_update(dci,
                                                  [indx_stackA_],
                                                  [tf.reshape(delta_cached_i_clear, [-1])])

            # make sure we have 0 in dci_set_afterA[y_st_, i+1]
            print_opi = tf.print(['i=', i,
                                  'delta_cached_Ao[y_st_]=', y_st_,
                                  delta_cached_Ao[y_st_, tf.cast(0, tf.int64)]],
                                     output_stream=sys.stdout,
                                     name='print_dci')
            print_opo = tf.print(['i=', i,
                                  'dci_set_afterA[y_st_, i+1]=', y_st_,
                                  dci_set_afterA[y_st_, tf.cast(i+1, tf.int64)]],
                                     output_stream=sys.stdout,
                                     name='print_dci')

        with tf.control_dependencies([loop0,
                                      # print_opi, print_opo
                                      ]):
            i_ = i+1
        return i_, A_A, A_bar_A, delta_cached_Ao, dci_set_afterA

    [loop_A, A, A_bar, delta_cachedr, delta_cached_iters_] = tf.while_loop(
        cond_A, body_A,
        loop_vars=[0, A, A_bar, delta_cached0, dci_set_after],
        name='whA')

    #
    # here we have the last cache, up-to-date, but we have not selected y_st (from that last one) yet.
    #
    y_st_l = pa2.sparse_argmax_cache_linear(delta_cachedr, A, V)
    y_st_last = tf.SparseTensor(indices=[[y_st_l, 0]], values=[tf.cast(y_st_l, dtype=tf.int64)], dense_shape=(N, 1))
    A, A_bar = sa2.append_to_A_remove_from_A_bar(A, A_bar, y_st_last)

    return A, delta_cachedr, delta_cached_iters_


if __name__ == '__main__':
    k = 7

    cover_spatial, file_cov_vv, file_cov_idxs, file_delt_ch_it, SAVE_SELECTION_file, \
        GEN_LOCAL_kernel, BETA_val \
        = a2t.get_spatial_splits()
    cov_vv_ = pd.read_csv(file_cov_vv)
    cutoff = 3

    cov_vv = tf.constant(cov_vv_)
    A, dc, dci = sparse_placement_algorithm_3(cov_vv, k, cover_spatial, cutoff)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [A_, dc_, dci_] = sess.run([A.values, dc, dci])
        print(['A=', A_])
        # print('dc=')
        # print(dc_[:3, :])
        # print('dci=')
        # print(dci_[:3, :])

        # checking the visualization issue
        for r in range(k):
            print('i=', r, 'y_st=', A_[r], 'dci[y_st, :]=', dci_[A_[r], :])

        pd.DataFrame(dci_).to_csv(file_delt_ch_it)
        pd.DataFrame(A_).to_csv(SAVE_SELECTION_file)
        pass