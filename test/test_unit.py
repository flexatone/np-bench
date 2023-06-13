import numpy as np
import np_bench as npb

def test_version():
    assert isinstance(npb.__version__, str)

def test_first_true_1d_a():
    for func in (
            npb.first_true_1d_npyiter,
            npb.first_true_1d_getptr,
            npb.first_true_1d_ptr,
            npb.first_true_1d_ptr_unroll,
            ):
        assert (
            func(np.array([False, False, True, False]), True) == 2)
        assert (
            func( np.array([False, True, True, False]), True) == 1)
        assert (
            func(np.array([False, False, False, False]), True) == -1)


def test_first_true_1d_b():
    for func in (
            # npb.first_true_1d_npyiter, # cannot do reverse
            npb.first_true_1d_getptr,
            npb.first_true_1d_ptr,
            npb.first_true_1d_ptr_unroll,
            ):
        assert (
            func( np.array([False, False, True, False]), False) == 2)
        assert (
            func( np.array([False, True, True, True]), False) == 3)
        assert (
            func(np.array([False, False, False, False]), False) == -1)


def test_first_true_1d_npyiter():

    assert (
        npb.first_true_1d_npyiter(np.array([False, False, True, False]), True) == 2
        )

    # import ipdb; ipdb.set_trace()

