import numpy as np
import np_bench as npb

def test_version():
    assert isinstance(npb.__version__, str)

def test_first_true_1d_a():
    for func in (
            npb.first_true_1d_getitem,
            npb.first_true_1d_scalar,
            npb.first_true_1d_npyiter,
            npb.first_true_1d_getptr,
            npb.first_true_1d_ptr,
            npb.first_true_1d_ptr_unroll,
            npb.first_true_1d_memcmp,
            npb.first_true_1d_uintcmp,
            ):
        assert (
            func(np.array([False, False, True, False]), True) == 2)
        assert (
            func( np.array([False, True, True, False]), True) == 1)
        assert (
            func(np.array([False, False, False, False]), True) == -1)


def test_first_true_1d_b():
    for func in (
            npb.first_true_1d_getitem,
            npb.first_true_1d_scalar,
            # npb.first_true_1d_npyiter, # cannot do reverse
            npb.first_true_1d_getptr,
            npb.first_true_1d_ptr,
            npb.first_true_1d_ptr_unroll,
            npb.first_true_1d_memcmp,
            npb.first_true_1d_uintcmp,
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


def test_first_true_2d_a():
    a1 = array = np.arange(24).reshape(4,6) % 5 == 0
    for func in (
            npb.first_true_2d_unroll,
            npb.first_true_2d_memcmp,
            ):
        post0 = func(array, axis=0)
        assert post0.tolist() == [0, -1, 3, 2, 1, 0]
        post1 = func(array, axis=1)
        assert post1.tolist() == [0, 4, 3, 2]


