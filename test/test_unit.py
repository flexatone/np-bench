import numpy as np
import np_bench as npb

def test_version():
    assert isinstance(npb.__version__, str)

def test_first_true_1d_a():
    assert (
        npb.first_true_1d(np.array([False, False, True, False])) ==
        2)

    assert (
        npb.first_true_1d(
                np.array([False, False, True, False]),
                forward=False) ==
        2)


def test_first_true_1d_b():
    assert (
        npb.first_true_1d(np.array([False, True, True, False])) ==
        1)

    assert (
        npb.first_true_1d(
                np.array([False, True, True, True]),
                forward=False) ==
        3)


    # import ipdb; ipdb.set_trace()

