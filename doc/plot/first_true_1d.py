import os
import sys
import typing as tp
from pathlib import Path
import numpy as np

from np_bench import first_true_1d_getitem
from np_bench import first_true_1d_scalar
from np_bench import first_true_1d_npyiter
from np_bench import first_true_1d_getptr
from np_bench import first_true_1d_ptr
from np_bench import first_true_1d_ptr_unroll

sys.path.append(os.getcwd())

from plot import ArrayProcessor
from plot import Fixture

#-------------------------------------------------------------------------------

class AKFirstTrueGetitem(ArrayProcessor):
    NAME = 'first_true_1d_getitem()'
    SORT = 0

    def __call__(self):
        _ = first_true_1d_getitem(self.array, True)

class AKFirstTrueScalar(ArrayProcessor):
    NAME = 'first_true_1d_scalar()'
    SORT = 1

    def __call__(self):
        _ = first_true_1d_scalar(self.array, True)

class AKFirstTrueNpyiter(ArrayProcessor):
    NAME = 'first_true_1d_npyiter()'
    SORT = 2

    def __call__(self):
        _ = first_true_1d_npyiter(self.array, True)

class AKFirstTrueGetptr(ArrayProcessor):
    NAME = 'first_true_1d_getptr()'
    SORT = 3

    def __call__(self):
        _ = first_true_1d_getptr(self.array, True)

class AKFirstTruePtr(ArrayProcessor):
    NAME = 'first_true_1d_ptr()'
    SORT = 4

    def __call__(self):
        _ = first_true_1d_ptr(self.array, True)

class AKFirstTruePtrUnroll(ArrayProcessor):
    NAME = 'first_true_1d_ptr_unroll()'
    SORT = 5

    def __call__(self):
        _ = first_true_1d_ptr_unroll(self.array, True)

#-------------------------------------------------------------------------------

class PYLoop(ArrayProcessor):
    NAME = 'Python Loop'
    SORT = -1

    def __call__(self):
        for i, e in enumerate(self.array):
            if e == True:
                break


class NPNonZero(ArrayProcessor):
    NAME = 'np.nonzero()'
    SORT = 30

    def __call__(self):
        _ = np.nonzero(self.array)[0][0]

class NPArgMax(ArrayProcessor):
    NAME = 'np.argmax()'
    SORT = 10

    def __call__(self):
        _ = np.argmax(self.array)

class NPNotAnyArgMax(ArrayProcessor):
    NAME = 'np.any(), np.argmax()'
    SORT = 20

    def __call__(self):
        _ = not np.any(self.array)
        _ = np.argmax(self.array)


#-------------------------------------------------------------------------------

class FixtureFactory(Fixture):
    NAME = ''

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return np.full(size, False, dtype=bool)

    def _get_array_filled(
            size: int,
            start_third: int, # 1 or 2
            density: float, # less than 1
            ) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        count = size * density
        start = int(len(a) * (start_third/3))
        length = len(a) - start
        step = int(length / count)
        fill = np.arange(start, len(a), step)
        a[fill] = True
        return a

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = cls.get_array(size)
        return cls.NAME, array

    DENSITY_TO_DISPLAY = {
        'single': '1 True',
        'tenth': '10% True',
        'third': '33% True',
    }

    POSITION_TO_DISPLAY = {
        'first_third': 'Fill 1/3 to End',
        'second_third': 'Fill 2/3 to End',
    }


class FFSingleFirstThird(FixtureFactory):
    NAME = 'single-first_third'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        a[int(len(a) * (1/3))] = True
        return a

class FFSingleSecondThird(FixtureFactory):
    NAME = 'single-second_third'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        a[int(len(a) * (2/3))] = True
        return a


class FFTenthPostFirstThird(FixtureFactory):
    NAME = 'tenth-first_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=1, density=.1)


class FFTenthPostSecondThird(FixtureFactory):
    NAME = 'tenth-second_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=2, density=.1)


class FFThirdPostFirstThird(FixtureFactory):
    NAME = 'third-first_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=1, density=1/3)


class FFThirdPostSecondThird(FixtureFactory):
    NAME = 'third-second_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=2, density=1/3)


CLS_PROCESSOR = (
    AKFirstTrueGetitem,
    AKFirstTrueScalar,
    AKFirstTrueNpyiter,
    AKFirstTrueGetptr,
    AKFirstTruePtr,
    AKFirstTruePtrUnroll,

    NPNonZero,
    NPNotAnyArgMax,
    # PYLoop,
    )

CLS_FF = (
    FFSingleFirstThird,
    FFSingleSecondThird,
    FFTenthPostFirstThird,
    FFTenthPostSecondThird,
    FFThirdPostFirstThird,
    FFThirdPostSecondThird,
)


if __name__ == '__main__':
    from plot import run_test
    run_test(sizes=(100_000, 1_000_000, 10_000_000),
            fixtures=CLS_FF,
            processors=CLS_PROCESSOR,
            fp=Path('/tmp/first_true_1d.png'),
            title='first_true_1d()',
            number=2,
            )


