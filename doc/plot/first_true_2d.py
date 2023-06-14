import os
import sys
import typing as tp
from pathlib import Path

from np_bench import first_true_2d
import np_bench as npb
import numpy as np

sys.path.append(os.getcwd())

from plot import ArrayProcessor
from plot import Fixture

#-------------------------------------------------------------------------------
class AKFirstTrueAxis0Forward(ArrayProcessor):
    NAME = 'ak.first_true_2d, axis=0'
    SORT = 0

    def __call__(self):
        _ = first_true_2d(self.array, forward=True, axis=0)

class AKFirstTrueAxis1Forward(ArrayProcessor):
    NAME = 'ak.first_true_2d, axis=1'
    SORT = 0

    def __call__(self):
        _ = first_true_2d(self.array, forward=True, axis=1)

class AKFirstTrueAxis0Reverse(ArrayProcessor):
    NAME = 'ak.first_true_2d, axis=0'
    SORT = 1

    def __call__(self):
        _ = first_true_2d(self.array, forward=False, axis=0)

class AKFirstTrueAxis1Reverse(ArrayProcessor):
    NAME = 'ak.first_true_2d, axis=1'
    SORT = 1

    def __call__(self):
        _ = first_true_2d(self.array, forward=False, axis=1)


class NPNonZero(ArrayProcessor):
    NAME = 'np.nonzero()'
    SORT = 3

    def __call__(self):
        x, y = np.nonzero(self.array)
        # list(zip(x, y)) # simulate iteration


class NPArgMaxAxis0(ArrayProcessor):
    NAME = 'np.any, np.argmax, axis=0'
    SORT = 4

    def __call__(self):
        _ = ~np.any(self.array, axis=0)
        _ = np.argmax(self.array, axis=0)

class NPArgMaxAxis1(ArrayProcessor):
    NAME = 'np.any, np.argmax, axis=1'
    SORT = 4

    def __call__(self):
        _ = ~np.any(self.array, axis=1)
        _ = np.argmax(self.array, axis=1)


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
        # TODO: find a better way to do this
        a = a.reshape(size // 10, 10)
        return a

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        # import ipdb; ipdb.set_trace()
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
        a = a.reshape(size // 10, 10)
        return a

class FFSingleSecondThird(FixtureFactory):
    NAME = 'single-second_third'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        a[int(len(a) * (2/3))] = True
        a = a.reshape(size // 10, 10)
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


# def get_versions() -> str:
#     import platform
#     return f'OS: {platform.system()} / np_bench: {npb.__version__} / NumPy: {np.__version__}\n'


CLS_PROCESSOR = (
    AKFirstTrueAxis0Forward,
    AKFirstTrueAxis1Forward,
    # AKFirstTrueAxis0Reverse,
    # AKFirstTrueAxis1Reverse,
    NPNonZero,
    NPArgMaxAxis0,
    NPArgMaxAxis1
    )

CLS_FF = (
    FFSingleFirstThird,
    FFSingleSecondThird,
    # FFTenthPostFirstThird,
    # FFTenthPostSecondThird,
    FFThirdPostFirstThird,
    FFThirdPostSecondThird,
)


if __name__ == '__main__':
    from plot import run_test

    directory = Path('doc/bnpy-scipy-2023/public')

    run_test(sizes=(100_000, 1_000_000, 10_000_000),
            fixtures=CLS_FF,
            processors=CLS_PROCESSOR,
            fp=directory / 'ft2d-fig-0.png',
            title='first_true_2d() Performance',
            number=2,
            )



