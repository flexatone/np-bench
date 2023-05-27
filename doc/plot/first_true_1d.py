import os
import sys
import typing as tp
from pathlib import Path

from np_bench import first_true_1d
import numpy as np

sys.path.append(os.getcwd())

from plot import ArrayProcessor
from plot import Fixture

#-------------------------------------------------------------------------------
class AKFirstTrue(ArrayProcessor):
    NAME = 'first_true_1d()'
    SORT = 0

    def __call__(self):
        _ = first_true_1d(self.array, forward=True)

class PYLoop(ArrayProcessor):
    NAME = 'Python Loop'
    SORT = 0

    def __call__(self):
        for i, e in enumerate(self.array):
            if e == True:
                break


class NPNonZero(ArrayProcessor):
    NAME = 'np.nonzero()'
    SORT = 3

    def __call__(self):
        _ = np.nonzero(self.array)[0][0]

class NPArgMax(ArrayProcessor):
    NAME = 'np.argmax()'
    SORT = 1

    def __call__(self):
        _ = np.argmax(self.array)

class NPNotAnyArgMax(ArrayProcessor):
    NAME = 'np.any(), np.argmax()'
    SORT = 2

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
    AKFirstTrue,
    NPNonZero,
    NPArgMax,
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


