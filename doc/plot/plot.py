from pathlib import Path
import typing as tp
import timeit
import sys
import os
import platform

import numpy as np
import static_frame as sf
import matplotlib.pyplot as plt

import np_bench as npb

class ArrayProcessor:
    NAME = ''
    SORT = -1

    def __init__(self, array: np.ndarray):
        self.array = array

    def __call__(self):
        '''Implement performance routine to be timed'''
        raise NotImplementedError()


def seconds_to_display(seconds: float, number: int) -> str:
    seconds /= number
    if seconds < 1e-4:
        return f'{seconds * 1e6: .1f} (µs)'
    if seconds < 1e-1:
        return f'{seconds * 1e3: .1f} (ms)'
    return f'{seconds: .1f} (s)'


def get_versions() -> str:
    return f'OS: {platform.system()} / np_bench: {npb.__version__} / NumPy: {np.__version__}\n'

def plot_performance(frame, *,
            number: int,
            fp: Path,
            title: str,
            ):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['size'].unique())
    processor_total = len(frame['cls_processor'].unique())
    fig, axes = plt.subplots(cat_total, fixture_total)

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap('plasma')

    color = cmap(np.arange(processor_total) / processor_total)

    # category is the size of the array
    for cat_count, (cat_label, cat) in enumerate(frame.iter_group_items('size')):
        for fixture_count, (fixture_label, fixture) in enumerate(
                cat.iter_group_items('fixture')):
            ax = axes[cat_count][fixture_count]
            # set order
            fixture = fixture.to_frame_go()
            cls_fixture = fixture['cls_fixture'].values[0]
            fixture['sort'] = [f.SORT for f in fixture['cls_processor'].values]
            fixture = fixture.sort_values('sort')

            results = fixture['time'].values.tolist()

            # names = [cls.NAME for cls in fixture['cls_processor'].values]
            # # x = np.arange(len(results))
            # names_display = names
            # post = ax.bar(names_display, results, color=color)

            x_labels = [f'{i}: {cls.NAME}' for i, cls in
                    zip(range(1, len(results) + 1), fixture['cls_processor'])
                    ]
            x_tick_labels = [str(l + 1) for l in range(len(x_labels))]
            x = np.arange(len(results))
            x_bar = ax.bar(x_labels, results, color=color)


            density, position = fixture_label.split('-')
            # cat_label is the size of the array
            title = f'{cat_label:.0e}\n{cls_fixture.DENSITY_TO_DISPLAY[density]}\n{cls_fixture.POSITION_TO_DISPLAY[position]}'

            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.75) # makes taller tan wide

            time_max = fixture["time"].max()
            time_min = fixture["time"].min()
            y_ticks = [0, time_min, time_max * 0.5, time_max]
            y_labels = [
                "",
                seconds_to_display(time_min),
                seconds_to_display(time_max * 0.5),
                seconds_to_display(time_max),
            ]
            if time_min > time_max * 0.25:
                # remove the min if it is greater than quarter
                y_ticks.pop(1)
                y_labels.pop(1)

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=4)
            ax.tick_params(
                axis="y",
                length=2,
                width=0.5,
                pad=1,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(x_tick_labels, fontsize=4)
            ax.tick_params(
                axis="x",
                length=2,
                width=0.5,
                pad=1,
            )

    fig.set_size_inches(9, 3.5) # width, height

    fig.legend(x_bar, x_labels, loc='center right', fontsize=6)
    # fig.legend(post, names_display, loc='center right', fontsize=8)

    # horizontal, vertical
    fig.text(.05, .96, f'{title} Performance: {number} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)

    plt.subplots_adjust(
            left=0.075,
            bottom=0.05,
            right=0.80,
            top=0.85,
            wspace=1, # width
            hspace=0.1,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')


class Fixture:

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        raise NotImplementedError()



def run_test(*,
        sizes: tp.Iterable[int],
        fixtures: tp.Iterable[Fixture],
        processors: tp.Iterable[tp.Any],
        number: int,
        fp: Path,
        title: str,
        ):
    records = []
    for size in sizes:
        for cls_fixture in fixtures:
            fixture_label, fixture = cls_fixture.get_label_array(size)
            for cls_processor in processors:
                runner = cls_processor(fixture)

                record = [cls_processor, cls_fixture, number, fixture_label, size]
                print(record)
                try:
                    result = timeit.timeit(
                            f'runner()',
                            globals=locals(),
                            number=number)
                except OSError:
                    result = np.nan
                finally:
                    pass
                record.append(result)
                records.append(record)

    f = sf.Frame.from_records(records,
            columns=('cls_processor', 'cls_fixture', 'number', 'fixture', 'size', 'time')
            )
    print(f)
    plot_performance(f, number=number, fp=fp, title=title)