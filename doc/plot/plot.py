from pathlib import Path
import typing as tp
import timeit
import sys
import os
import platform
import math


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


def get_versions(number) -> str:
    return f'Plots of duration (lower is faster) / OS: {platform.system()} / NumPy: {np.__version__} / Iterations: {number}\n'

def plot_performance(frame, *,
            number: int,
            fp: Path,
            title: str,
            log_scale: bool = False,
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

            x_labels = [f'{i}: {cls.NAME}' for i, cls in
                    zip(range(1, len(results) + 1),
                    fixture['cls_processor'].values)
                    ]
            x_tick_labels = [str(l + 1) for l in range(len(x_labels))]
            x = np.arange(len(results))
            x_bar = ax.bar(x_labels, results, color=color)


            density, position = fixture_label.split('-')
            # cat_label is the size of the array
            plot_title = f'{cat_label:.0e}\n{cls_fixture.DENSITY_TO_DISPLAY[density]}\n{cls_fixture.POSITION_TO_DISPLAY[position]}'

            ax.set_title(plot_title, fontsize=6)
            ax.set_box_aspect(0.75) # makes taller tan wide


            time_max = fixture["time"].max()
            time_min = fixture["time"].min()

            if log_scale:
                ax.set_yscale('log')
                y_ticks = []
                for v in range(
                        math.floor(math.log(time_min, 10)),
                        math.floor(math.log(time_max, 10)) + 1,
                        ):
                    y_ticks.append(1 * pow(10, v))
                ax.set_yticks(y_ticks)
            else:
                y_ticks = [0, time_min, time_max * 0.5, time_max]
                y_labels = [
                    "",
                    seconds_to_display(time_min, number),
                    seconds_to_display(time_max * 0.5, number),
                    seconds_to_display(time_max, number),
                ]
                if time_min > time_max * 0.25:
                    # remove the min if it is greater than quarter
                    y_ticks.pop(1)
                    y_labels.pop(1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)

            ax.tick_params(
                axis="y",
                length=2,
                width=0.5,
                pad=1,
                labelsize=4,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(x_tick_labels)
            ax.tick_params(
                axis="x",
                length=2,
                width=0.5,
                pad=1,
                labelsize=4,
            )

    fig.set_size_inches(6, 3.5) # width, height

    fig.legend(x_bar, x_labels, loc='center right', fontsize=4)
    # fig.legend(post, names_display, loc='center right', fontsize=8)

    # horizontal, vertical
    fig.text(.05, .96, title, fontsize=10)
    fig.text(.05, .90, get_versions(number), fontsize=6)

    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.86,
            top=0.8,
            wspace=0, # width
            hspace=1.0,
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
    if 'Log' in title:
        plot_performance(f, number=number, fp=fp, title=title, log_scale=True)
    else:
        plot_performance(f, number=number, fp=fp, title=title, log_scale=False)


