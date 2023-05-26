import typing as tp
import setuptools
import site
import os

NPB_VERSION = "0.1.0"

def get_ext_dir(*components: tp.Iterable[str]) -> tp.Sequence[str]:
    dirs = []
    for sp in site.getsitepackages():
        fp = os.path.join(sp, *components)
        if os.path.exists(fp):
            dirs.append(fp)
    return dirs


extension = setuptools.Extension(
    "np_bench",
    ["np_bench.c"],
    include_dirs=get_ext_dir("numpy", "core", "include"),
    library_dirs=get_ext_dir("numpy", "core", "lib"),
    define_macros=[("NPB_VERSION", NPB_VERSION)],
    libraries=["npymath"],
)

setuptools.setup(
    name="np_bench",
    version=NPB_VERSION,
    description="NumPy Performance Benchmarks",
    python_requires=">=3.7.0",
    install_requires=["numpy>=1.18.5"],
    url="https://github.com/flexatone/np-bench",
    author="Christopher Ariza",
    license="MIT",
    ext_modules=[extension],
)
