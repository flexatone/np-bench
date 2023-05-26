import functools
import sys

import invoke

run = functools.partial(invoke.Context.run, echo=True, pty=True)

@invoke.task()
def clean(context):
    run(
        context,
        f"{sys.executable} -m pip --disable-pip-version-check uninstall --yes np_bench",
    )

    for artifact in ("*.egg-info", "*.so", "build", "dist"):
        run(context, f"rm -rf {artifact}")

@invoke.task(clean)
def build(context):
    run(context, f"{sys.executable} -m pip --disable-pip-version-check -v install .")

@invoke.task(build)
def test(context):
    run(context, f"{sys.executable} -m pytest -s")
