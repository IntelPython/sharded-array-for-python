import os
import subprocess
import sys

import pytest

env_ranks = os.getenv("MPI_LOCALNRANKS")
if env_ranks is not None:
    pytest.skip(
        "subprocess tests cannot be run with mpiexec", allow_module_level=True
    )


@pytest.fixture()
def sharpy_script(tmp_path):
    content = """import sharpy as sp
import os
device = os.getenv("SHARPY_DEVICE", "")
sp.init(False)
a = a = sp.ones((4,), device=device)
assert a.size == 4
print("SUCCESS")
sp.fini()"""
    p = tmp_path / "sharpy_script.py"
    p.write_text(content)
    return p


def run_script(sharpy_script, env=None):
    if env is None:
        env = {}
    cmd = [sys.executable, str(sharpy_script)]
    fullenv = os.environ.copy()
    for k, v in env.items():
        fullenv[k] = str(v)
    cp = subprocess.run(
        cmd, env=fullenv, capture_output=True, check=True, text=True
    )
    assert cp.stdout.strip() == "SUCCESS"


@pytest.mark.parametrize("value", [0, 1, 4])
def test_verbose(sharpy_script, value):
    run_script(sharpy_script, {"SHARPY_VERBOSE": value})


@pytest.mark.parametrize("value", ["foo", "invalid_value"])
def test_verbose_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_VERBOSE": value})


@pytest.mark.parametrize(
    "value", [0, 1, "Y", "y", "on", "True", "true", "false"]
)
def test_cache(sharpy_script, value):
    run_script(sharpy_script, {"SHARPY_USE_CACHE": value})


@pytest.mark.parametrize("value", ["dog", -1])
def test_cache_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_USE_CACHE": value})


@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_opt(sharpy_script, value):
    run_script(sharpy_script, {"SHARPY_OPT_LEVEL": value})


@pytest.mark.parametrize("value", ["dog", 4])
def test_opt_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_OPT_LEVEL": value})


@pytest.mark.parametrize(
    "value",
    [
        "",
        "host",
        pytest.param(
            "opencl:gpu", marks=pytest.mark.xfail(reason="GPU is broken")
        ),
    ],
)
def test_device(sharpy_script, value):
    run_script(sharpy_script, {"SHARPY_DEVICE": value})


@pytest.mark.parametrize("value", ["dog", "opencl::gpu", "gpu\u0007"])
def test_device_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_DEVICE": value})


@pytest.mark.parametrize("value", [0, 1])
def test_forcedist(sharpy_script, value):
    run_script(sharpy_script, {"SHARPY_FORCE_DIST": value})


@pytest.mark.parametrize("value", ["dog"])
def test_forcedist_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_FORCE_DIST": value})


@pytest.mark.xfail(reason="effective in GPU mode only")
@pytest.mark.parametrize("value", ["foo\u0007", "dog"])
def test_gpulib_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_GPUX_SO": value})


@pytest.mark.parametrize("value", [1])
def test_skipcomm(sharpy_script, value):
    run_script(sharpy_script, {"SHARPY_SKIP_COMM": value})


@pytest.mark.parametrize("value", ["dog"])
def test_skipcomm_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_SKIP_COMM": value})


@pytest.mark.parametrize("value", [1])
def test_noasync(sharpy_script, value):
    run_script(sharpy_script, {"SHARPY_NO_ASYNC": value})


@pytest.mark.parametrize("value", ["dog"])
def test_noasync_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_NO_ASYNC": value})


@pytest.mark.parametrize("value", ["foo\u0007"])
def test_passes_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_PASSES": value})


@pytest.mark.parametrize("value", ["foo\u0007"])
def test_mlirroot_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"MLIRROOT": value})


@pytest.mark.xfail(reason="effective in GPU mode only")
@pytest.mark.parametrize("value", ["foo\u0007"])
def test_imexroot_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"IMEXROOT": value})


@pytest.mark.parametrize("value", ["foo bar", "foo"])
def test_fallback_invalid(sharpy_script, value):
    with pytest.raises(subprocess.CalledProcessError):
        run_script(sharpy_script, {"SHARPY_FALLBACK": value})
