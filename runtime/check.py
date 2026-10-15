"""Builds gpuphys.c with the translated kernels and runs runtime/test_gpuphys.c.

    python3 runtime/check.py [--crust DIR]
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpk  # noqa: E402


def main(argv):
    crust = argv[argv.index("--crust") + 1] if "--crust" in argv else None
    work = tempfile.mkdtemp(prefix="gpuphys-")
    gpk.main([work] + (["--crust", crust] if crust else []))
    srcs = [os.path.join(work, "gpk_%s.c" % n) for n in gpk.KERNEL_FILES]
    exe = os.path.join(work, "test_gpuphys")
    cc = os.environ.get("CC", "cc")
    r = subprocess.run([cc, "-O1", "-w", "-I", HERE, "-o", exe, os.path.join(HERE, "test_gpuphys.c"),
                        os.path.join(HERE, "gpuphys.c")] + srcs + ["-lm"])
    if r.returncode:
        return r.returncode
    return subprocess.run([exe]).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
