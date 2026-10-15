#!/usr/bin/env python3
"""Builds gpuphys.wasm, builds the same scenario natively, and runs tests/test.mjs in Node against both.

    python3 runtime/wasm/check.py [--crust DIR]
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.dirname(HERE)
sys.path.insert(0, RUNTIME)
import gpk  # noqa: E402


def main(argv):
    crust = ["--crust", argv[argv.index("--crust") + 1]] if "--crust" in argv else []
    r = subprocess.run([sys.executable, os.path.join(HERE, "build.py")] + crust)
    if r.returncode:
        return r.returncode
    work = tempfile.mkdtemp(prefix="gpuphys-native-")
    gpk.main([work] + crust)
    exe = os.path.join(work, "parity")
    srcs = [os.path.join(work, "gpk_%s.c" % n) for n in gpk.KERNEL_FILES]
    r = subprocess.run([os.environ.get("CC", "cc"), "-O1", "-w", "-I", RUNTIME, "-o", exe,
                        os.path.join(HERE, "tests", "parity.c"), os.path.join(RUNTIME, "gpuphys.c")] + srcs + ["-lm"])
    if r.returncode:
        return r.returncode
    out = os.path.join(work, "native.txt")
    with open(out, "w") as fh:
        subprocess.run([exe], stdout=fh, check=True)
    return subprocess.run(["node", os.path.join(HERE, "tests", "test.mjs"), out]).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
