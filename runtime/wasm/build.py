#!/usr/bin/env python3
"""Builds gpuphys.wasm: the C runtime plus Crust's C translation of the kernels, compiled by clang for
wasm32 with no libc (runtime/wasm/libc.c supplies the little that is needed).

    python3 runtime/wasm/build.py [--crust DIR] [--out DIR] [--cc clang]

Writes gpuphys.wasm next to gpuphys.js (default: runtime/wasm/dist). Needs clang with the wasm32
target and wasm-ld; no emscripten, no wasi-sdk. gpuphys.js loads it in a browser or Node.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.dirname(HERE)
sys.path.insert(0, RUNTIME)
import gpk  # noqa: E402


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.exit("build.py: %s\n%s%s" % (" ".join(cmd[:3]), p.stdout, p.stderr))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--crust")
    ap.add_argument("--out", default=os.path.join(HERE, "dist"))
    ap.add_argument("--cc", default=os.environ.get("CLANG") or shutil.which("clang") or "clang")
    ap.add_argument("--opt", default="-O2")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="gpuphys-wasm-")
    gpk.main([work] + (["--crust", a.crust] if a.crust else []))
    res = subprocess.run([a.cc, "-print-resource-dir"], capture_output=True, text=True).stdout.strip()
    flags = ["--target=wasm32", a.opt, "-w", "-ffreestanding", "-fno-builtin-sinf", "-fno-builtin-cosf", "-fno-math-errno",
             "-nostdinc", "-isystem", os.path.join(HERE, "include"), "-isystem", os.path.join(res, "include"),
             "-I", RUNTIME, "-c"]
    objs = []
    srcs = [os.path.join(RUNTIME, "gpuphys.c"), os.path.join(HERE, "libc.c")] + \
           [os.path.join(work, "gpk_%s.c" % n) for n in gpk.KERNEL_FILES]
    for s in srcs:
        o = os.path.join(work, os.path.basename(s)[:-2] + ".o")
        extra = ["-fno-builtin"] if s.endswith("libc.c") else []
        run([a.cc] + flags + extra + ["-o", o, s])
        objs.append(o)
    out = os.path.join(a.out, "gpuphys.wasm")
    run([a.cc, "--target=wasm32", "-nostdlib", "-fuse-ld=lld", "-Wl,--no-entry", "-Wl,--export-all",
         "-Wl,--allow-undefined", "-Wl,--initial-memory=33554432", "-Wl,--max-memory=1073741824", "-Wl,--stack-first",
         "-Wl,-z,stack-size=1048576", "-Wl,--export=__heap_base", "-o", out] + objs)
    print("wrote %s (%d bytes)" % (out, os.path.getsize(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
