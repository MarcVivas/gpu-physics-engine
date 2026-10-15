#!/usr/bin/env python3
"""SPIR-V to WGSL for the engine's kernels: what a browser's WebGPU runs.

WebGPU takes WGSL, not SPIR-V. Crust compiles the Rust-subset kernels to SPIR-V; this turns each
module of a `--push-buffer` build into WGSL with the `naga` command line tool (the translator
wgpu and Firefox use):

    cargo install naga-cli --locked
    python3 tools/spv2wgsl.py build/spirv            # writes NAME.wgsl next to NAME.spv
    python3 build.py --push-buffer --wgsl --out gpu/spirv --crust ../crust    # build and convert

Changes made on the way, because WebGPU is stricter than Vulkan:

  * The scalar parameters of a kernel (binding 30, a read-write storage buffer in the SPIR-V) become a
    `var<uniform>`. WebGPU allows only 10 storage buffers per stage on most adapters, and
    `solve_collisions_filtered` needs 10 besides the parameters.
  * The converted module is read back and validated by naga, so a bad conversion fails here
    and not in the browser.

The manifest gets `"wgsl": "NAME.wgsl"` for every module and `_meta.wgsl = {"params": "uniform"}`.
"""
import json
import os
import re
import shutil
import subprocess
import sys

PARAMS = re.compile(r"(@group\(0\) @binding\(30\)\s*\n)var<storage, read_write> (\w+): (\w+);")


class WgslError(RuntimeError):
    pass


def find_naga(explicit=None):
    for c in (explicit, os.environ.get("NAGA"), shutil.which("naga"),
              os.path.expanduser("~/.cargo/bin/naga")):
        if c and os.path.exists(c):
            return c
    raise WgslError("naga not found: cargo install naga-cli --locked (or set $NAGA)")


def convert(out_dir, naga=None, quiet=False):
    naga = find_naga(naga)
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath) as fh:
        manifest = json.load(fh)
    if manifest.get("_meta", {}).get("push_mode") != "buffer":
        raise WgslError("%s was not built with --push-buffer; WebGPU has no push constants" % out_dir)
    done = []
    for name, rec in manifest.items():
        if name == "_meta":
            continue
        spv = os.path.join(out_dir, rec["file"])
        wgsl = os.path.join(out_dir, name + ".wgsl")
        p = subprocess.run([naga, spv, wgsl], capture_output=True, text=True)
        if p.returncode != 0:
            raise WgslError("naga could not convert %s:\n%s%s" % (rec["file"], p.stdout, p.stderr))
        with open(wgsl) as fh:
            text = fh.read()
        text, n = PARAMS.subn(r"\1var<uniform> \2: \3;", text)
        want = sum(1 for e in rec["entry_points"] if e.get("push"))
        if n != want:
            raise WgslError("%s: %d parameter buffers turned into uniforms, expected %d" % (name, n, want))
        with open(wgsl, "w") as fh:
            fh.write(text)
        p = subprocess.run([naga, wgsl], capture_output=True, text=True)      # validate what a browser will read
        if p.returncode != 0:
            raise WgslError("converted %s.wgsl does not validate:\n%s%s" % (name, p.stdout, p.stderr))
        rec["wgsl"] = name + ".wgsl"
        done.append(name)
        if not quiet:
            print("  wgsl  %-24s %6d bytes, %d uniform parameter block(s)" % (name + ".wgsl", len(text), n))
    manifest["_meta"]["wgsl"] = {"params": "uniform"}
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return done


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        convert(sys.argv[1])
    except WgslError as e:
        sys.exit("spv2wgsl: %s" % e)
