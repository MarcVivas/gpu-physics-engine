#!/usr/bin/env python3
"""Build the GPU physics engine's compute kernels with Crust, not rustc/cargo.

The engine's GPU work is written in WGSL (src/**/*.wgsl) and driven by wgpu
from Rust. This script bypasses rustc, cargo and the WGSL toolchain: it takes
the Rust-subset ports in kernels/*.rs and compiles each one to SPIR-V with
Crust's `--target spirv` back end.

    python3 build.py                 compile every kernel to build/spirv/*.spv
    python3 build.py --check         ... and run each in Crust's SPIR-V
                                     interpreter against a Python model
    python3 build.py --status        also list every WGSL shader and what is
                                     left to port
    python3 build.py --only grid     just one kernel (repeatable)
    python3 build.py --crust DIR     where Crust lives (default: $CRUST_DIR,
                                     ./crust, ../crust, ~/crust)

Outputs, in build/spirv/: NAME.spv for each kernel file, and manifest.json
with every entry point's workgroup size, buffer bindings and push-constant
layout -- what a host needs to create the pipeline and bind its buffers.

A kernel is a `#[kernel(local_size = N)]` function. Its `&[T]` / `&mut [T]`
parameters are storage buffers (numbered in order, or pinned with
`#[binding(N)]`) and its scalar parameters are the push constants. The
host-side Rust (winit, wgpu, rendering) is not compiled here.

Exit status: 0 when every kernel built (and passed --check); 1 otherwise.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(HERE, "kernels")
SRC = os.path.join(HERE, "src")


# ------------------------------------------------------------ locate Crust

def find_crust(explicit):
    """The Crust checkout: --crust, $CRUST_DIR, then the usual places."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("CRUST_DIR"):
        candidates.append(os.environ["CRUST_DIR"])
    candidates += [os.path.join(HERE, "crust"),
                   os.path.join(HERE, "..", "crust"),
                   os.path.expanduser("~/crust")]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(os.path.join(c, "shivyc", "spirv_gen.py")):
            return c
    sys.exit("build.py: cannot find Crust with the SPIR-V back end "
             "(shivyc/spirv_gen.py).\n  Pass --crust DIR or set CRUST_DIR.\n"
             "  Tried: " + ", ".join(os.path.abspath(c) for c in candidates))


# ----------------------------------------------------------------- compile

def compile_kernel(crust, src, out, push_buffer=False):
    """Run Crust on one kernel file. Returns (ok, message, lowering notes)."""
    env = dict(os.environ)
    env["CRUST_SPIRV_REPORT"] = "1"
    if push_buffer:
        env["CRUST_SPIRV_PUSH"] = "buffer"
    env["PYTHONPATH"] = crust + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "shivyc.main", "--target", "spirv", src,
           "-o", out]
    p = subprocess.run(cmd, cwd=crust, env=env, capture_output=True,
                       text=True)
    notes = {}
    other = []
    for line in p.stderr.splitlines():
        m = re.match(r"spirv: (\w+): (.*)", line)
        if m:
            notes[m.group(1)] = m.group(2)
        elif line.strip():
            other.append(line)
    text = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(other + p.stdout.splitlines()))
    if p.returncode != 0 or not os.path.exists(out):
        return False, text.strip() or "crust failed (exit %d)" % p.returncode, \
            notes
    return True, text.strip(), notes


def describe(crust, spv):
    """Validate a module with Crust's validator and read back its interface."""
    sys.path.insert(0, crust)
    from shivyc import spirv_reader
    from shivyc.spirv import bytes_to_words
    with open(spv, "rb") as fh:
        words = bytes_to_words(fh.read())
    errors = spirv_reader.validate(words)
    return errors, (spirv_reader.describe(words) if not errors else [])


# ------------------------------------------------------------------ status

def classify(path, text):
    """Why a WGSL shader is not (yet) portable, or None if it should be."""
    if "@vertex" in text or "@fragment" in text:
        return "render shader (vertex/fragment): not a compute kernel"
    why = []
    if "subgroup" in text:
        why.append("subgroup operations")
    if "var<workgroup>" in text or "workgroupBarrier" in text:
        why.append("workgroup memory and barriers (SPIRV.md phase 6)")
    if "atomic" in text:
        why.append("atomics (SPIRV.md phase 6)")
    return ", ".join(why) or None


def shader_status(built):
    """[(wgsl path, [entry points], state, detail)] for every .wgsl file."""
    rows = []
    for path in sorted(glob.glob(os.path.join(SRC, "**", "*.wgsl"),
                                 recursive=True)):
        with open(path) as fh:
            text = fh.read()
        entries = re.findall(r"@compute[^\n]*\n\s*fn\s+(\w+)", text)
        rel = os.path.relpath(path, HERE)
        if not entries:
            rows.append((rel, [], "skip", classify(path, text) or ""))
            continue
        todo = [e for e in entries if e not in built]
        if not todo:
            rows.append((rel, entries, "ported", ""))
        else:
            rows.append((rel, entries, "todo", classify(path, text)
                         or "not ported yet"))
    return rows


# -------------------------------------------------------------------- main

def main(argv):
    ap = argparse.ArgumentParser(
        description="Compile the physics engine's GPU kernels with Crust "
                    "(Rust subset -> SPIR-V), bypassing rustc and cargo.")
    ap.add_argument("--crust", help="path to the Crust checkout")
    ap.add_argument("--out", default=os.path.join(HERE, "build", "spirv"),
                    help="output directory (default build/spirv)")
    ap.add_argument("--only", action="append", default=[],
                    help="build only this kernel file (name without .rs)")
    ap.add_argument("--check", action="store_true",
                    help="run each kernel in the SPIR-V interpreter against "
                         "a Python model of the WGSL")
    ap.add_argument("--status", action="store_true",
                    help="list every WGSL shader and its port status")
    ap.add_argument("--push-buffer", action="store_true",
                    help="keep each kernel's scalar parameters in a read-only "
                         "storage buffer (binding 30) instead of push "
                         "constants: no device feature needed, and what the "
                         "wgpu / WebGPU hosts use")
    ap.add_argument("--wgsl", action="store_true",
                    help="also write NAME.wgsl for WebGPU (implies --push-buffer; needs the naga "
                         "command line tool, see tools/spv2wgsl.py)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 while any compute shader is still unported")
    a = ap.parse_args(argv)
    if a.wgsl:
        a.push_buffer = True

    a.out = os.path.abspath(a.out)
    crust = find_crust(a.crust)
    os.makedirs(a.out, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(KERNELS, "*.rs")))
    if a.only:
        sources = [s for s in sources
                   if os.path.basename(s)[:-3] in a.only]
        if not sources:
            sys.exit("build.py: no kernel named %s in kernels/"
                     % ", ".join(a.only))

    print("crust  %s" % crust)
    print("out    %s\n" % os.path.relpath(a.out, HERE))

    manifest = {}
    failed = 0
    built_entries = {}
    t0 = time.time()
    for src in sources:
        name = os.path.basename(src)[:-3]
        out = os.path.join(a.out, name + ".spv")
        if os.path.exists(out):
            os.remove(out)
        ok, msg, notes = compile_kernel(crust, src, out, a.push_buffer)
        if not ok:
            failed += 1
            print("  FAIL  %-24s" % (name + ".rs"))
            for line in msg.splitlines()[:8]:
                print("          " + line)
            continue
        errors, entries = describe(crust, out)
        if errors:
            failed += 1
            print("  FAIL  %-24s the module did not validate" % (name + ".rs"))
            for e in errors[:5]:
                print("          " + e)
            continue
        slow = [fn for fn in notes if notes[fn].startswith("dispatcher")]
        size = os.path.getsize(out)
        names = ", ".join(e["name"] for e in entries)
        print("  ok    %-24s %5d bytes  %s" % (name + ".rs", size, names))
        if slow:
            print("          note: %s fell back to the slow dispatcher loop "
                  "(%s)" % (", ".join(slow), notes[slow[0]]))
        for e in entries:
            built_entries[e["name"]] = name
        manifest[name] = {"file": name + ".spv", "entry_points": entries}

    manifest["_meta"] = {"push_mode": "buffer" if a.push_buffer else "push_constants"}
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    if a.wgsl and not failed:
        sys.path.insert(0, os.path.join(HERE, "tools"))
        import spv2wgsl
        try:
            spv2wgsl.convert(a.out)
        except spv2wgsl.WgslError as e:
            print("  FAIL  wgsl: %s" % e)
            failed += 1

    print("\n%d of %d kernel files built in %.1fs"
          % (len(sources) - failed, len(sources), time.time() - t0))

    if a.check and not failed:
        print("\nchecking against the WGSL models:")
        sys.path.insert(0, KERNELS)
        import selftest
        bad = selftest.run(a.out, crust)
        if bad:
            failed += bad
        else:
            print("all checks passed")
    elif a.check:
        print("\n--check skipped: fix the build failures first")

    unported = 0
    if a.status or a.strict:
        print("\nWGSL shaders:")
        for rel, entries, state, detail in shader_status(built_entries):
            label = {"ported": "ported", "todo": "TODO  ",
                     "skip": "n/a   "}[state]
            line = "  %s %-38s %s" % (label, rel, ", ".join(entries))
            if detail:
                line += "   <- " + detail
            print(line)
            if state == "todo":
                unported += 1
        print("\n%d compute shader file(s) still to port" % unported)

    if failed:
        return 1
    if a.strict and unported:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
