#!/usr/bin/env python3
"""Builds the WGSL kernels, serves web/ and runs web/test.html in headless Chromium (WebGPU on SwiftShader, no GPU needed).

    python3 web/check.py [--crust DIR] [--only ball,crate,...] [--kernels DIR]

Needs `naga` (cargo install naga-cli --locked) and Playwright for Node with Chromium (PLAYWRIGHT_BROWSERS_PATH).
"""
import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    kernels = arg("--kernels")
    work = tempfile.mkdtemp(prefix="gpuphys-web-")
    if not kernels:
        kernels = os.path.join(work, "spirv")
        cmd = [sys.executable, os.path.join(ROOT, "build.py"), "--wgsl", "--out", kernels]
        if arg("--crust"):
            cmd += ["--crust", arg("--crust")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stdout + r.stderr)
    os.symlink(HERE, os.path.join(work, "web"))
    if not os.path.exists(os.path.join(work, "spirv")):
        os.symlink(kernels, os.path.join(work, "spirv"))

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=work, **k)

        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://localhost:%d/web/test.html?kernels=../spirv" % port
    if arg("--only"):
        url += "&only=" + arg("--only")
    js = os.path.join(work, "run.js")
    with open(js, "w") as fh:
        fh.write(RUNNER)
    env = dict(os.environ)
    npm_root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    env["NODE_PATH"] = npm_root + os.pathsep + env.get("NODE_PATH", "")
    r = subprocess.run(["node", js, url], env=env, capture_output=True, text=True, timeout=int(arg("--timeout", 1500)))
    sys.stderr.write(r.stderr[-2000:] if r.returncode else "")
    try:
        results = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        sys.exit("no results:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    bad = 0
    for x in results:
        print(("ok   " if x["ok"] else "FAIL ") + x["name"] + ("   " + x["extra"] if x["extra"] else ""))
        bad += not x["ok"]
    print("%d FAILED" % bad if bad else "all checks passed")
    return 1 if bad else 0


RUNNER = r"""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ executablePath: process.env.CHROMIUM || '/opt/pw-browsers/chromium', args: ['--headless=new','--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=swiftshader','--use-webgpu-adapter=swiftshader','--ignore-gpu-blocklist','--enable-unsafe-swiftshader'] });
  const p = await b.newPage();
  p.on('pageerror', e => console.error('[pageerror]', e.message));
  await p.goto(process.argv[2]);
  await p.waitForFunction(() => window.__done, null, { timeout: 0, polling: 500 });
  console.log(JSON.stringify(await p.evaluate(() => window.__results)));
  await b.close();
})();
"""

if __name__ == "__main__":
    sys.exit(main())
