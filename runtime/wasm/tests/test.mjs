// Runs the parity scenario in WebAssembly (and a few behaviours) and, when given a native run's output,
// checks both agree:  node tests/test.mjs [native-output.txt]
import { loadGpuPhys, Joint, Event } from "../gpuphys.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const gp = await loadGpuPhys(join(here, "..", "dist", "gpuphys.wasm"));
let failed = 0;
const check = (name, ok, extra = "") => { console.log((ok ? "ok   " : "FAIL ") + name + (extra ? "   " + extra : "")); if (!ok) failed++; };

// --- the parity scenario
function scenario() {
  const w = gp.createWorld({ capacity: 512, maxJoints: 1024, width: 20, height: 10, gravity: [0, -18] });
  const tiles = new Uint32Array(200);
  for (let x = 0; x < 20; x++) tiles[x] = 1;
  tiles[8 + 3 * 20] = tiles[9 + 3 * 20] = 1;
  w.setTiles(20, 10, tiles);
  w.enableEvents(true);
  const ball = w.addBall(5, 6, 0.35, 1, 0);
  const box = w.addBox(10, 6, 1.5, 1, 0.3);
  const hex = []; for (let i = 0; i < 6; i++) hex.push(Math.fround(0.6 * Math.cos(2 * Math.PI * i / 6)), Math.fround(0.6 * Math.sin(2 * Math.PI * i / 6)));
  const poly = w.addPolygon(14, 6, hex, { angle: 0.2 });
  const rope = w.addRope(3, 8, 7, 8, 10, 0.08, true, false);
  const bouncy = w.addBall(16, 7, 0.35);
  w.setBodyMaterial(bouncy, { friction: 0.4, bounciness: 0.8 });
  const plank = w.addBox(12.5, 8, 2, 0.3);
  const hinge = w.addJoint({ type: Joint.REVOLUTE, a: plank, b: -1, x: 11.5, y: 8 });
  for (let s = 0; s < 240; s++) w.step(1 / 60);
  return { w, ids: [ball, box, poly, rope, bouncy, plank], hinge };
}
const { w, ids, hinge } = scenario();
const lines = [];
for (const id of ids) { const p = w.bodyPose(id); lines.push(`body ${id} ${p.x.toFixed(3)} ${p.y.toFixed(3)} ${p.angle.toFixed(3)}`); }
lines.push(`hinge ${w.jointAngle(hinge).toFixed(3)}`);
lines.push(`events ${w.pollEvents(64).length}`);
lines.push(`ray ${w.raycast(2, 5, 0, -1, 100, { mask: 3 }) ? 1 : 0}`);
lines.push(`circles ${w.circleCount}`);
console.log(lines.join("\n"));

if (process.argv[2]) {
  const native = readFileSync(process.argv[2], "utf8").trim().split("\n");
  const close = (a, b) => {
    const ta = a.split(" "), tb = b.split(" ");
    return ta[0] === tb[0] && ta.length === tb.length && ta.slice(1).every((v, i) => Math.abs(parseFloat(v) - parseFloat(tb[i + 1])) < 0.02);
  };
  native.forEach((n, i) => check("matches the native build: " + n, close(n, lines[i] ?? ""), lines[i]));
}

// --- behaviours
{
  const w2 = gp.createWorld({ width: 20, height: 10, gravity: [0, -9.81] });
  const t = new Uint32Array(200); for (let x = 0; x < 20; x++) t[x] = 1;
  w2.setTiles(20, 10, t);
  const b = w2.addBall(5, 4, 0.35);
  for (let s = 0; s < 120; s++) w2.step(1 / 60);
  const p = w2.bodyPose(b);
  check("a ball falls and rests on the floor", Math.abs(p.y - 1.35) < 0.05, `y ${p.y.toFixed(3)}`);
  const hit = w2.raycast(5, 8, 0, -1, 100);
  check("a ray finds the ball", hit && hit.kind === 2 && hit.body === b);
  check("tileAt", w2.tileAt(3.5, 0.5) !== null && w2.tileAt(3.5, 5.5) === null);
}
{
  const w3 = gp.createWorld({ width: 20, height: 10, gravity: [0, -18] });
  const t = new Uint32Array(200); for (let x = 0; x < 20; x++) t[x] = 1;
  w3.setTiles(20, 10, t); w3.enableEvents(true);
  const b = w3.addBall(5, 4, 0.35);
  let began = 0;
  for (let s = 0; s < 90; s++) { w3.step(1 / 60); for (const e of w3.pollEvents()) if (e.kind === Event.BEGIN && e.a === b) began++; }
  check("a ball landing reports one contact begin", began === 1, `events ${began}`);
  const memBefore = w3.gp.x.memory.buffer.byteLength;
  for (let i = 0; i < 300; i++) w3.addBall(2 + (i % 15), 5 + Math.floor(i / 15) * 0.8, 0.2);   // enough to make the heap grow
  for (let s = 0; s < 20; s++) w3.step(1 / 60);
  check("hundreds of bodies are added and stepped", w3.circleCount === 301 && w3.readCircles().length === 903, `memory ${memBefore} -> ${w3.gp.x.memory.buffer.byteLength}`);
}
{
  const w4 = gp.createWorld({ width: 20, height: 10, gravity: [0, -18] });
  const t = new Uint32Array(200); for (let x = 0; x < 20; x++) t[x] = 1;
  w4.setTiles(20, 10, t); w4.enableEvents(true);
  let s = 11; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  check("sand is poured", w4.brushSand(10, 4, 0.6, 30, 0.08, rnd) === 30);
  for (let i = 0; i < 90; i++) w4.step(1 / 60);
  const pos = w4.readCircles(); let low = 9, ok = true;
  for (let i = 0; i < 30; i++) { low = Math.min(low, pos[3 * i + 1]); if (!Number.isFinite(pos[3 * i])) ok = false; }
  check("sand settles on the floor", ok && low > 1.0 + 0.08 - 0.03 && low < 1.4, `lowest grain ${low.toFixed(3)}`);
  check("free circles have no body and make no pairs", w4.circle(0).body === -1 && w4.bodyPairs().length === 0);
  const cx = [...Array(30).keys()].reduce((a, i) => a + pos[3 * i], 0) / 30;
  const pushed = w4.blast(cx - 1, 1.2, 3, 80);
  for (let i = 0; i < 5; i++) w4.step(1 / 60);
  const after = w4.readCircles(); const cx2 = [...Array(30).keys()].reduce((a, i) => a + after[3 * i], 0) / 30;
  check("a blast to the left pushes the sand right", pushed > 0 && cx2 > cx + 0.2, `moved ${(cx2 - cx).toFixed(2)}`);
}
{
  const w5 = gp.createWorld({ width: 20, height: 10, gravity: [0, -9.81] });
  check("a curve is added", w5.addCurve(4, 6, 10, 2, 16, 6, 0.4, 0, 0) === 0 && w5.curveCount === 1);
  const b = w5.addBall(8, 8, 0.5);
  for (let s = 0; s < 600; s++) w5.step(1 / 60);
  const p = w5.bodyPose(b);
  check("a ball rests in the bowl made of one Bezier curve", Math.abs(p.x - 10) < 1.2 && p.y > 4.45 && p.y < 4.75, `at (${p.x.toFixed(2)}, ${p.y.toFixed(2)})`);
  w5.clearCurves();
  check("clearCurves", w5.curveCount === 0);
}
console.log(failed ? failed + " FAILED" : "all checks passed");
process.exit(failed ? 1 : 0);
