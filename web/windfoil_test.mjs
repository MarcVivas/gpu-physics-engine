// node web/windfoil_test.mjs : the closed form against supersampling, on the CPU.
import { buildPieces, boxWinding, coverage, windingAt, circleContour, polyContour, blobContour, reverseContour } from "./windfoil.js";

let failed = 0;
const check = (name, ok, extra = "") => { console.log((ok ? "ok   " : "FAIL ") + name + (extra ? "   " + extra : "")); if (!ok) failed++; };

function compare(name, path, rule, box, px = 0.37, ss = 24) {
  const pieces = buildPieces(path), [x0, y0, x1, y1] = box;
  let worst = 0, sum = 0, n = 0;
  for (let y = y0; y < y1; y += px) for (let x = x0; x < x1; x += px) {
    const cx = x + px / 2, cy = y + px / 2, F = boxWinding(pieces, cx, cy, px / 2, px / 2);
    let acc = 0;                                              // supersampled box-filtered winding
    for (let j = 0; j < ss; j++) for (let i = 0; i < ss; i++) acc += windingAt(path, x + (i + 0.5) / ss * px, y + (j + 0.5) / ss * px);
    const ref = acc / (ss * ss), d = Math.abs(F - ref);
    worst = Math.max(worst, d); sum += d; n++;
    if (rule) { const dc = Math.abs(coverage(F, rule) - coverage(ref, rule)); worst = Math.max(worst, dc); }
  }
  check(name, worst < 0.02, `worst ${worst.toFixed(4)} mean ${(sum / n).toFixed(5)} over ${n} pixels`);
}

compare("a circle of 8 arcs", [circleContour(5, 5, 3.2)], "nonzero", [0.5, 0.5, 9.5, 9.5]);
compare("the same, clockwise (negative winding)", [circleContour(5, 5, 3.2, 8, true)], "nonzero", [0.5, 0.5, 9.5, 9.5]);
compare("a ring: outer counter-clockwise, inner clockwise", [circleContour(5, 5, 3.5), circleContour(5, 5, 2, 8, true)], "nonzero", [0.5, 0.5, 9.5, 9.5]);
compare("a triangle with straight edges", [polyContour([1.1, 1.3, 8.7, 2.2, 4.4, 8.9])], "nonzero", [0.5, 0.5, 9.5, 9.5]);
const star = [];
for (let i = 0; i < 5; i++) { const a = Math.PI / 2 + i * 4 * Math.PI / 5; star.push(5 + 4 * Math.cos(a), 5 + 4 * Math.sin(a)); }
compare("a pentagram, nonzero", [polyContour(star)], "nonzero", [0.5, 0.5, 9.5, 9.5]);
compare("a pentagram, even-odd (the middle is a hole)", [polyContour(star)], "evenodd", [0.5, 0.5, 9.5, 9.5]);
compare("a smooth blob", [blobContour([2, 2, 8, 1.5, 9, 6, 5, 9, 1.5, 6.5])], "nonzero", [0.5, 0.5, 9.5, 9.5]);
compare("a thin sliver (under one pixel wide)", [[[2, 5, 5, 5.45, 8, 5], [8, 5, 5, 4.55, 2, 5]]], "nonzero", [1.5, 4, 8.5, 6]);
compare("two overlapping circles, nonzero (winding 2 in the lens)", [circleContour(4, 5, 2.5), circleContour(6, 5, 2.5)], "nonzero", [0.5, 2, 9.5, 8]);
compare("two overlapping circles, even-odd", [circleContour(4, 5, 2.5), circleContour(6, 5, 2.5)], "evenodd", [0.5, 2, 9.5, 8]);
compare("pixels much larger than the shape (minification)", [circleContour(5, 5, 0.8)], "nonzero", [0, 0, 10, 10], 2.5);
// a pixel entirely inside is 1, entirely outside 0
{
  const pieces = buildPieces([circleContour(5, 5, 3)]);
  check("inside is exactly 1, outside exactly 0", Math.abs(boxWinding(pieces, 5, 5, 0.25, 0.25) - 1) < 1e-5 && Math.abs(boxWinding(pieces, 9, 9, 0.25, 0.25)) < 1e-6);
}
console.log(failed ? failed + " FAILED" : "all checks passed");
process.exit(failed ? 1 : 0);
