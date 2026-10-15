"""Known-answer tests for the Crust ports of the GPU kernels.

`python3 build.py --check` compiles the kernels and then calls run(spv_dir,
crust_dir) here. Each test runs one compiled .spv module in Crust's reference
SPIR-V interpreter and compares every output byte with a plain-Python model of
what the original WGSL does -- including f32 rounding after every operation,
so a mismatch is a real difference and not float noise.

No GPU, no Vulkan, no wgpu: the interpreter executes the SPIR-V instruction by
instruction and refuses undefined behaviour (out-of-bounds, divide by zero).
"""
import math
import os
import random
import struct
import sys


def f32(x):
    """Round a Python float to the nearest f32."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def pack_f(vals):
    return bytearray(struct.pack("<%df" % len(vals), *vals))


def pack_u(vals):
    return bytearray(struct.pack("<%dI" % len(vals), *vals))


def unpack_f(buf):
    return list(struct.unpack("<%df" % (len(buf) // 4), bytes(buf)))


def unpack_u(buf):
    return list(struct.unpack("<%dI" % (len(buf) // 4), bytes(buf)))


def groups(n, local=64):
    return ((n + local - 1) // local, 1, 1)


# ---------------------------------------------------------------- models

def split_by_bits(n):
    x = n & 0x0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x


def morton(cx, cy):
    return (split_by_bits(cx & 0xFFFFFFFF)
            | (split_by_bits(cy & 0xFFFFFFFF) << 1)) & 0xFFFFFFFF


def cell_of(v, cell_size):
    return math.floor(f32(f32(v) / f32(cell_size)))


UNUSED = 0xFFFFFFFF


# ----------------------------------------------------------------- tests

def test_home_cell_ids(mods, rnd):
    machine = mods["home_cell_ids"]
    n = 100
    cell = 2.5
    pos = [f32(rnd.uniform(-9, 60)) for _ in range(2 * n)]
    home = pack_u([0xDEADBEEF] * n)
    ids = pack_u([0xDEADBEEF] * n)
    machine.run("create_home_cell_ids", groups(n),
                {0: pack_f(pos), 1: home, 2: ids},
                struct.pack("<If", n, cell))
    want_home = [morton(cell_of(pos[2 * i], cell), cell_of(pos[2 * i + 1], cell))
                 for i in range(n)]
    assert unpack_u(home) == want_home, "home cell ids differ"
    assert unpack_u(ids) == list(range(n)), "particle ids differ"


def test_rearrange(mods, rnd):
    machine = mods["rearrange"]
    n = 90
    order = list(range(n))
    rnd.shuffle(order)
    pos = [f32(rnd.uniform(0, 100)) for _ in range(2 * n)]
    prev = [f32(rnd.uniform(0, 100)) for _ in range(2 * n)]
    rad = [f32(rnd.uniform(0.1, 2)) for _ in range(n)]
    out_pos = pack_f([0.0] * (2 * n))
    out_rad = pack_f([0.0] * n)
    out_prev = pack_f([0.0] * (2 * n))
    machine.run("rearrange", groups(n),
                {0: pack_f(pos), 1: pack_f(rad), 2: pack_f(prev),
                 3: pack_u(order), 4: out_pos, 5: out_rad, 6: out_prev},
                struct.pack("<I", n))
    for i in range(n):
        j = order[i]
        assert unpack_f(out_pos)[2 * i:2 * i + 2] == pos[2 * j:2 * j + 2]
        assert unpack_f(out_prev)[2 * i:2 * i + 2] == prev[2 * j:2 * j + 2]
        assert unpack_f(out_rad)[i] == rad[j]


def verlet_model(cur, prv, r, dt, w, h, pressed, mouse):
    cx, cy = cur
    px, py = prv
    vx, vy = f32(cx - px), f32(cy - py)
    ax = ay = 0.0
    if pressed == 1:
        dx, dy = f32(mouse[0] - cx), f32(mouse[1] - cy)
        ln = f32(math.sqrt(f32(f32(dx * dx) + f32(dy * dy))))
        ax = f32(ax + f32(f32(dx / ln) * 150.0))
        ay = f32(ay + f32(f32(dy / ln) * 150.0))
    dt2 = f32(dt * dt)
    nx = f32(f32(f32(cx + vx) + f32(ax * dt2)))
    ny = f32(f32(f32(cy + vy) + f32(ay * dt2)))
    nx = f32(min(max(nx, r), f32(w - r)))
    ny = f32(min(max(ny, r), f32(h - r)))
    return (nx, ny), (cx, cy)


def test_verlet(mods, rnd):
    machine = mods["particle_integration"]
    n = 70
    w, h, dt = 100.0, 60.0, 0.016
    for pressed in (0, 1):
        # Some particles start outside the walls so every clamp bound (both
        # axes, low and high) is exercised, not just the unconstrained path.
        pos = []
        for _ in range(n):
            pos.append(f32(rnd.uniform(-3, w + 3)))
            pos.append(f32(rnd.uniform(-3, h + 3)))
        prv = [f32(p - rnd.uniform(-0.3, 0.3)) for p in pos]
        rad = [f32(rnd.uniform(0.2, 1.0)) for _ in range(n)]
        mouse = (f32(33.3), f32(21.7))
        pos_b, prv_b = pack_f(pos), pack_f(prv)
        machine.run("verlet_integration",
                    groups(n), {0: pos_b, 1: prv_b, 2: pack_f(rad)},
                    struct.pack("<fffIffI", dt, w, h, pressed, mouse[0],
                                mouse[1], n))
        got_pos, got_prv = unpack_f(pos_b), unpack_f(prv_b)
        for i in range(n):
            (nx, ny), (ox, oy) = verlet_model(
                (pos[2 * i], pos[2 * i + 1]), (prv[2 * i], prv[2 * i + 1]),
                rad[i], f32(dt), f32(w), f32(h), pressed, mouse)
            assert (got_prv[2 * i], got_prv[2 * i + 1]) == (ox, oy)
            assert abs(got_pos[2 * i] - nx) <= 1e-5 * max(1.0, abs(nx)), \
                (pressed, i, got_pos[2 * i], nx)
            assert abs(got_pos[2 * i + 1] - ny) <= 1e-5 * max(1.0, abs(ny)), \
                (pressed, i, got_pos[2 * i + 1], ny)


def grid_model(pos, rad, cell_size, n):
    cell_ids = [None] * (4 * n)
    object_ids = [None] * (4 * n)
    for o in range(n):
        px, py = pos[2 * o], pos[2 * o + 1]
        sq = f32(rad[o] * rad[o])
        hx, hy = cell_of(px, cell_size), cell_of(py, cell_size)
        base = o * 4
        cell_ids[base] = morton(hx, hy)
        object_ids[base] = o
        count = 0
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                if x == 0 and y == 0:
                    continue
                nx, ny = hx + x, hy + y
                minx, miny = f32(nx * cell_size), f32(ny * cell_size)
                maxx, maxy = f32(minx + cell_size), f32(miny + cell_size)
                qx = f32(min(max(px, minx), maxx))
                qy = f32(min(max(py, miny), maxy))
                dx, dy = f32(px - qx), f32(py - qy)
                if f32(f32(dx * dx) + f32(dy * dy)) < sq:
                    count += 1
                    cell_ids[base + count] = morton(nx, ny)
                    object_ids[base + count] = o
        for slot in range(count + 1, 4):
            cell_ids[base + slot] = UNUSED
    return cell_ids, object_ids


def test_grid(mods, rnd):
    machine = mods["grid"]
    n = 80
    cell = 2.0
    pos = [f32(rnd.uniform(0.5, 30)) for _ in range(2 * n)]
    rad = [f32(rnd.uniform(0.2, 1.3)) for _ in range(n)]
    cell_ids = pack_u([0x11111111] * (4 * n))
    object_ids = pack_u([0x22222222] * (4 * n))
    machine.run("build_cell_ids_array", groups(n),
                {0: pack_f(pos), 2: cell_ids, 3: object_ids, 4: pack_f(rad)},
                struct.pack("<fI", cell, n))
    want_c, want_o = grid_model(pos, rad, cell, n)
    got_c = unpack_u(cell_ids)
    for i in range(4 * n):
        assert got_c[i] == want_c[i], ("cell id", i, got_c[i], want_c[i])
    got_o = unpack_u(object_ids)
    for i in range(4 * n):
        if want_o[i] is not None:
            assert got_o[i] == want_o[i], ("object id", i)
        else:
            assert got_o[i] == 0x22222222, ("unused slot was written", i)
    assert any(c != UNUSED for c in want_c[1::4]), "no phantom cells tested"


def test_collision_cells(mods, rnd):
    machine = mods["collision_cell_builder"]
    for trial in range(6):
        total = rnd.choice([1, 2, 5, 17, 64, 130])
        used = max(0, total - rnd.randint(0, 6))
        ids = sorted(rnd.randint(0, max(1, total // 3)) for _ in range(used))
        ids += [UNUSED] * (total - used)
        chunks = (total + 3) // 4
        # The semantic model: every run of two or more equal ids (not the
        # UNUSED marker) is one collision cell, recorded by where it starts.
        starts = []
        i = 0
        while i < total:
            j = i
            while j + 1 < total and ids[j + 1] == ids[i]:
                j += 1
            if ids[i] != UNUSED and j > i:
                starts.append(i)
            i = j + 1
        counts = pack_u([0] * chunks)
        uniform = pack_u([chunks, total])
        machine.run("count_objects_for_each_chunk", groups(chunks),
                    {0: counts, 3: uniform, 4: pack_u(ids)})
        # inclusive prefix sum of the per-chunk counts (the engine does this
        # on the GPU between the two passes)
        running = 0
        summed = []
        for c in unpack_u(counts):
            running += c
            summed.append(running)
        assert running == len(starts), ("trial", trial, "chunk counts sum to",
                                        running, "expected", len(starts))
        cells = pack_u([0xCAFEF00D] * (len(starts) + 2))
        args = pack_u([0xFFFFFFFF] * 3)
        machine.run("build_collision_cells_array", groups(max(chunks, 1)),
                    {0: pack_u(summed), 1: cells, 2: args, 3: uniform,
                     4: pack_u(ids)})
        got = unpack_u(cells)
        assert got[:len(starts)] == starts, ("trial", trial, ids, got, starts)
        assert got[len(starts):] == [0xCAFEF00D] * 2, "wrote past the end"
        want_x = (len(starts) + 63) // 64
        assert unpack_u(args) == [want_x, 1, 1], ("dispatch args",
                                                 unpack_u(args))


def unsplit(n):
    x = n & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF
    return x


def solver_model(pos, rad, ids, objs, cells, total, color):
    """collision_solver.wgsl, one collision cell after another (the
    interpreter runs invocations in order, so that is the same schedule)."""
    pos = list(pos)
    for start in cells:
        h = ids[start]
        gx, gy = unsplit(h), unsplit(h >> 1)
        if 1 + gx % 2 + (gy % 2) * 2 != color:
            continue
        i = start
        while i < total and ids[i] == h:
            a = objs[i]
            j = i + 1
            while j < total and ids[j] == h:
                b = objs[j]
                ax, ay, bx, by = pos[2*a], pos[2*a+1], pos[2*b], pos[2*b+1]
                ra, rb = rad[a], rad[b]
                vx, vy = f32(ax - bx), f32(ay - by)
                d = f32(math.sqrt(f32(f32(vx * vx) + f32(vy * vy))))
                rs = f32(ra + rb)
                if f32(rs * rs) > f32(d * d) and d > f32(0.0001):
                    pen = f32(rs - d)
                    dx, dy = f32(vx / d), f32(vy / d)
                    st = f32(0.6)
                    cx = f32(f32(dx * pen) * st)
                    cy = f32(f32(dy * pen) * st)
                    ia, ib = f32(1.0 / ra), f32(1.0 / rb)
                    fa = f32(ia / f32(ia + ib))
                    fb = f32(ib / f32(ia + ib))
                    pos[2*a] = f32(ax + f32(cx * fa))
                    pos[2*a+1] = f32(ay + f32(cy * fa))
                    pos[2*b] = f32(bx - f32(cx * fb))
                    pos[2*b+1] = f32(by - f32(cy * fb))
                j += 1
            i += 1
    return pos


def test_collision_solver(mods, rnd):
    machine = mods["collision_solver"]
    touched = 0
    for trial in range(8):
        n = rnd.choice([6, 20, 50])
        pos = [f32(rnd.uniform(0, 6)) for _ in range(2 * n)]
        rad = [f32(rnd.uniform(0.2, 0.9)) for _ in range(n)]
        pairs = sorted((morton(int(pos[2*k] // 2), int(pos[2*k+1] // 2)), k)
                       for k in range(n))
        ids = [p[0] for p in pairs]
        objs = [p[1] for p in pairs]
        total = len(pairs)
        cells = [i for i in range(total)
                 if (i == 0 or ids[i - 1] != ids[i])
                 and i + 1 < total and ids[i + 1] == ids[i]]
        if not cells:
            continue
        counts = [0, 0, len(cells)]
        for color in (1, 2, 3, 4):
            want = solver_model(pos, rad, ids, objs, cells, total, color)
            buf = pack_f(pos)
            machine.run("solve_collisions", groups(len(cells)),
                        {0: pack_u(counts), 1: pack_u(cells), 2: pack_u(ids),
                         3: pack_u(objs), 4: buf, 5: pack_f(rad),
                         6: pack_u([3, total])}, pack_u([color]))
            got = unpack_f(buf)
            assert got == want, ("trial", trial, "color", color,
                                 [k for k in range(len(want))
                                  if got[k] != want[k]][:6])
            touched += sum(1 for k in range(len(want)) if want[k] != pos[k])
    assert touched, "no collision was ever resolved"


M32 = 0xFFFFFFFF


def test_prefix_sum(mods, rnd):
    machine = mods["prefix_sum"]
    for total in (1, 100, 255, 256, 257, 700, 1024):
        blocks = (total + 255) // 256
        data = [rnd.randint(0, 0xFFFFFFFF if total == 700 else 1000)
                for _ in range(total)]
        # pass 1: per-block inclusive scan; block totals only for full blocks
        want = []
        marker = 0xABCD1234
        sums = [marker] * 256
        for b in range(blocks):
            run = 0
            for i in range(b * 256, min(total, b * 256 + 256)):
                run = (run + data[i]) & M32
                want.append(run)
            if (b + 1) * 256 <= total:
                sums[b] = run
        buf = pack_u(data)
        sbuf = pack_u([marker] * 256)
        machine.run("prefix_sum_of_each_block", (blocks, 1, 1),
                    {0: buf, 1: sbuf}, pack_u([total]))
        assert unpack_u(buf) == want, ("pass 1 data", total)
        assert unpack_u(sbuf) == sums, ("pass 1 block sums", total)

        # pass 2: inclusive scan of the 256 block sums
        raw = [rnd.randint(0, M32) for _ in range(256)]
        want2, run = [], 0
        for v in raw:
            run = (run + v) & M32
            want2.append(run)
        sbuf = pack_u(raw)
        machine.run("prefix_sum_of_the_block_sums", (1, 1, 1),
                    {0: pack_u([0]), 1: sbuf})
        assert unpack_u(sbuf) == want2, ("pass 2", total)

        # pass 3: add the previous block's (scanned) total to each element
        scanned = [rnd.randint(0, M32) for _ in range(256)]
        d3 = [rnd.randint(0, M32) for _ in range(total)]
        want3 = [(d3[i] + (scanned[i // 256 - 1] if i >= 256 else 0)) & M32
                 for i in range(total)]
        buf = pack_u(d3)
        machine.run("add_block_prefix_sums_to_the_buffer", (blocks, 1, 1),
                    {0: buf, 1: pack_u(scanned)}, pack_u([total]))
        assert unpack_u(buf) == want3, ("pass 3", total)


def test_radix_sort(mods, rnd):
    machine = mods["radix_sort"]

    def one_pass(keys, pay, shift, wgs, blocks):
        n = len(keys)
        push = pack_u([n, shift, wgs, blocks])
        hist = pack_u([0xFFFFFFFF] * (wgs * 256))
        machine.run("build_histogram", (wgs, 1, 1),
                    {0: pack_u(keys), 1: hist}, push)
        want_hist = [[0] * 256 for _ in range(wgs)]
        for i, k in enumerate(keys):
            want_hist[i // (blocks * 256)][(k >> shift) & 255] += 1
        got = unpack_u(hist)
        assert got == [c for row in want_hist for c in row], \
            ("histogram", n, shift)
        out_k = pack_u([0xDEADBEEF] * n)
        out_p = pack_u([0xDEADBEEF] * n)
        machine.run("scatter_keys", (wgs, 1, 1),
                    {0: pack_u(keys), 1: hist, 2: pack_u(pay), 3: out_k,
                     4: out_p}, push)
        order = sorted(range(n), key=lambda i: (keys[i] >> shift) & 255)
        assert unpack_u(out_k) == [keys[i] for i in order], \
            ("scatter keys", n, shift)
        assert unpack_u(out_p) == [pay[i] for i in order], \
            ("scatter payload (stability)", n, shift)
        return unpack_u(out_k), unpack_u(out_p)

    # a single pass, several shapes
    for n, wgs, blocks in ((1, 1, 1), (300, 2, 1), (700, 2, 2)):
        keys = [rnd.randint(0, 0xFFFFFFFF) for _ in range(n)]
        if n == 700:
            keys = [k & 0x0F0F0F0F for k in keys]      # many equal digits
        pay = list(range(n))
        one_pass(keys, pay, 8, wgs, blocks)
    # four passes sort a whole key
    n = 600
    keys = [rnd.randint(0, 0xFFFFFFFF) for _ in range(n)]
    pay = list(range(n))
    k, p = keys, pay
    for shift in (0, 8, 16, 24):
        k, p = one_pass(k, p, shift, 3, 1)
    assert k == sorted(keys), "four passes did not sort"
    assert [keys[i] for i in p] == k, "payload did not follow its key"


def test_prowl_world(mods, rnd):
    machine = mods["prowl_world"]
    # --- integrate_world
    n = 150
    pos = [f32(rnd.uniform(-1, 21)) for _ in range(2 * n)]
    prv = [f32(pos[k] + f32(rnd.uniform(-0.2, 0.2))) for k in range(2 * n)]
    rad = [f32(rnd.choice([0.25, 0.5, 0.5, 0.75])) for _ in range(n)]
    dt, gx, gy, damp, w, h = (f32(1 / 60), f32(0.0), f32(-18.0), f32(0.99),
                              f32(20.0), f32(12.0))
    want_p, want_q = list(pos), list(prv)
    for idx in range(n):
        i = 2 * idx
        cx, cy = pos[i], pos[i + 1]
        vx = f32(f32(cx - prv[i]) * damp)
        vy = f32(f32(cy - prv[i + 1]) * damp)
        dt2 = f32(dt * dt)
        nx = f32(f32(cx + vx) + f32(gx * dt2))
        ny = f32(f32(cy + vy) + f32(gy * dt2))
        want_q[i], want_q[i + 1] = cx, cy
        r = rad[idx]
        nx = min(max(nx, r), f32(w - r))
        ny = min(max(ny, r), f32(h - r))
        want_p[i], want_p[i + 1] = nx, ny
    bp, bq = pack_f(pos), pack_f(prv)
    machine.run("integrate_world", groups(n),
                {0: bp, 1: bq, 2: pack_f(rad)},
                pack_f([dt, gx, gy, damp, w, h]) + pack_u([n]))
    assert unpack_f(bp) == want_p, "integrate_world positions"
    assert unpack_f(bq) == want_q, "integrate_world previous positions"

    # --- tile_collide
    tw, th = 12, 8
    tiles = [0] * (tw * th)
    for x in range(tw):
        tiles[x] = 1                                  # a floor, row 0
    for y in range(th):
        tiles[y * tw + 11] = 1                        # a wall, column 11
    for x in range(4, 7):
        tiles[3 * tw + x] = 1                         # a platform, row 3
    n = 200
    pos = [f32(rnd.uniform(0, tw)) if k % 2 == 0 else f32(rnd.uniform(0, th))
           for k in range(2 * n)]
    rad = [f32(rnd.choice([0.3, 0.5, 0.5, 0.9])) for _ in range(n)]

    def solid(tx, ty):
        return 0 <= tx < tw and 0 <= ty < th and tiles[ty * tw + tx] != 0

    want = list(pos)
    pushed = 0
    for idx in range(n):
        r = rad[idx]
        x, y = want[2 * idx], want[2 * idx + 1]
        x0, x1 = math.floor(f32(x - r)), math.floor(f32(x + r))
        y0, y1 = math.floor(f32(y - r)), math.floor(f32(y + r))
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                if not solid(tx, ty):
                    continue
                mnx, mny = f32(tx), f32(ty)
                qx = min(max(x, mnx), f32(mnx + 1.0))
                qy = min(max(y, mny), f32(mny + 1.0))
                dx, dy = f32(x - qx), f32(y - qy)
                d2 = f32(f32(dx * dx) + f32(dy * dy))
                if d2 > 0.0:
                    if d2 < f32(r * r):
                        d = f32(math.sqrt(d2))
                        push = f32(f32(r - d) / d)
                        x = f32(x + f32(dx * push))
                        y = f32(y + f32(dy * push))
                else:
                    left, right = f32(x - mnx), f32(f32(mnx + 1.0) - x)
                    down, up = f32(y - mny), f32(f32(mny + 1.0) - y)
                    best, mx, my = 1.0e9, 0.0, 0.0
                    if not solid(tx - 1, ty) and left < best:
                        best, mx, my = left, -f32(left + r), 0.0
                    if not solid(tx + 1, ty) and right < best:
                        best, mx, my = right, f32(right + r), 0.0
                    if not solid(tx, ty - 1) and down < best:
                        best, mx, my = down, 0.0, -f32(down + r)
                    if not solid(tx, ty + 1) and up < best:
                        best, mx, my = up, 0.0, f32(up + r)
                    x, y = f32(x + mx), f32(y + my)
        if (x, y) != (want[2 * idx], want[2 * idx + 1]):
            pushed += 1
        want[2 * idx], want[2 * idx + 1] = x, y
    assert pushed > 20, "the test never pushed anything out of a tile"
    buf = pack_f(pos)
    machine.run("tile_collide", groups(n),
                {0: buf, 2: pack_f(rad), 5: pack_u(tiles)},
                pack_u([tw, th, n]))
    assert unpack_f(buf) == want, "tile_collide"


def _close(got, want, tol=2e-4, what=""):
    bad = [k for k in range(len(want)) if abs(got[k] - want[k]) > tol]
    assert not bad, (what, bad[:6], [(got[k], want[k]) for k in bad[:3]])


def test_prowl_bodies(mods, rnd):
    m = mods["prowl_bodies"]
    # --- solve_joints: distance, rope, strut and a fixed anchor (tolerance compare: the
    # model works in doubles)
    n = 24
    pos = [rnd.uniform(0, 10) for _ in range(2 * n)]
    rad = [f32(rnd.choice([0.25, 0.5])) for _ in range(n)]
    ji, jf, model = [], [], []
    for k in range(12):
        a, b = 2 * k, 2 * k + 1
        flags = k % 3                                   # 0 rigid, 1 rope, 2 strut
        rest, stiff = rnd.uniform(0.3, 3.0), rnd.choice([1.0, 0.5])
        if k == 11:
            b = 0xFFFFFFFF
        ji += [a, b, flags, 0]
        jf += [rest, stiff, 4.0, 5.0]
        model.append((a, b, flags, rest, stiff))
    want = list(pos)
    for a, b, flags, rest, stiff in model:
        ax, ay = want[2 * a], want[2 * a + 1]
        bx, by = (4.0, 5.0) if b == 0xFFFFFFFF else (want[2 * b], want[2 * b + 1])
        ia = 1.0 / rad[a]
        ib = 0.0 if b == 0xFFFFFFFF else 1.0 / rad[b]
        d = math.hypot(bx - ax, by - ay)
        diff = d - rest
        if (flags == 1 and diff < 0) or (flags == 2 and diff > 0) or d < 1e-4:
            continue
        corr = diff * stiff / (ia + ib)
        ux, uy = (bx - ax) / d, (by - ay) / d
        want[2 * a], want[2 * a + 1] = ax + ux * corr * ia, ay + uy * corr * ia
        if b != 0xFFFFFFFF:
            want[2 * b], want[2 * b + 1] = bx - ux * corr * ib, by - uy * corr * ib
    buf = pack_f(pos)
    m.run("solve_joints", groups(12), {0: buf, 2: pack_f(rad), 6: pack_u(ji), 7: pack_f(jf)}, pack_u([0, 12]))
    _close(unpack_f(buf), want, what="solve_joints")

    # --- integrate_tagged: tag 1 uses sand damping
    n = 40
    pos = [f32(rnd.uniform(1, 9)) for _ in range(2 * n)]
    prv = [f32(pos[k] + rnd.uniform(-0.1, 0.1)) for k in range(2 * n)]
    rad = [f32(0.3)] * n
    tags = [rnd.choice([0, 1, 2]) for _ in range(n)]
    dt, gx, gy, damp, sd = 1 / 60, 0.0, -18.0, 0.999, 0.9
    want_p = list(pos)
    for i in range(n):
        d = sd if tags[i] == 1 else damp
        for ax_, g in ((0, gx), (1, gy)):
            c, q = pos[2 * i + ax_], prv[2 * i + ax_]
            want_p[2 * i + ax_] = min(max(c + (c - q) * d + g * dt * dt, 0.3), (10.0 if ax_ == 0 else 8.0) - 0.3)
    bp, bq = pack_f(pos), pack_f(prv)
    m.run("integrate_tagged", groups(n), {0: bp, 1: bq, 2: pack_f(rad), 3: pack_u(tags)},
          pack_f([dt, gx, gy, damp, sd, 10.0, 8.0]) + pack_u([n]))
    _close(unpack_f(bp), want_p, 1e-5, "integrate_tagged")
    assert unpack_f(bq) == pos, "integrate_tagged previous"

    # --- tile_friction on a floor: the sliding speed along it shrinks, the fall does not change
    tw, th = 6, 3
    tiles = [1] * tw + [0] * (tw * (th - 1))
    pos = [2.0, 1.25, 3.0, 2.5, 4.0, 1.25]
    prv = [1.9, 1.26, 2.9, 2.6, 4.0, 1.15]               # 0 touches the floor, 1 is in the air, 2 is leaving it
    bq = pack_f(prv)
    m.run("tile_friction", groups(3), {0: pack_f(pos), 1: bq, 2: pack_f([0.25] * 3), 5: pack_u(tiles)},
          pack_u([tw, th, 3]) + pack_f([0.5, 0.5]))
    got = unpack_f(bq)
    _close(got[2:4], prv[2:4], 1e-6, "airborne circle untouched")
    assert abs((pos[0] - got[0]) - 0.05) < 1e-4, ("friction halves the tangential velocity", got)
    assert abs(got[1] - prv[1]) < 1e-6, "falling into the floor is not absorbed"
    assert abs((pos[5] - got[5]) - 0.05) < 1e-4, ("half of the leaving speed is absorbed", got)

    # --- raycast against tiles and circles
    tiles = [0] * (tw * th * 2)
    tw, th = 8, 6
    tiles = [0] * (tw * th)
    for x in range(tw):
        tiles[x] = 1
    for y in range(th):
        tiles[y * tw + 7] = 1
    tiles[3 * tw + 3] = 1
    cpos = [5.0, 2.0, 2.0, 4.5]
    crad = [0.5, 0.4]
    cgrp = [0, 7]
    rays = []
    for _ in range(40):
        a = rnd.uniform(0, 6.283)
        rays.append((rnd.uniform(0.5, 6.5), rnd.uniform(1.2, 5.0), math.cos(a), math.sin(a), 20.0))
    def model_ray(ox, oy, dx, dy, mx, mask, ignore):
        best, kind, nx, ny = mx, 0, 0.0, 0.0
        if mask & 1:
            t = 0.0
            while t < best:
                px, py = ox + dx * (t + 1e-4), oy + dy * (t + 1e-4)
                tx, ty = math.floor(px), math.floor(py)
                if 0 <= tx < tw and 0 <= ty < th and tiles[ty * tw + tx]:
                    best, kind = t, 1
                    break
                t += 5e-4
        if mask & 2:
            for c in range(2):
                if ignore and cgrp[c] == ignore:
                    continue
                ex, ey = ox - cpos[2 * c], oy - cpos[2 * c + 1]
                b = ex * dx + ey * dy
                cc = ex * ex + ey * ey - crad[c] ** 2
                disc = b * b - cc
                if disc >= 0:
                    t = 0.0 if cc < 0 else -b - math.sqrt(disc)
                    if 0 <= t < best:
                        best, kind = t, 2
        return best if kind else -1.0, kind
    for mask, ignore in ((3, 0), (1, 0), (2, 0), (3, 7)):
        flat = []
        for r in rays:
            flat += list(r) + [0.0, 0.0, 0.0]
        hits, ids = pack_f([0.0] * (4 * len(rays))), pack_u([0] * len(rays))
        m.run("raycast", groups(len(rays)),
              {0: pack_f(cpos), 2: pack_f(crad), 3: pack_u(cgrp), 5: pack_u(tiles), 8: pack_f(flat),
               9: hits, 10: ids},
              pack_u([tw, th, 2, len(rays), mask, ignore]))
        h = unpack_f(hits)
        nhit = 0
        for i, r in enumerate(rays):
            t, kind = model_ray(*r, mask, ignore)
            assert int(h[4 * i + 3]) == kind, ("ray", i, mask, ignore, h[4 * i:4 * i + 4], t, kind)
            if kind:
                nhit += 1
                assert abs(h[4 * i] - t) < 2e-3, ("ray t", i, h[4 * i], t)
                if t > 1e-3:
                    assert abs(math.hypot(h[4 * i + 1], h[4 * i + 2]) - 1.0) < 1e-3, ("unit normal", i, h[4*i:4*i+4])
        assert nhit > 5, "rays never hit anything"

    # --- sensor_count: a box and a circle, with a tag filter
    n = 60
    pos = [f32(rnd.uniform(0, 10)) for _ in range(2 * n)]
    rad = [f32(rnd.choice([0.1, 0.3])) for _ in range(n)]
    tags = [rnd.choice([0, 1, 2]) for _ in range(n)]
    sensors = [5.0, 5.0, 2.0, 1.0,   3.0, 3.0, 1.5, -1.0,   8.0, 2.0, 1.0, 1.0]
    masks = [0, 0, 1 << 1]
    want = []
    for s in range(3):
        sx, sy, hw, hh = sensors[4 * s:4 * s + 4]
        cnt = 0
        for i in range(n):
            if masks[s] and not (masks[s] >> tags[i]) & 1:
                continue
            x, y, r = pos[2 * i], pos[2 * i + 1], rad[i]
            if hh < 0:
                hit = (x - sx) ** 2 + (y - sy) ** 2 < (hw + r) ** 2
            else:
                qx, qy = min(max(x, sx - hw), sx + hw), min(max(y, sy - hh), sy + hh)
                hit = (x - qx) ** 2 + (y - qy) ** 2 < r * r
            cnt += hit
        want.append(cnt)
    counts = pack_u([0, 0, 0])
    m.run("sensor_count", groups(n), {0: pack_f(pos), 2: pack_f(rad), 4: pack_u(tags), 11: pack_f(sensors),
                                      12: pack_u(masks), 13: counts}, pack_u([n, 3]))
    assert unpack_u(counts) == want, ("sensor_count", unpack_u(counts), want)
    assert sum(want) > 3, "no sensor counted anything"

    # --- solve_collisions_grouped: no groups = solve_collisions; one shared group = no change
    for trial in range(4):
        n = 30
        pos = [f32(rnd.uniform(0, 6)) for _ in range(2 * n)]
        rad = [f32(rnd.uniform(0.2, 0.9)) for _ in range(n)]
        pairs = sorted((morton(int(pos[2*k] // 2), int(pos[2*k+1] // 2)), k) for k in range(n))
        ids, objs, total = [p[0] for p in pairs], [p[1] for p in pairs], len(pairs)
        cells = [i for i in range(total) if (i == 0 or ids[i - 1] != ids[i]) and i + 1 < total and ids[i + 1] == ids[i]]
        if not cells:
            continue
        counts = [0, 0, len(cells)]
        for color in (1, 2, 3, 4):
            for grp, same in ((0, False), (9, True)):
                buf = pack_f(pos)
                m.run("solve_collisions_grouped", groups(len(cells)),
                      {0: pack_u(counts), 1: pack_u(cells), 2: pack_u(ids), 3: pack_u(objs), 4: buf,
                       5: pack_f(rad), 6: pack_u([3, total]), 7: pack_u([grp] * n)}, pack_u([color]))
                want = pos if same else solver_model(pos, rad, ids, objs, cells, total, color)
                assert unpack_f(buf) == want, ("grouped", trial, color, grp)


def _can_collide(ca, ma, ga, cb, mb, gb):
    if ga != 0 and ga == gb:
        return ga > 0
    return (ma & cb) != 0 and (mb & ca) != 0


def merge5(mat, inv):
    """prowl_material's per-circle array: [friction, bounciness, gravity_scale, damping, inv_mass]."""
    return [v for i in range(len(inv)) for v in (mat[4 * i:4 * i + 4] + [inv[i]])]


def filtered_model(pos, prv, rad, inv, flt, mat, ids, objs, cells, total, color, thr):
    """solve_collisions_filtered in doubles (tolerance compare), cell after cell."""
    pos, prv = list(pos), list(prv)
    for start in cells:
        h = ids[start]
        if 1 + unsplit(h) % 2 + (unsplit(h >> 1) % 2) * 2 != color:
            continue
        i = start
        while i < total and ids[i] == h:
            a = objs[i]
            j = i + 1
            while j < total and ids[j] == h:
                b = objs[j]
                j += 1
                ga = flt[4 * a + 2] - (1 << 32) if flt[4 * a + 2] >= 1 << 31 else flt[4 * a + 2]
                gb = flt[4 * b + 2] - (1 << 32) if flt[4 * b + 2] >= 1 << 31 else flt[4 * b + 2]
                if flt[4 * a + 3] != 0 and flt[4 * a + 3] == flt[4 * b + 3]:
                    continue
                if not _can_collide(flt[4 * a], flt[4 * a + 1], ga, flt[4 * b], flt[4 * b + 1], gb):
                    continue
                w = inv[a] + inv[b]
                if w <= 0:
                    continue
                vx, vy = pos[2 * a] - pos[2 * b], pos[2 * a + 1] - pos[2 * b + 1]
                d = math.hypot(vx, vy)
                rs = rad[a] + rad[b]
                if not (rs * rs > d * d and d > 1e-4):
                    continue
                nx, ny = vx / d, vy / d
                corr = (rs - d) * 0.6
                sa, sb = inv[a] / w, inv[b] / w
                rvx = (pos[2 * a] - prv[2 * a]) - (pos[2 * b] - prv[2 * b])
                rvy = (pos[2 * a + 1] - prv[2 * a + 1]) - (pos[2 * b + 1] - prv[2 * b + 1])
                vn = rvx * nx + rvy * ny
                vt = rvx * -ny + rvy * nx
                pos[2 * a] += nx * corr * sa
                pos[2 * a + 1] += ny * corr * sa
                pos[2 * b] -= nx * corr * sb
                pos[2 * b + 1] -= ny * corr * sb
                mu = math.sqrt(mat[4 * a] * mat[4 * b])
                e = max(mat[4 * a + 1], mat[4 * b + 1])
                want, load = vn, corr
                if vn <= 0:
                    want = e * -vn if -vn > thr else 0.0
                    load = corr - vn
                dvn = want - (vn + corr)
                lim = mu * load
                dvt = min(max(-vt, -lim), lim)
                cx, cy = dvn * nx + dvt * -ny, dvn * ny + dvt * nx
                prv[2 * a] -= cx * sa
                prv[2 * a + 1] -= cy * sa
                prv[2 * b] += cx * sb
                prv[2 * b + 1] += cy * sb
            i += 1
    return pos, prv


def _filtered_run(m, pos, prv, rad, inv, flt, mat, thr=0.01):
    """All circles in one collision cell (colour 1); returns (positions, previous)."""
    n = len(rad)
    bp, bq = pack_f(pos), pack_f(prv)
    m.run("solve_collisions_filtered", groups(1),
          {0: pack_u([0, 0, 1]), 1: pack_u([0]), 2: pack_u([0] * n), 3: pack_u(list(range(n))), 4: bp,
           5: pack_f(rad), 6: pack_u([3, n]), 7: pack_u(flt), 8: pack_f(merge5(mat, inv)), 10: bq},
          pack_u([1]) + pack_f([thr]))
    return unpack_f(bp), unpack_f(bq)


def test_prowl_material(mods, rnd):
    m = mods["prowl_material"]
    # --- integrate_material: gravity scale and damping per circle, zero inverse mass stays put
    n = 40
    pos = [f32(rnd.uniform(1, 9)) for _ in range(2 * n)]
    prv = [f32(pos[k] + rnd.uniform(-0.1, 0.1)) for k in range(2 * n)]
    rad = [f32(0.3)] * n
    mat = []
    for k in range(n):
        mat += [0.4, 0.0, rnd.choice([0.0, 0.5, 1.0, -1.0]), rnd.choice([1.0, 0.9])]
    inv = [0.0 if k % 5 == 0 else 1.0 / 0.3 for k in range(n)]
    dt, gx, gy, damp = 1 / 60, 0.0, -18.0, 0.999
    want_p, want_q = list(pos), list(prv)
    for i in range(n):
        if inv[i] == 0.0:
            want_q[2 * i], want_q[2 * i + 1] = pos[2 * i], pos[2 * i + 1]
            continue
        for ax_, g in ((0, gx), (1, gy)):
            c, q = pos[2 * i + ax_], prv[2 * i + ax_]
            nxt = c + (c - q) * damp * mat[4 * i + 3] + g * mat[4 * i + 2] * dt * dt
            want_p[2 * i + ax_] = min(max(nxt, 0.3), (10.0 if ax_ == 0 else 8.0) - 0.3)
            want_q[2 * i + ax_] = c
    bp, bq = pack_f(pos), pack_f(prv)
    m.run("integrate_material", groups(n), {0: bp, 1: bq, 2: pack_f(rad), 11: pack_f(merge5(mat, inv))},
          pack_f([dt, gx, gy, damp, 10.0, 8.0]) + pack_u([n]))
    _close(unpack_f(bp), want_p, 1e-5, "integrate_material positions")
    _close(unpack_f(bq), want_q, 1e-6, "integrate_material previous")

    # --- solve_collisions_filtered against its model: random circles, filters, masses, materials
    touched = 0
    for trial in range(10):
        n = rnd.choice([6, 14, 30])
        pos = [f32(rnd.uniform(0, 6)) for _ in range(2 * n)]
        prv = [f32(pos[k] + rnd.uniform(-0.15, 0.15)) for k in range(2 * n)]
        rad = [f32(rnd.uniform(0.2, 0.9)) for _ in range(n)]
        inv = [0.0 if rnd.random() < 0.1 else f32(rnd.uniform(0.5, 4.0)) for _ in range(n)]
        flt, mat = [], []
        for k in range(n):
            flt += [1 << rnd.randrange(3), rnd.choice([M32, 0b011, 0b101, 0]), rnd.choice([0, 0, 0, 3, -3]) & M32, rnd.choice([0, 0, 0, 7])]
            mat += [rnd.choice([0.0, 0.4, 1.0]), rnd.choice([0.0, 0.5, 1.0]), 1.0, 1.0]
        pairs = sorted((morton(int(pos[2*k] // 2), int(pos[2*k+1] // 2)), k) for k in range(n))
        ids = [p[0] for p in pairs]
        objs = [p[1] for p in pairs]
        total = len(pairs)
        cells = [i for i in range(total)
                 if (i == 0 or ids[i - 1] != ids[i]) and i + 1 < total and ids[i + 1] == ids[i]]
        if not cells:
            continue
        for color in (1, 2, 3, 4):
            wp, wq = filtered_model(pos, prv, rad, inv, flt, mat, ids, objs, cells, total, color, 0.01)
            bp, bq = pack_f(pos), pack_f(prv)
            m.run("solve_collisions_filtered", groups(len(cells)),
                  {0: pack_u([0, 0, len(cells)]), 1: pack_u(cells), 2: pack_u(ids), 3: pack_u(objs), 4: bp,
                   5: pack_f(rad), 6: pack_u([3, total]), 7: pack_u(flt), 8: pack_f(merge5(mat, inv)), 10: bq},
                  pack_u([color]) + pack_f([0.01]))
            _close(unpack_f(bp), wp, 1e-3, "filtered positions")
            _close(unpack_f(bq), wq, 1e-3, "filtered previous")
            touched += sum(1 for k in range(len(wp)) if abs(wp[k] - pos[k]) > 1e-9)
    assert touched, "no collision was ever resolved"

    # --- known answers: two equal circles, a runs into a resting b (overlap 0.1, approach 0.2 per step)
    ones = [1.0, 1.0]
    all_f = [1, M32, 0, 0, 1, M32, 0, 0]
    def pair(e, vy=0.0, mu=0.4, flt=all_f, inv=ones):
        mat = [mu, e, 1.0, 1.0, mu, e, 1.0, 1.0]
        p, q = _filtered_run(m, [1.0, 1.0, 1.9, 1.0], [0.8, 1.0 - vy, 1.9, 1.0], [0.5, 0.5], inv, flt, mat)
        return p, q, (p[0] - q[0], p[1] - q[1]), (p[2] - q[2], p[3] - q[3])
    _, _, va, vb = pair(1.0, mu=0.0)
    assert abs(va[0]) < 1e-5 and abs(vb[0] - 0.2) < 1e-5, ("bounciness 1 swaps the velocities", va, vb)
    _, _, va, vb = pair(0.0, mu=0.0)
    assert abs(va[0] - 0.1) < 1e-5 and abs(vb[0] - 0.1) < 1e-5, ("bounciness 0 leaves them moving together", va, vb)
    _, _, va, vb = pair(0.0, vy=0.3, mu=0.0)
    assert abs(va[1] - 0.3) < 1e-5, ("no friction keeps the sliding speed", va)
    _, _, va, vb = pair(0.0, vy=0.3, mu=0.4)
    assert abs(va[1] - 0.248) < 1e-4 and abs(vb[1] - 0.052) < 1e-4, ("friction takes mu * load, shared", va, vb)
    # filters: a mask without the other's category, or a negative shared group, lets them pass through
    for fl, name in (([1, 0b10, 0, 0, 1, M32, 0, 0], "mask"), ([1, M32, -2 & M32, 0, 1, M32, -2 & M32, 0], "group")):
        p, q, _, _ = pair(0.0, flt=fl)
        assert p == [1.0, 1.0, f32(1.9), 1.0] and q == [f32(0.8), 1.0, f32(1.9), 1.0], ("filtered pair untouched", name, p, q)
    p, q, _, _ = pair(0.0, flt=[1, 0, 5, 0, 1, 0, 5, 0])
    assert p != [1.0, 1.0, f32(1.9), 1.0], "a shared positive group collides even with empty masks"
    p, q, _, _ = pair(0.0, flt=[1, M32, 5, 9, 1, M32, 5, 9])
    assert p == [1.0, 1.0, f32(1.9), 1.0], "circles of one body never collide, whatever their group"
    # mass: a heavy circle (inverse mass 0.1) barely moves, an immovable one (0) does not
    p, q, _, _ = pair(0.0, inv=[1.0, 0.1])
    assert abs(p[2] - f32(1.9)) < abs(p[0] - 1.0) / 5, ("the heavy one is pushed less", p)
    p, q, _, _ = pair(0.0, inv=[1.0, 0.0])
    assert p[2] == f32(1.9) and q[2] == f32(1.9), ("an immovable circle is not moved", p, q)

    # --- tile_material on a floor
    tw, th = 6, 3
    tiles = [1] * tw + [0] * (tw * (th - 1))
    for circ_mu, circ_e, tile_mu, tile_e, want_slide, want_leave in (
            (0.5, 0.0, 0.5, 0.0, 0.5, 1.0), (0.0, 0.0, 0.5, 0.0, 0.0, 1.0), (0.5, 0.75, 0.5, 0.0, 0.5, 0.25),
            (0.5, 0.0, 0.5, 1.0, 0.5, 0.0)):
        pos = [2.0, 1.25, 4.0, 1.25]
        prv = [1.9, 1.26, 4.0, 1.15]                    # 0 slides on the floor, 1 is leaving it
        bq = pack_f(prv)
        m.run("tile_material", groups(2), {0: pack_f(pos), 1: bq, 2: pack_f([0.25] * 2), 5: pack_u(tiles),
                                           11: pack_f(merge5([circ_mu, circ_e, 1.0, 1.0] * 2, [1.0, 1.0])), 14: pack_f([0.0, 0.0])},
              pack_u([tw, th, 2]) + pack_f([tile_mu, tile_e, 0.05]))
        got = unpack_f(bq)
        assert abs((pos[0] - got[0]) - 0.1 * (1.0 - math.sqrt(circ_mu * tile_mu))) < 1e-4, ("friction", circ_mu, tile_mu, got)
        assert abs((pos[3] - got[3]) - 0.1 * (1.0 - want_leave)) < 1e-4, ("leaving speed kept", circ_e, tile_e, got)
    # a circle driven into the floor bounces off at e times its speed; a slow one just stops
    for vy, e, want in ((0.4, 0.5, 0.2), (0.4, 0.0, 0.0), (0.4, 1.0, 0.4), (0.01, 1.0, 0.0)):
        bq = pack_f([2.0, 1.25 + vy])
        m.run("tile_material", groups(1), {0: pack_f([2.0, 1.25]), 1: bq, 2: pack_f([0.25]), 5: pack_u(tiles),
                                           11: pack_f(merge5([0.0, e, 1.0, 1.0], [1.0])), 14: pack_f([0.0])},
              pack_u([tw, th, 1]) + pack_f([0.5, 0.0, 0.05]))
        assert abs((1.25 - unpack_f(bq)[1]) - want) < 1e-5, ("bounce off the floor", vy, e, unpack_f(bq))
    # friction acts on the slip: a circle rolling without slipping (spin = speed / radius) keeps its speed
    # (rolling to the right on the floor: v.t = -vx for t = (-1, 0), so omega = -vx / r = -0.1 / 0.25 per step)
    for om, want_slow in ((-0.4, 0.0), (0.0, 0.05)):
        bq = pack_f([2.0 - 0.1, 1.25])
        m.run("tile_material", groups(1), {0: pack_f([2.0, 1.25]), 1: bq, 2: pack_f([0.25]), 5: pack_u(tiles),
                                           11: pack_f(merge5([1.0, 0.0, 1.0, 1.0], [1.0])), 14: pack_f([om])},
              pack_u([tw, th, 1]) + pack_f([0.25, 0.0, 0.05]))
        lost = (2.0 - unpack_f(bq)[0]) - 0.1
        assert abs(-lost - want_slow) < 1e-4, ("rolling is not slowed, sliding is", om, lost)
    # an immovable circle is not touched
    bq = pack_f([1.9, 1.26])
    m.run("tile_material", groups(1), {0: pack_f([2.0, 1.25]), 1: bq, 2: pack_f([0.25]), 5: pack_u(tiles),
                                       11: pack_f(merge5([0.5, 0.0, 1.0, 1.0], [0.0])), 14: pack_f([0.0])},
          pack_u([tw, th, 1]) + pack_f([0.5, 0.0, 0.05]))
    assert unpack_f(bq) == [f32(1.9), f32(1.26)], "an immovable circle keeps its previous position"


def _bez_pt(c, t):
    u = 1 - t
    return (u * u * c[0] + 2 * u * t * c[2] + t * t * c[4], u * u * c[1] + 2 * u * t * c[3] + t * t * c[5])


def _nearest_dense(c, x, y, n=4000):
    best = (1e30, 0.0)
    for i in range(n + 1):
        qx, qy = _bez_pt(c, i / n)
        d = math.hypot(x - qx, y - qy)
        if d < best[0]:
            best = (d, i / n)
    return best


def curve_rec(c, friction=0.2, bounce=0.0, side=0):
    xs, ys = c[0::2], c[1::2]
    return list(c) + [min(xs), min(ys), max(xs), max(ys), friction, bounce, float(side), 0.0, 0.0, 0.0]


def test_prowl_curves(mods, rnd):
    m = mods["prowl_curves"]

    def collide(curves, circles, r=0.25, inv=1.0):
        """circles: [(x, y, px, py)] -> new positions."""
        n = len(circles)
        pos = pack_f([v for c in circles for v in c[:2]])
        prv = pack_f([v for c in circles for v in c[2:]])
        flat = [v for c in curves for v in c]
        m.run("curve_collide", groups(n), {0: pos, 1: prv, 2: pack_f([r] * n), 11: pack_f(merge5([0.4, 0, 1, 1] * n, [inv] * n)),
                                           15: pack_f(flat)}, pack_u([n, len(curves)]))
        u = unpack_f(pos)
        return [(u[2 * i], u[2 * i + 1]) for i in range(n)]

    # a thin bowl: from (1,3) down through (3,1) to (5,3): the lowest point is (3, 2)
    bowl = [1.0, 3.0, 3.0, 1.0, 5.0, 3.0]
    wall = curve_rec(bowl)
    # a circle on top of it is lifted to r above the curve, one under it is pushed down to r below
    (a,) = collide([wall], [(3.0, 2.1, 3.0, 2.5)])
    assert abs(a[0] - 3.0) < 1e-4 and abs(a[1] - 2.25) < 2e-3, ("lifted onto the bowl", a)
    (a,) = collide([wall], [(3.0, 1.9, 3.0, 1.2)])
    assert abs(a[0] - 3.0) < 1e-4 and abs(a[1] - 1.75) < 2e-3, ("pushed down below the thin wall", a)
    # a circle that is far enough stays; many random circles end exactly r from the curve, on their side
    pts = []
    for _ in range(60):
        x, y = rnd.uniform(0.5, 5.5), rnd.uniform(1.2, 3.4)
        pts.append((f32(x), f32(y)))
    got = collide([wall], [(x, y, x, y + 0.3) for x, y in pts])
    for (x, y), (gx, gy) in zip(pts, got):
        d0, _ = _nearest_dense(bowl, x, y)
        d1, _ = _nearest_dense(bowl, gx, gy)
        if d0 >= 0.25 - 1e-4:
            assert (gx, gy) == (x, y) or math.hypot(gx - x, gy - y) < 1e-6, ("far circles stay", (x, y), (gx, gy))
        else:
            assert abs(d1 - 0.25) < 3e-3, ("pushed out to r", (x, y), (gx, gy), d1)
    # the end of a thin wall: a circle past the tip is pushed away from the tip, not snapped to a face
    tip = bowl[:2]
    (a,) = collide([wall], [(0.95, 3.1, 1.5, 3.3)])
    assert abs(math.hypot(a[0] - tip[0], a[1] - tip[1]) - 0.25) < 1e-3, ("radial push at the free end", a)
    # an immovable circle is left alone
    (a,) = collide([wall], [(3.0, 2.1, 3.0, 2.5)], inv=0.0)
    assert a == (f32(3.0), f32(2.1)), ("static circle", a)

    # a solid hill: from (1,1) over (3,5) to (5,1), the inside is below (right of the direction): side = -1
    hill = [1.0, 1.0, 3.0, 5.0, 5.0, 1.0]
    solid = curve_rec(hill, side=-1)
    for x, y in ((3.0, 3.1), (3.0, 2.9), (3.0, 2.85)):
        (a,) = collide([solid], [(x, y, x, y)])
        d1, _ = _nearest_dense(hill, a[0], a[1])
        assert abs(d1 - 0.25) < 3e-3 and a[1] > 3.0, ("pushed out of the hill, upwards", (x, y), a)
    (a,) = collide([solid], [(3.0, 3.9, 3.0, 3.9)])
    assert a == (f32(3.0), f32(3.9)), ("clear of the hill", a)
    # the same shape with the inside above: a circle on the underside stays below, the cap is solid
    cap = curve_rec(hill, side=1)
    (a,) = collide([cap], [(3.0, 2.9, 3.0, 2.0)])
    assert a[1] < 3.0 and abs(_nearest_dense(hill, a[0], a[1])[0] - 0.25) < 3e-3, ("below a hill that is solid above", a)

    # --- curve_material: the same contact numbers as tile_material, on a flat curve
    flat = [1.0, 1.0, 3.0, 1.0, 5.0, 1.0]

    def material(mu_c, e_c, mu_w, e_w, pos, prv, om=0.0, side=0, thr=0.05):
        n = len(pos) // 2
        bq = pack_f(prv)
        rec = curve_rec(flat, friction=mu_w, bounce=e_w, side=side)
        m.run("curve_material", groups(n), {0: pack_f(pos), 1: bq, 2: pack_f([0.25] * n),
                                            11: pack_f(merge5([mu_c, e_c, 1.0, 1.0] * n, [1.0] * n)), 14: pack_f([om] * n),
                                            15: pack_f(rec)}, pack_u([n, 1]) + pack_f([thr]))
        return unpack_f(bq)
    for circ_mu, circ_e, w_mu, w_e, want_leave in ((0.5, 0.0, 0.5, 0.0, 1.0), (0.0, 0.0, 0.5, 0.0, 1.0),
                                                   (0.5, 0.75, 0.5, 0.0, 0.25), (0.5, 0.0, 0.5, 1.0, 0.0)):
        pos = [2.0, 1.25, 4.0, 1.25]
        prv = [1.9, 1.26, 4.0, 1.15]
        got = material(circ_mu, circ_e, w_mu, w_e, pos, prv)
        assert abs((pos[0] - got[0]) - 0.1 * (1.0 - math.sqrt(circ_mu * w_mu))) < 1e-4, ("friction", got)
        assert abs((pos[3] - got[3]) - 0.1 * (1.0 - want_leave)) < 1e-4, ("leaving speed kept", got)
    for vy, e, want in ((0.4, 0.5, 0.2), (0.4, 0.0, 0.0), (0.4, 1.0, 0.4), (0.01, 1.0, 0.0)):
        got = material(0.0, e, 0.5, 0.0, [2.0, 1.25], [2.0, 1.25 + vy])
        assert abs((1.25 - got[1]) - want) < 1e-5, ("bounce", vy, e, got)
    # a circle that is not touching, or under a solid contour, is untouched
    got = material(0.5, 0.0, 0.5, 0.0, [2.0, 2.0], [1.9, 2.05])
    assert got == [f32(1.9), f32(2.05)], ("no contact", got)
    got = material(0.5, 0.0, 0.5, 0.0, [2.0, 1.25], [1.9, 1.3], side=1)       # inside is on the left = above: the circle is inside
    assert got == [f32(1.9), f32(1.3)], ("inside a solid contour", got)
    got = material(0.5, 0.0, 0.5, 0.0, [2.0, 0.75], [1.9, 0.7], side=1)        # below it: outside
    assert got != [f32(1.9), f32(0.7)], ("outside a solid contour touches", got)


TESTS = [test_home_cell_ids, test_rearrange, test_verlet, test_grid,
         test_collision_cells, test_collision_solver, test_prefix_sum, test_radix_sort, test_prowl_world, test_prowl_bodies, test_prowl_material, test_prowl_curves]


class _Module:
    """One compiled .spv, runnable by entry point name. Kernels built with
    --push-buffer keep their scalar parameters in a storage buffer; the push
    bytes the tests pass are put there."""

    def __init__(self, interp, words, describe):
        self.machine = interp.Machine(words)
        self.iface = {e["name"]: e for e in describe(words)}

    def run(self, entry, grid, buffers, push=b""):
        pb = self.iface[entry].get("push_binding")
        if pb is not None:
            buffers = dict(buffers)
            buffers[pb["binding"]] = bytearray(push)
            push = b""
        self.machine.run(entry, grid, buffers, push)


def run(spv_dir, crust_dir, only=None, out=sys.stdout):
    """Run every test; returns the number of failures."""
    sys.path.insert(0, crust_dir)
    sys.path.insert(0, os.path.join(crust_dir, "tools"))
    import spirv_interp
    from shivyc.spirv import bytes_to_words
    from shivyc import spirv_reader
    needed = {"home_cell_ids", "rearrange", "particle_integration", "grid",
              "collision_cell_builder", "collision_solver", "prefix_sum", "radix_sort", "prowl_world", "prowl_bodies", "prowl_material", "prowl_curves"}
    mods = {}
    for name in sorted(needed):
        path = os.path.join(spv_dir, name + ".spv")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            mods[name] = _Module(spirv_interp, bytes_to_words(fh.read()),
                                 spirv_reader.describe)
    failures = 0
    for test in TESTS:
        short = test.__name__[len("test_"):]
        if only and short not in only:
            continue
        kernel = {"home_cell_ids": "home_cell_ids", "rearrange": "rearrange",
                  "verlet": "particle_integration", "grid": "grid",
                  "collision_cells": "collision_cell_builder",
                  "collision_solver": "collision_solver",
                  "prefix_sum": "prefix_sum",
                  "radix_sort": "radix_sort",
                  "prowl_world": "prowl_world", "prowl_bodies": "prowl_bodies",
                  "prowl_material": "prowl_material", "prowl_curves": "prowl_curves"}[short]
        if kernel not in mods:
            out.write("  skip  %-18s (%s.spv was not built)\n" % (short, kernel))
            continue
        try:
            test(mods, random.Random(20240601))
            out.write("  ok    %s\n" % short)
        except AssertionError as e:
            failures += 1
            out.write("  FAIL  %-18s %s\n" % (short, str(e)[:240]))
        except Exception as e:                       # interpreter refusals
            failures += 1
            out.write("  ERROR %-18s %s: %s\n" % (short, type(e).__name__,
                                                   str(e)[:240]))
    return failures
