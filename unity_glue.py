"""GPU 2D physics for Crust's unity_pack: a drop-in for Box2D-Packed's physics_box2d.c.

unity_pack packs a Unity (or Godot) project into engine.c + data.c; its 2D physics is a generated
glue file (physics_box2d.c, from Box2D-Packed's box2d_unity.py) over libbox2d.a. This module
generates physics_gpu.c instead, which meets the same contract

    engine_box2d_step()                       one fixed step, pushes / pulls the packed tables
    engine_box2d_raycast / _all               Physics2D.Raycast / RaycastAll
    engine_box2d_overlap_circle / _all / _point
    engine_box2d_queries_start_in_colliders

over this repository's circle physics (runtime/gpuphys.c: the Crust-translated GPU kernels run on
the CPU), so a packed player's Rigidbody2D / Collider2D scene runs on the same solver the GPU
kernels implement.

    from unity_glue import emit_glue
    emit_glue(outdir, plan)          # -> physics_gpu.c, gpuphys/*.c, gpuphys/gpuphys.h

What maps (see UNITY_PACK.md, "GPU physics"):
  dynamic CircleCollider2D     a ball          dynamic BoxCollider2D   a crate (a grid of circles)
  static Box / Circle          rasterised into the solid-tile grid (cell = 2 x the biggest circle)
  trigger colliders            OnTrigger*2D, from analytic overlap tests
  Collision2D enter/stay/exit  from body-body and body-tile contacts
  gravityScale, drag, velocity, teleports, layer masks, Raycast* / Overlap* queries
  kinematic bodies             held where the script puts them; they push, they are never pushed
  hinge (to the world), distance, spring, fixed joints
What is refused (PackError says why): capsule / polygon / terrain colliders, other joints, joint motors,
limits and breaking, Collision2D contact points, AddComponent'd Rigidbody2D or joints, several solid
colliders on one body.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "runtime"))

GLUE_FILE = "physics_gpu.c"
RUNTIME_FILES = ("gpuphys.c", "gpuphys.h")


class GpuPhysicsError(ValueError):
    """The scene uses something the GPU physics option cannot do."""


def find_root(root=None):
    """A gpu-physics-engine checkout: `root`, $GPU_PHYSICS_ROOT, or ../gpu-physics-engine."""
    for c in (root, os.environ.get("GPU_PHYSICS_ROOT"), HERE):
        if c and os.path.isfile(os.path.join(c, "runtime", "gpuphys.c")):
            return os.path.abspath(c)
    return None


def validate(plan):
    """Raise GpuPhysicsError for what this backend cannot simulate."""
    cols = plan.get("collider2d") or []
    rbs = plan.get("rigidbody2d") or []
    budget = (plan.get("addcomponent_budget") or {}).get("Rigidbody2D") or 0
    if budget:
        raise GpuPhysicsError("GPU physics: AddComponent<Rigidbody2D> is not supported (the world is built "
                              "from the authored bodies)")
    if int(plan.get("joints2d_cap") or 0) > len(plan.get("joints2d") or []):
        raise GpuPhysicsError("GPU physics: AddComponent<...Joint2D> is not supported (joints are built "
                              "from the authored ones)")
    for ji, j in enumerate(plan.get("joints2d") or []):
        kind = j.get("kind")
        if kind not in (0, 1, 2, 3):
            raise GpuPhysicsError("GPU physics: joint %d is a slider / wheel / friction / relative / target "
                                  "joint; only hinge, distance, spring and fixed joints are mapped" % ji)
        if j.get("use_motor") or j.get("use_limits"):
            raise GpuPhysicsError("GPU physics: joint %d uses a motor or limits; they are not mapped" % ji)
        if float(j.get("break_force", float("inf"))) < 1.0e37 or float(j.get("break_torque", float("inf"))) < 1.0e37:
            raise GpuPhysicsError("GPU physics: joint %d can break (breakForce / breakTorque); breaking is "
                                  "not mapped" % ji)
        rb_b = j.get("rb_b", -1)
        if kind == 0 and rb_b is not None and rb_b >= 0 and (plan.get("rigidbody2d") or [])[rb_b].get("body_type") != 2:
            raise GpuPhysicsError("GPU physics: hinge joint %d connects two moving bodies; a hinge needs the "
                                  "world or a static body" % ji)
    if plan.get("physics2d_contacts"):
        raise GpuPhysicsError("GPU physics: Collision2D contact points (GetContact / contactCount) are not "
                              "available; circle contacts have no manifold")
    if plan.get("physics2d_terrain"):
        raise GpuPhysicsError("GPU physics: terrain colliders are not supported")
    if plan.get("physics2d_polygons"):
        raise GpuPhysicsError("GPU physics: polygon colliders are not supported")
    if plan.get("godot"):
        raise GpuPhysicsError("GPU physics: Godot projects are not supported yet (Unity only)")
    solid = {}
    for ci, c in enumerate(cols):
        if c.get("kind") not in (0, 1):
            raise GpuPhysicsError("GPU physics: collider %d is a capsule / polygon / terrain; only box and "
                                  "circle colliders are supported" % ci)
        rb = c.get("rb2d", -1)
        if rb is not None and rb >= 0 and not c.get("is_trigger"):
            solid[rb] = solid.get(rb, 0) + 1
            if solid[rb] > 1:
                raise GpuPhysicsError("GPU physics: Rigidbody2D %d has several solid colliders" % rb)
    for ri, r in enumerate(rbs):
        if r.get("body_type", 0) in (0, 1) and not solid.get(ri):
            raise GpuPhysicsError("GPU physics: Rigidbody2D %d has no solid collider" % ri)


def emit_glue(outdir, plan, gpu_root=None, crust=None, sub_steps=None):
    """Write physics_gpu.c and the runtime sources under outdir/gpuphys/. Returns the paths."""
    root = find_root(gpu_root)
    if root is None:
        raise GpuPhysicsError("GPU physics needs a gpu-physics-engine checkout (--gpu-physics PATH, "
                              "$GPU_PHYSICS_ROOT, or ../gpu-physics-engine)")
    validate(plan)
    import gpk
    if crust is None:
        crust = os.environ.get("CRUST_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(
            __import__("shivyc").__file__)))
    sub = os.path.join(outdir, "gpuphys")
    os.makedirs(sub, exist_ok=True)
    paths = []
    for f in RUNTIME_FILES:
        dst = os.path.join(sub, f)
        _write_if_different(dst, open(os.path.join(root, "runtime", f)).read())
        paths.append(dst)
    for name in gpk.KERNEL_FILES:
        text, _ = gpk.translate_file(name, gpk.find_crust(crust))
        dst = os.path.join(sub, "gpk_%s.c" % name)
        _write_if_different(dst, "/* generated by gpu-physics-engine from kernels/%s.rs */\n%s" % (name, text))
        paths.append(dst)
    glue = _glue_text(plan)
    dst = os.path.join(outdir, GLUE_FILE)
    _write_if_different(dst, glue)
    paths.append(dst)
    return paths


def objects():
    """Object files the Makefile / build link for GPU physics, relative to the outdir."""
    import gpk
    return ["physics_gpu.o", "gpuphys/gpuphys.o"] + ["gpuphys/gpk_%s.o" % n for n in gpk.KERNEL_FILES]


def _write_if_different(path, text):
    try:
        if open(path).read() == text:
            return
    except OSError:
        pass
    with open(path, "w") as fh:
        fh.write(text)


COLLIDER_EXTERNS = """extern const int _Collider2D_count;
extern const int _Collider2D_kind[];
extern const int _Collider2D_is_trigger[];
extern const int _Collider2D_rb2d[];
extern const float _Collider2D_ox[];
extern const float _Collider2D_oy[];
extern const float _Collider2D_hw[];
extern const float _Collider2D_hh[];
extern const float _Collider2D_edge_r[];
extern const float _Collider2D_friction[];
extern const float _Collider2D_bounciness[];
extern const float _Collider2D_cos[];
extern const float _Collider2D_sin[];
extern const int _Collider2D_layer[];
"""


def _glue_text(plan):
    n_rb = max(1, len(plan.get("rigidbody2d") or []))
    cols = plan.get("collider2d") or []
    n_col = max(1, len(cols))
    decls = COLLIDER_EXTERNS
    if not cols:  # unity_pack emits no collider tables without colliders
        decls = (decls.replace("extern const", "static const").replace("_Collider2D_count;", "_Collider2D_count = 0;")
                 .replace("[];", "[1];"))
    flags = {
        "TRIGGERS": 1 if plan.get("physics2d_triggers") else 0,
        "LIVE": 1 if plan.get("physics2d_live") else 0,
        "ROTATION": 1 if plan.get("physics2d_rotation") else 0,
        "JOINTS": 1 if plan.get("joints2d") else 0,
        # Physics2D.queriesHitTriggers: Unity's default is on
        "QHT": 0 if plan.get("physics2d_queries_hit_triggers") in (0, False) else 1,
    }
    text = GLUE
    text = text.replace("@COLLIDER_DECLS@", decls)
    text = text.replace("@N_RB@", str(n_rb)).replace("@N_COL@", str(n_col))
    text = text.replace("@MAX_PAIRS@", str(max(1, min(4096, n_col * (n_col - 1) // 2))))
    for k, v in flags.items():
        text = text.replace("@%s@" % k, str(v))
    return text


GLUE = r"""/* Generated by gpu-physics-engine/unity_glue.py for unity_pack. Do not edit.
 * Unity's 2D physics on the GPU physics engine's circle solver (gpuphys), in place of Box2D-Packed. */
#include "gpuphys/gpuphys.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GPU_TRIGGERS @TRIGGERS@
#define GPU_LIVE @LIVE@
#define GPU_ROTATION @ROTATION@
#define GPU_JOINTS @JOINTS@
#define GPU_QUERIES_HIT_TRIGGERS @QHT@

/* unity_pack tables (data.c) */
extern float Time_fixedDeltaTime;
extern float Physics2D_gravity_x;
extern float Physics2D_gravity_y;
extern int _Rigidbody2D_count;
extern float _Rigidbody2D_vel_x[];
extern float _Rigidbody2D_vel_y[];
extern float _Rigidbody2D_gravity_scale[];
extern float _Rigidbody2D_linear_damping[];
extern int _Rigidbody2D_body_type[];
@COLLIDER_DECLS@
#if GPU_JOINTS
extern int _Joint2D_count;
extern int _Joint2D_kind[];
extern int _Joint2D_rb_a[];
extern int _Joint2D_rb_b[];
extern int _Joint2D_enabled[];
extern int _Joint2D_auto_anchor[];
extern int _Joint2D_auto_distance[];
extern int _Joint2D_max_distance_only[];
extern float _Joint2D_anchor_x[];
extern float _Joint2D_anchor_y[];
extern float _Joint2D_canchor_x[];
extern float _Joint2D_canchor_y[];
extern float _Joint2D_distance[];
extern float _Joint2D_frequency[];
#endif

/* unity_pack exports (engine.c) */
void engine_rb2d_get_pos( int rb, float* x, float* y );
void engine_rb2d_set_pos( int rb, float x, float y );
void engine_col2d_center( int ci, float* x, float* y );
void engine_col2d_contact( int a, int b );
#if GPU_TRIGGERS
void engine_col2d_trigger( int a, int b );
#endif
#if GPU_ROTATION
float engine_rb2d_get_rot( int rb );
void engine_rb2d_set_rot( int rb, float a );
#endif
#if GPU_LIVE
int engine_rb2d_live( int rb );
int engine_col2d_live( int ci );
#endif

void engine_box2d_step( void );
int engine_box2d_queries_start_in_colliders = 1;

enum
{
	MAX_RB = @N_RB@,
	MAX_COL = @N_COL@,
	MAX_PAIRS = @MAX_PAIRS@,
	MAX_BODIES = 8192,
	MAX_CELLS = 4 * 1024 * 1024,
	MAX_QUERY = 64,
};

static GpWorld* W;
static int ready;
static float cell, org_x, org_y; /* world units per tile; world position of tile (0, 0) */
static int tiles_w, tiles_h;
static int* tile_owner; /* collider index + 1 of the static collider that made a tile solid */

static int rb_body[MAX_RB]; /* gp body of a Rigidbody2D, -1 for none */
static int rb_col[MAX_RB]; /* its solid collider */
static float rb_offx[MAX_RB], rb_offy[MAX_RB]; /* collider centre - body origin */
static float last_x[MAX_RB], last_y[MAX_RB]; /* pose and velocity as last pulled */
static float last_vx[MAX_RB], last_vy[MAX_RB];
static int body_rb[MAX_BODIES];
static int rb_on[MAX_RB];

static float fixed_dt( void )
{
	return Time_fixedDeltaTime > 1e-8f ? Time_fixedDeltaTime : 1.0f / 60.0f;
}

/* ---- collider shapes (world space, from the packed tables) ---- */

/* local coordinates of (x, y) in collider ci's frame; returns the half extents */
static void col_local( int ci, float x, float y, float* lx, float* ly )
{
	float cx, cy;
	engine_col2d_center( ci, &cx, &cy );
	float c = _Collider2D_cos[ci], s = _Collider2D_sin[ci];
	float dx = x - cx, dy = y - cy;
	*lx = c * dx + s * dy;
	*ly = -s * dx + c * dy;
}

/* distance from (x, y) to collider ci's shape: negative inside */
static float col_distance( int ci, float x, float y )
{
	float lx, ly;
	col_local( ci, x, y, &lx, &ly );
	if ( _Collider2D_kind[ci] == 1 )
		return sqrtf( lx * lx + ly * ly ) - _Collider2D_hw[ci];
	float er = _Collider2D_edge_r[ci];
	float qx = fabsf( lx ) - _Collider2D_hw[ci], qy = fabsf( ly ) - _Collider2D_hh[ci];
	float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
	float out = sqrtf( ox * ox + oy * oy );
	float in = fmaxf( qx, qy );
	return ( out > 0 ? out : ( in < 0 ? in : 0 ) ) - er;
}

static int col_live( int ci )
{
#if GPU_LIVE
	return engine_col2d_live( ci );
#else
	(void)ci;
	return 1;
#endif
}

/* ---- world construction ---- */

static void body_dims( int ci, float* r_circle )
{
	if ( _Collider2D_kind[ci] == 1 )
		*r_circle = _Collider2D_hw[ci];
	else
	{
		float bw = 2 * _Collider2D_hw[ci], bh = 2 * _Collider2D_hh[ci];
		float r0 = fminf( bw, bh ) / 4.0f;
		int rows = (int)floorf( bh / ( 2 * r0 ) + 0.5f ), cols = (int)floorf( bw / ( 2 * r0 ) + 0.5f );
		if ( rows < 2 ) rows = 2;
		if ( cols < 2 ) cols = 2;
		*r_circle = fminf( bw / ( 2.0f * cols ), bh / ( 2.0f * rows ) );
	}
}

static int box_circles( int ci )
{
	float bw = 2 * _Collider2D_hw[ci], bh = 2 * _Collider2D_hh[ci];
	float r0 = fminf( bw, bh ) / 4.0f;
	int rows = (int)floorf( bh / ( 2 * r0 ) + 0.5f ), cols = (int)floorf( bw / ( 2 * r0 ) + 0.5f );
	if ( rows < 2 ) rows = 2;
	if ( cols < 2 ) cols = 2;
	return rows * cols;
}

static int is_static_solid( int ci )
{
	if ( _Collider2D_is_trigger[ci] )
		return 0;
	int rb = _Collider2D_rb2d[ci];
	return rb < 0 || rb >= _Rigidbody2D_count || _Rigidbody2D_body_type[rb] == 2;
}

#if GPU_JOINTS
/* the circle of body b nearest the world point (x, y), skipping `not` (-1: none) */
static int nearest_circle( int body, float x, float y, int not )
{
	int best = -1;
	float bd = 1e30f;
	for ( int k = 0; k < gp_body_circles( W, body ); ++k )
	{
		int c = gp_body_circle( W, body, k );
		if ( c == not )
			continue;
		float dx = gp_circle_x( W, c ) - x, dy = gp_circle_y( W, c ) - y, d = dx * dx + dy * dy;
		if ( d < bd )
			bd = d, best = c;
	}
	return best;
}

static float dist2( float ax, float ay, float bx, float by )
{
	return sqrtf( ( ax - bx ) * ( ax - bx ) + ( ay - by ) * ( ay - by ) );
}

/* Unity's hinge / distance / spring / fixed joints as distance constraints between the circles nearest
 * the anchors (a hinge to the world: two circles at their distances from the anchor; a fixed joint:
 * up to two circle pairs at their current distances). The anchors are read unrotated. */
static void build_joints( void )
{
	float dt = fixed_dt();
	for ( int j = 0; j < _Joint2D_count; ++j )
	{
		int ra = _Joint2D_rb_a[j], rb = _Joint2D_rb_b[j];
		if ( !_Joint2D_enabled[j] || ra < 0 || ra >= MAX_RB || rb_body[ra] < 0 )
			continue;
		float px, py;
		engine_rb2d_get_pos( ra, &px, &py );
		float ax = px + _Joint2D_anchor_x[j], ay = py + _Joint2D_anchor_y[j];
		int other = rb >= 0 && rb < MAX_RB ? rb_body[rb] : -1;
		float wx, wy; /* the connected anchor, world */
		if ( rb >= 0 && rb < MAX_RB )
		{
			float qx, qy;
			engine_rb2d_get_pos( rb, &qx, &qy );
			wx = qx + _Joint2D_canchor_x[j], wy = qy + _Joint2D_canchor_y[j];
		}
		else
			wx = _Joint2D_canchor_x[j], wy = _Joint2D_canchor_y[j];
		if ( _Joint2D_auto_anchor[j] ) /* autoConfigureConnectedAnchor: where the own anchor is now */
			wx = ax, wy = ay;
		int kind = _Joint2D_kind[j];
		int ca = nearest_circle( rb_body[ra], ax, ay, -1 );
		if ( ca < 0 )
			continue;
		float cax = gp_circle_x( W, ca ), cay = gp_circle_y( W, ca );
		if ( kind == 0 ) /* hinge to the world (or a static body) */
		{
			int c2 = gp_body_circles( W, rb_body[ra] ) > 1 ? nearest_circle( rb_body[ra], ax, ay, ca ) : -1;
			gp_add_joint( W, ca, -1, GP_RIGID, dist2( cax, cay, wx, wy ), 1.0f, wx, wy );
			if ( c2 >= 0 )
				gp_add_joint( W, c2, -1, GP_RIGID, dist2( gp_circle_x( W, c2 ), gp_circle_y( W, c2 ), wx, wy ), 1.0f,
							  wx, wy );
			continue;
		}
		if ( kind == 3 ) /* fixed */
		{
			int c2 = gp_body_circles( W, rb_body[ra] ) > 1 ? nearest_circle( rb_body[ra], ax, ay, ca ) : -1;
			for ( int pass = 0; pass < 2; ++pass )
			{
				int c = pass ? c2 : ca;
				if ( c < 0 )
					continue;
				float cx = gp_circle_x( W, c ), cy = gp_circle_y( W, c );
				if ( other < 0 )
					gp_add_joint( W, c, -1, GP_RIGID, 0.0f, 1.0f, cx, cy );
				else
				{
					int cb = nearest_circle( other, cx, cy, -1 );
					gp_add_joint( W, c, cb, GP_RIGID, dist2( cx, cy, gp_circle_x( W, cb ), gp_circle_y( W, cb ) ),
								  1.0f, 0, 0 );
				}
			}
			continue;
		}
		/* distance / spring: the authored length between the anchors, moved to the nearest circles */
		float length = _Joint2D_auto_distance[j] ? dist2( ax, ay, wx, wy ) : _Joint2D_distance[j];
		int kindj = _Joint2D_max_distance_only[j] ? GP_ROPE : GP_RIGID;
		float stiff = 1.0f;
		if ( kind == 2 && _Joint2D_frequency[j] > 0.0f )
		{
			stiff = _Joint2D_frequency[j] * dt * 3.0f; /* a spring: the correction a step, from its hertz */
			stiff = stiff < 0.02f ? 0.02f : ( stiff > 1.0f ? 1.0f : stiff );
		}
		if ( other < 0 )
		{
			float slack = dist2( cax, cay, wx, wy ) - dist2( ax, ay, wx, wy );
			gp_add_joint( W, ca, -1, kindj, length + slack, stiff, wx, wy );
		}
		else
		{
			int cb = nearest_circle( other, wx, wy, -1 );
			float slack = dist2( cax, cay, gp_circle_x( W, cb ), gp_circle_y( W, cb ) ) - dist2( ax, ay, wx, wy );
			gp_add_joint( W, ca, cb, kindj, length + slack, stiff, 0, 0 );
		}
	}
}
#endif

static void build_world( void )
{
	/* the biggest circle sets the tile: a circle of radius <= half a tile is what the grid handles */
	float rmax = 0.0f;
	int circles = 0, joints = 0;
	float minx = 1e30f, miny = 1e30f, maxx = -1e30f, maxy = -1e30f;
	for ( int ci = 0; ci < _Collider2D_count && ci < MAX_COL; ++ci )
	{
		float cx, cy;
		engine_col2d_center( ci, &cx, &cy );
		float c = fabsf( _Collider2D_cos[ci] ), s = fabsf( _Collider2D_sin[ci] );
		float ex = _Collider2D_kind[ci] == 1 ? _Collider2D_hw[ci] : c * _Collider2D_hw[ci] + s * _Collider2D_hh[ci];
		float ey = _Collider2D_kind[ci] == 1 ? _Collider2D_hw[ci] : s * _Collider2D_hw[ci] + c * _Collider2D_hh[ci];
		ex += _Collider2D_edge_r[ci], ey += _Collider2D_edge_r[ci];
		minx = fminf( minx, cx - ex ), maxx = fmaxf( maxx, cx + ex );
		miny = fminf( miny, cy - ey ), maxy = fmaxf( maxy, cy + ey );
		if ( is_static_solid( ci ) || _Collider2D_is_trigger[ci] )
			continue;
		float r;
		body_dims( ci, &r );
		rmax = fmaxf( rmax, r );
		int n = _Collider2D_kind[ci] == 1 ? 1 : box_circles( ci );
		circles += n;
		joints += n > 1 ? 6 * n : 0;
	}
	if ( !( minx < maxx ) )
		minx = miny = -1.0f, maxx = maxy = 1.0f;
	if ( rmax <= 0.0f )
		rmax = 0.25f;
	cell = 2.0f * rmax;
	/* the world box: everything authored, plus room to fall and fly */
	const float margin = 25.0f;
	float bw = maxx - minx + 2 * margin, bh = maxy - miny + 2 * margin;
	while ( ( bw / cell ) * ( bh / cell ) > (float)MAX_CELLS )
		cell *= 1.25f; /* ponytail: a vast world coarsens the tile grid */
	org_x = minx - margin, org_y = miny - margin;
	tiles_w = (int)ceilf( bw / cell ), tiles_h = (int)ceilf( bh / cell );
#if GPU_JOINTS
	joints += 8 * _Joint2D_count;
#endif
	W = gp_create( circles + 1, joints + 1, org_x, org_y, tiles_w * cell, tiles_h * cell, cell, Physics2D_gravity_x,
				   Physics2D_gravity_y );
	tile_owner = calloc( (size_t)tiles_w * tiles_h, sizeof( int ) );
	unsigned* solid = calloc( (size_t)tiles_w * tiles_h, sizeof( unsigned ) );

	/* static colliders -> solid tiles: a tile is solid when the shape reaches any of 5 sample points */
	for ( int ci = 0; ci < _Collider2D_count && ci < MAX_COL; ++ci )
	{
		if ( !is_static_solid( ci ) )
			continue;
		float cx, cy;
		engine_col2d_center( ci, &cx, &cy );
		float c = fabsf( _Collider2D_cos[ci] ), s = fabsf( _Collider2D_sin[ci] );
		float ex = ( _Collider2D_kind[ci] == 1 ? _Collider2D_hw[ci] : c * _Collider2D_hw[ci] + s * _Collider2D_hh[ci] ) + _Collider2D_edge_r[ci];
		float ey = ( _Collider2D_kind[ci] == 1 ? _Collider2D_hw[ci] : s * _Collider2D_hw[ci] + c * _Collider2D_hh[ci] ) + _Collider2D_edge_r[ci];
		int x0 = (int)floorf( ( cx - ex - org_x ) / cell ), x1 = (int)floorf( ( cx + ex - org_x ) / cell );
		int y0 = (int)floorf( ( cy - ey - org_y ) / cell ), y1 = (int)floorf( ( cy + ey - org_y ) / cell );
		if ( x0 < 0 ) x0 = 0;
		if ( y0 < 0 ) y0 = 0;
		if ( x1 >= tiles_w ) x1 = tiles_w - 1;
		if ( y1 >= tiles_h ) y1 = tiles_h - 1;
		static const float sx[5] = { 0.5f, 0.2f, 0.8f, 0.2f, 0.8f }, sy[5] = { 0.5f, 0.2f, 0.2f, 0.8f, 0.8f };
		for ( int ty = y0; ty <= y1; ++ty )
			for ( int tx = x0; tx <= x1; ++tx )
				for ( int k = 0; k < 5; ++k )
					if ( col_distance( ci, org_x + ( tx + sx[k] ) * cell, org_y + ( ty + sy[k] ) * cell ) <= 0.0f )
					{
						solid[(size_t)ty * tiles_w + tx] = 1;
						tile_owner[(size_t)ty * tiles_w + tx] = ci + 1;
						break;
					}
	}
	gp_set_tiles( W, tiles_w, tiles_h, solid );
	/* one surface for the world: the mean of the solid colliders' materials */
	{
		float f = 0, bn = 0;
		int m = 0;
		for ( int ci = 0; ci < _Collider2D_count && ci < MAX_COL; ++ci )
			if ( !_Collider2D_is_trigger[ci] )
				f += _Collider2D_friction[ci], bn += _Collider2D_bounciness[ci], m += 1;
		if ( m )
			gp_set_material( W, f / m, bn / m );
	}
	free( solid );

	/* dynamic bodies */
	float dt = fixed_dt();
	for ( int rb = 0; rb < MAX_RB; ++rb )
		rb_body[rb] = -1, rb_col[rb] = -1, rb_on[rb] = 1;
	memset( body_rb, -1, sizeof body_rb );
	for ( int ci = 0; ci < _Collider2D_count && ci < MAX_COL; ++ci )
	{
		int rb = _Collider2D_rb2d[ci];
		if ( rb < 0 || rb >= _Rigidbody2D_count || rb >= MAX_RB || _Collider2D_is_trigger[ci] ||
			 _Rigidbody2D_body_type[rb] == 2 )
			continue;
		float px, py, cx, cy;
		engine_rb2d_get_pos( rb, &px, &py );
		engine_col2d_center( ci, &cx, &cy );
		rb_offx[rb] = cx - px, rb_offy[rb] = cy - py;
		float vx = _Rigidbody2D_vel_x[rb], vy = _Rigidbody2D_vel_y[rb];
		int b;
		if ( _Collider2D_kind[ci] == 1 )
			b = gp_add_ball( W, cx, cy, _Collider2D_hw[ci], vx, vy, dt );
		else
			b = gp_add_box( W, cx, cy, 2 * _Collider2D_hw[ci], 2 * _Collider2D_hh[ci],
							atan2f( _Collider2D_sin[ci], _Collider2D_cos[ci] ), vx, vy, dt, 1.0f );
		if ( b < 0 || b >= MAX_BODIES )
			continue;
		rb_body[rb] = b, rb_col[rb] = ci, body_rb[b] = rb;
		if ( _Rigidbody2D_body_type[rb] == 1 )
			gp_body_set_kinematic( W, b, 1 );
#if !GPU_ROTATION
		gp_body_lock_rotation( W, b, 1 );
#endif
		last_x[rb] = px, last_y[rb] = py, last_vx[rb] = vx, last_vy[rb] = vy;
	}
#if GPU_JOINTS
	build_joints();
#endif
	ready = 1;
}

/* ---- the step ---- */

static void pull_pose( int rb )
{
	int b = rb_body[rb];
	float cx, cy, ang;
	gp_body_pose( W, b, &cx, &cy, &ang );
	float ox = rb_offx[rb], oy = rb_offy[rb];
	/* the offset turns with the body when rotation is simulated */
#if GPU_ROTATION
	float a0 = atan2f( _Collider2D_sin[rb_col[rb]], _Collider2D_cos[rb_col[rb]] );
	float da = ang - a0, c = cosf( da ), s = sinf( da );
	float rx = c * ox - s * oy, ry = s * ox + c * oy;
	ox = rx, oy = ry;
#endif
	(void)ang;
	float x = cx - ox, y = cy - oy;
	engine_rb2d_set_pos( rb, x, y );
	last_x[rb] = x, last_y[rb] = y;
	float vx, vy;
	gp_body_velocity( W, b, fixed_dt(), &vx, &vy );
	_Rigidbody2D_vel_x[rb] = vx, _Rigidbody2D_vel_y[rb] = vy;
	last_vx[rb] = vx, last_vy[rb] = vy;
#if GPU_ROTATION
	if ( _Collider2D_kind[rb_col[rb]] != 1 ) /* a ball has no orientation */
		engine_rb2d_set_rot( rb, ang );
#endif
}

static int pair_n;
static int pair_a[MAX_PAIRS], pair_b[MAX_PAIRS];

static void add_pair( int x, int y )
{
	if ( x < 0 || y < 0 || x == y )
		return;
	int lo = x < y ? x : y, hi = x < y ? y : x;
	for ( int i = 0; i < pair_n; ++i )
		if ( pair_a[i] == lo && pair_b[i] == hi )
			return;
	if ( pair_n < MAX_PAIRS )
		pair_a[pair_n] = lo, pair_b[pair_n] = hi, pair_n += 1;
}

#if GPU_TRIGGERS
/* a trigger collider overlaps a dynamic body when a circle of the body reaches the shape */
static int trig_a[MAX_PAIRS], trig_b[MAX_PAIRS];

static void trigger_pairs( void )
{
	int n = gp_circle_count( W ), tn = 0;
	for ( int ti = 0; ti < _Collider2D_count && ti < MAX_COL; ++ti )
	{
		if ( !_Collider2D_is_trigger[ti] || !col_live( ti ) )
			continue;
		int own = _Collider2D_rb2d[ti];
		for ( int k = 0; k < n; ++k )
		{
			int rb = body_rb[gp_circle_body( W, k )];
			if ( rb < 0 || rb == own || !rb_on[rb] )
				continue;
			if ( col_distance( ti, gp_circle_x( W, k ), gp_circle_y( W, k ) ) >= gp_circle_radius( W, k ) )
				continue;
			int lo = ti < rb_col[rb] ? ti : rb_col[rb], hi = ti < rb_col[rb] ? rb_col[rb] : ti;
			int dup = 0;
			for ( int i = 0; i < tn; ++i )
				dup |= trig_a[i] == lo && trig_b[i] == hi;
			if ( !dup && tn < MAX_PAIRS )
			{
				trig_a[tn] = lo, trig_b[tn] = hi, tn += 1;
				engine_col2d_trigger( lo, hi );
			}
		}
	}
}
#endif

void engine_box2d_step( void )
{
	if ( !ready )
		build_world();
	float dt = fixed_dt();

	/* push what scripts changed: teleports, velocities, gravityScale, drag, liveness */
	for ( int rb = 0; rb < _Rigidbody2D_count && rb < MAX_RB; ++rb )
	{
		int b = rb_body[rb];
		if ( b < 0 )
			continue;
#if GPU_LIVE
		int on = engine_rb2d_live( rb ) && engine_col2d_live( rb_col[rb] );
		if ( on != rb_on[rb] )
			gp_body_set_active( W, b, on ), rb_on[rb] = on;
		if ( !on )
			continue;
#endif
		float x, y;
		engine_rb2d_get_pos( rb, &x, &y );
		if ( _Rigidbody2D_body_type[rb] == 1 )
		{
			gp_body_move( W, b, x + rb_offx[rb], y + rb_offy[rb] ); /* kinematic: the script drives it */
			continue;
		}
		if ( fabsf( x - last_x[rb] ) + fabsf( y - last_y[rb] ) > 1e-6f )
			gp_body_set_position( W, b, x + rb_offx[rb], y + rb_offy[rb] );
		float vx = _Rigidbody2D_vel_x[rb], vy = _Rigidbody2D_vel_y[rb];
		if ( fabsf( vx - last_vx[rb] ) + fabsf( vy - last_vy[rb] ) > 1e-6f )
			gp_body_set_velocity( W, b, dt, vx, vy );
		float gs = _Rigidbody2D_gravity_scale[rb];
		if ( gs != 1.0f )
			gp_body_add_velocity( W, b, dt, ( gs - 1.0f ) * Physics2D_gravity_x * dt,
								  ( gs - 1.0f ) * Physics2D_gravity_y * dt );
		float d = _Rigidbody2D_linear_damping[rb];
		if ( d > 0.0f )
		{
			float f = 1.0f / ( 1.0f + dt * d ), cvx, cvy;
			gp_body_velocity( W, b, dt, &cvx, &cvy );
			gp_body_add_velocity( W, b, dt, cvx * ( f - 1.0f ), cvy * ( f - 1.0f ) );
		}
	}

	gp_step( W, dt );

	for ( int rb = 0; rb < _Rigidbody2D_count && rb < MAX_RB; ++rb )
		if ( rb_body[rb] >= 0 && rb_on[rb] && _Rigidbody2D_body_type[rb] != 1 )
			pull_pose( rb );

	/* contacts: body-body, and body-tile mapped to the static collider that made the tile */
	pair_n = 0;
	int a[MAX_PAIRS], b[MAX_PAIRS];
	int n = gp_body_pairs( W, a, b, MAX_PAIRS );
	for ( int i = 0; i < n; ++i )
	{
		int ra = body_rb[a[i]], rbb = body_rb[b[i]];
		if ( ra >= 0 && rbb >= 0 )
			add_pair( rb_col[ra], rb_col[rbb] );
	}
	int body[MAX_PAIRS], tile[MAX_PAIRS];
	n = gp_tile_contacts( W, body, tile, MAX_PAIRS );
	for ( int i = 0; i < n; ++i )
	{
		int rb = body_rb[body[i]];
		if ( rb >= 0 && tile[i] >= 0 && tile[i] < tiles_w * tiles_h && tile_owner[tile[i]] > 0 )
			add_pair( rb_col[rb], tile_owner[tile[i]] - 1 );
	}
	for ( int i = 0; i < pair_n; ++i )
		engine_col2d_contact( pair_a[i], pair_b[i] );
#if GPU_TRIGGERS
	trigger_pairs();
#endif
}

/* ---- queries ---- */

static int mask_has( int ci, unsigned int mask )
{
	return ( ( mask >> ( _Collider2D_layer[ci] & 31 ) ) & 1u ) != 0;
}

/* a collider the queries see: live, on a masked layer, and a trigger only when queriesHitTriggers */
static int query_ok( int ci, unsigned int mask )
{
	return ci >= 0 && ( GPU_QUERIES_HIT_TRIGGERS || !_Collider2D_is_trigger[ci] ) && col_live( ci ) &&
		   mask_has( ci, mask );
}

/* static colliders and triggers are tested analytically (a dynamic body's trigger moves with it) */
static int analytic( int ci )
{
	return is_static_solid( ci ) || ( GPU_QUERIES_HIT_TRIGGERS && _Collider2D_is_trigger[ci] );
}

/* ray (o, unit d) against a static collider: smallest t in [0, max], or -1 */
static float ray_static( int ci, float ox, float oy, float dx, float dy, float max, float* nx, float* ny )
{
	float cx, cy;
	engine_col2d_center( ci, &cx, &cy );
	float c = _Collider2D_cos[ci], s = _Collider2D_sin[ci];
	float px = c * ( ox - cx ) + s * ( oy - cy ), py = -s * ( ox - cx ) + c * ( oy - cy );
	float qx = c * dx + s * dy, qy = -s * dx + c * dy;
	float t = -1, lnx = 0, lny = 0;
	if ( _Collider2D_kind[ci] == 1 )
	{
		float r = _Collider2D_hw[ci], b = px * qx + py * qy, cc = px * px + py * py - r * r;
		float disc = b * b - cc;
		if ( disc >= 0 )
		{
			float t0 = -b - sqrtf( disc );
			if ( t0 >= 0 )
				t = t0, lnx = ( px + qx * t0 ) / r, lny = ( py + qy * t0 ) / r;
		}
	}
	else
	{
		float hx = _Collider2D_hw[ci] + _Collider2D_edge_r[ci], hy = _Collider2D_hh[ci] + _Collider2D_edge_r[ci];
		float t0 = 0, t1 = max;
		int axis = -1, sign = 0, hit = 1;
		float lo[2] = { -hx, -hy }, p[2] = { px, py }, q[2] = { qx, qy };
		for ( int k = 0; k < 2 && hit; ++k )
		{
			float h = k ? hy : hx;
			if ( fabsf( q[k] ) < 1e-9f )
			{
				if ( p[k] < -h || p[k] > h )
					hit = 0;
				continue;
			}
			float ta = ( -h - p[k] ) / q[k], tb = ( h - p[k] ) / q[k];
			int sg = -1;
			if ( ta > tb )
			{
				float tmp = ta;
				ta = tb, tb = tmp, sg = 1;
			}
			if ( ta > t0 )
				t0 = ta, axis = k, sign = sg;
			if ( tb < t1 )
				t1 = tb;
			if ( t0 > t1 )
				hit = 0;
		}
		(void)lo;
		if ( hit && axis >= 0 )
			t = t0, lnx = axis == 0 ? (float)sign : 0, lny = axis == 1 ? (float)sign : 0;
	}
	if ( t < 0 || t > max )
		return -1;
	*nx = c * lnx - s * lny;
	*ny = s * lnx + c * lny;
	return t;
}

typedef struct Hit
{
	float x, y, nx, ny, t;
	int collider;
} Hit;

/* nearest solid collider along the ray beyond `from` t, skipping the colliders in skip[] */
static int nearest( float ox, float oy, float dx, float dy, float dist, unsigned int mask, const int* skip, int nskip,
					Hit* h )
{
	int found = 0;
	h->t = dist;
	for ( int ci = 0; ci < _Collider2D_count && ci < MAX_COL; ++ci )
	{
		if ( !analytic( ci ) || !query_ok( ci, mask ) )
			continue;
		int sk = 0;
		for ( int k = 0; k < nskip; ++k )
			sk |= skip[k] == ci;
		if ( sk )
			continue;
		float nx, ny, t = ray_static( ci, ox, oy, dx, dy, h->t, &nx, &ny );
		if ( t >= 0 && t <= h->t )
			h->t = t, h->nx = nx, h->ny = ny, h->collider = ci, found = 1;
	}
	/* dynamic bodies: the circles; layers and skips re-cast past a masked-out or skipped circle */
	float travelled = 0.0f, rx = ox, ry = oy;
	for ( int guard = 0; guard < MAX_QUERY && W; ++guard )
	{
		GpHit g;
		if ( !gp_raycast( W, rx, ry, dx, dy, h->t - travelled, GP_RAY_CIRCLES, -1, &g ) )
			break;
		int rb = g.body >= 0 ? body_rb[g.body] : -1;
		int ci = rb >= 0 ? rb_col[rb] : -1;
		int sk = ci < 0 || !rb_on[rb] || !query_ok( ci, mask );
		for ( int k = 0; k < nskip; ++k )
			sk |= skip[k] == ci;
		if ( !sk )
		{
			float t = travelled + g.distance;
			if ( t <= h->t )
				h->t = t, h->nx = g.nx, h->ny = g.ny, h->collider = ci, found = 1;
			break;
		}
		float step = g.distance + 1e-3f;
		travelled += step, rx += dx * step, ry += dy * step;
		if ( travelled >= h->t )
			break;
	}
	if ( found )
		h->x = ox + dx * h->t, h->y = oy + dy * h->t;
	return found;
}

/* solid colliders containing (x, y) */
static int inside_set( float x, float y, unsigned int mask, int* out, int max )
{
	int n = 0;
	for ( int ci = 0; ci < _Collider2D_count && ci < MAX_COL && n < max; ++ci )
		if ( analytic( ci ) && query_ok( ci, mask ) && col_distance( ci, x, y ) < 0.0f )
			out[n++] = ci;
	if ( W )
	{
		int body;
		int total = gp_circle_count( W );
		for ( int k = 0; k < total && n < max; ++k )
		{
			body = gp_circle_body( W, k );
			int rb = body >= 0 ? body_rb[body] : -1;
			if ( rb < 0 || !rb_on[rb] || !query_ok( rb_col[rb], mask ) )
				continue;
			float dx = x - gp_circle_x( W, k ), dy = y - gp_circle_y( W, k );
			if ( dx * dx + dy * dy < gp_circle_radius( W, k ) * gp_circle_radius( W, k ) )
			{
				int dup = 0;
				for ( int i = 0; i < n; ++i )
					dup |= out[i] == rb_col[rb];
				if ( !dup )
					out[n++] = rb_col[rb];
			}
		}
	}
	return n;
}

static int ray_setup( float dx, float dy, float* distance, float* ux, float* uy )
{
	float len = sqrtf( dx * dx + dy * dy );
	if ( len <= 0.0f )
		return 0;
	if ( !( *distance < 1.0e6f ) )
		*distance = 1.0e6f;
	*ux = dx / len, *uy = dy / len;
	return 1;
}

/* out: point x, y, normal x, y, fraction, distance */
int engine_box2d_raycast( float ox, float oy, float dx, float dy, float distance, unsigned int mask, float* out )
{
	float ux, uy;
	if ( !ray_setup( dx, dy, &distance, &ux, &uy ) )
		return -1;
	int inside[8];
	int n_in = inside_set( ox, oy, mask, inside, 8 );
	if ( n_in > 0 && engine_box2d_queries_start_in_colliders )
	{
		out[0] = ox, out[1] = oy, out[2] = -ux, out[3] = -uy, out[4] = 0.0f, out[5] = 0.0f;
		return inside[0];
	}
	Hit h;
	if ( !nearest( ox, oy, ux, uy, distance, mask, inside, n_in, &h ) )
		return -1;
	out[0] = h.x, out[1] = h.y, out[2] = h.nx, out[3] = h.ny;
	out[4] = distance > 0 ? h.t / distance : 0.0f, out[5] = h.t;
	return h.collider;
}

int engine_box2d_raycast_all( float ox, float oy, float dx, float dy, float distance, unsigned int mask, float* out,
							  int* colliders, int max )
{
	float ux, uy;
	if ( !ray_setup( dx, dy, &distance, &ux, &uy ) )
		return 0;
	int inside[8];
	int n_in = inside_set( ox, oy, mask, inside, 8 );
	int n = 0;
	int seen[MAX_QUERY];
	int nseen = 0;
	for ( int i = 0; engine_box2d_queries_start_in_colliders && i < n_in && n < max; ++i, ++n )
	{
		float* o = out + 6 * n;
		o[0] = ox, o[1] = oy, o[2] = -ux, o[3] = -uy, o[4] = 0.0f, o[5] = 0.0f;
		colliders[n] = inside[i];
		seen[nseen++] = inside[i];
	}
	for ( int i = 0; !engine_box2d_queries_start_in_colliders && i < n_in && nseen < MAX_QUERY; ++i )
		seen[nseen++] = inside[i];
	while ( n < max && nseen < MAX_QUERY )
	{
		Hit h;
		if ( !nearest( ox, oy, ux, uy, distance, mask, seen, nseen, &h ) )
			break;
		float* o = out + 6 * n;
		o[0] = h.x, o[1] = h.y, o[2] = h.nx, o[3] = h.ny;
		o[4] = distance > 0 ? h.t / distance : 0.0f, o[5] = h.t;
		colliders[n++] = h.collider;
		seen[nseen++] = h.collider;
	}
	/* nearest first */
	for ( int i = 1; i < n; ++i )
		for ( int j = i; j > 0 && out[6 * j + 5] < out[6 * ( j - 1 ) + 5]; --j )
		{
			for ( int k = 0; k < 6; ++k )
			{
				float t = out[6 * j + k];
				out[6 * j + k] = out[6 * ( j - 1 ) + k], out[6 * ( j - 1 ) + k] = t;
			}
			int c = colliders[j];
			colliders[j] = colliders[j - 1], colliders[j - 1] = c;
		}
	return n;
}

int engine_box2d_overlap_circle_all( float x, float y, float radius, unsigned int mask, int* colliders, int max )
{
	int n = 0;
	for ( int ci = 0; ci < _Collider2D_count && ci < MAX_COL && n < max; ++ci )
		if ( analytic( ci ) && query_ok( ci, mask ) && col_distance( ci, x, y ) <= radius )
			colliders[n++] = ci;
	int total = W ? gp_circle_count( W ) : 0;
	for ( int k = 0; k < total && n < max; ++k )
	{
		int body = gp_circle_body( W, k );
		int rb = body >= 0 ? body_rb[body] : -1;
		if ( rb < 0 || !rb_on[rb] || !query_ok( rb_col[rb], mask ) )
			continue;
		float dx = x - gp_circle_x( W, k ), dy = y - gp_circle_y( W, k ), r = radius + gp_circle_radius( W, k );
		if ( dx * dx + dy * dy <= r * r )
		{
			int dup = 0;
			for ( int i = 0; i < n; ++i )
				dup |= colliders[i] == rb_col[rb];
			if ( !dup )
				colliders[n++] = rb_col[rb];
		}
	}
	return n;
}

int engine_box2d_overlap_circle( float x, float y, float radius, unsigned int mask )
{
	int c;
	return engine_box2d_overlap_circle_all( x, y, radius, mask, &c, 1 ) > 0 ? c : -1;
}

int engine_box2d_overlap_point( float x, float y, unsigned int mask )
{
	int c;
	return inside_set( x, y, mask, &c, 1 ) > 0 ? c : -1;
}
"""
