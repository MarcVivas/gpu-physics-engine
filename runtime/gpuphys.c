/* gpuphys.c -- see gpuphys.h. The kernels (gpk_*.c) are Crust's C translation of kernels/NAME.rs. */
#include "gpuphys.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- the kernels (translated from Rust) -------------------------------------------------- */
void build_cell_ids_array( float *positions, unsigned int *cell_ids, unsigned int *object_ids, float *radius,
						   float cell_size, unsigned int num_particles );
void count_objects_for_each_chunk( unsigned int *chunk_obj_count, unsigned int *uniform_data, unsigned int *cell_ids );
void build_collision_cells_array( unsigned int *chunk_obj_count, unsigned int *collision_cells,
								  unsigned int *indirect_args, unsigned int *uniform_data, unsigned int *cell_ids );
void solve_collisions_grouped( unsigned int *chunk_obj_count, unsigned int *collision_cells, unsigned int *cell_ids,
							   unsigned int *object_ids, float *positions, float *radius, unsigned int *uniform_data,
							   unsigned int *groups, unsigned int color );
void integrate_tagged( float *positions, float *previous, float *radius, unsigned int *tags, float dt, float gx,
					   float gy, float damping, float sand_damping, float ww, float wh, unsigned int n );
void tile_collide( float *positions, float *radius, unsigned int *tiles, unsigned int tw, unsigned int th,
				   unsigned int n );
void tile_friction( float *positions, float *previous, float *radius, unsigned int *tiles, unsigned int tw,
					unsigned int th, unsigned int n, float mu, float absorb );
void solve_joints( float *positions, float *radius, unsigned int *joint_i, float *joint_f, unsigned int first,
				   unsigned int count );
/* kernels/prowl_material.rs (gpk_prowl_material.c). Weak, so a host that links only the first four translated
 * kernel files still links; gp_body_set_material / gp_body_set_filter then return 0 and the step is the plain one. */
void integrate_material( float *positions, float *previous, float *radius, float *material, float dt,
						 float gx, float gy, float damping, float ww, float wh, unsigned int n ) __attribute__( ( weak ) );
void solve_collisions_filtered( unsigned int *chunk_obj_count, unsigned int *collision_cells, unsigned int *cell_ids,
								unsigned int *object_ids, float *positions, float *radius, unsigned int *uniform_data,
								unsigned int *filter, float *material, float *previous,
								unsigned int color, float bounce_threshold ) __attribute__( ( weak ) );
void tile_material( float *positions, float *previous, float *radius, unsigned int *tiles, float *material,
					float *omega, unsigned int tw, unsigned int th, unsigned int n, float tile_friction,
					float tile_bounciness, float bounce_threshold ) __attribute__( ( weak ) );
/* kernels/prowl_curves.rs (gpk_prowl_curves.c): quadratic Bezier colliders. Weak like the material kernels. */
void curve_collide( float *positions, float *previous, float *radius, float *material, float *curves,
					unsigned int n, unsigned int ncurves ) __attribute__( ( weak ) );
void curve_material( float *positions, float *previous, float *radius, float *material, float *omega, float *curves,
					 unsigned int n, unsigned int ncurves, float bounce_threshold ) __attribute__( ( weak ) );
void raycast( float *positions, float *radius, unsigned int *groups, unsigned int *tiles, float *rays, float *hits,
			  unsigned int *hit_ids, unsigned int tw, unsigned int th, unsigned int ncircles, unsigned int nrays,
			  unsigned int mask, unsigned int ignore_group );

/* The kernels' intrinsics, run sequentially: GP_LAUNCH sets the invocation id and calls. */
static unsigned gp__gid;
unsigned gpu_global_id( int d )
{
	return d == 0 ? gp__gid : 0u;
}
unsigned gpu_atomic_add( unsigned *p, unsigned v )
{
	unsigned old = *p;
	*p = old + v;
	return old;
}
#define GP_LAUNCH( count, CALL )                                                                                       \
	do                                                                                                                 \
	{                                                                                                                  \
		unsigned n_ = (unsigned)( count );                                                                             \
		for ( gp__gid = 0; gp__gid < n_; ++gp__gid )                                                                   \
		{                                                                                                              \
			CALL;                                                                                                      \
		}                                                                                                              \
	}                                                                                                                  \
	while ( 0 )

#define MAX_CELLS 4
#define CELL_SIZE_MULTIPLIER 2.2f
#define FIXED 0xFFFFFFFFu
#define UNUSED 0xFFFFFFFFu

typedef struct Joint
{
	int a, b, flags;
	float rest, k, fx, fy;
} Joint;

typedef struct Body
{
	int *c;
	int nc, cols, rows, active, is_box, locked, is_poly;
	float stiff, lock_th, ox_l, oy_l; /* a polygon: shape-matching stiffness, the locked angle, the circle centroid from the origin (world units, unturned) */
	float w, h; /* world units */
	float *lx, *ly; /* each circle's offset from the centre when unturned, in tile units */
} Body;

typedef struct Contact
{
	int kind;      /* 0 collision, 1 trigger */
	int a, b;      /* bodies, a < b for a collision; b < 0: the tiles; a trigger: a is the sensor */
	int ca, cb;    /* circles last seen touching */
	int tile;
} Contact;

typedef struct JNode
{
	int body;   /* a box body: all its circles move rigidly; < 0 with circ < 0: the world */
	int circ;   /* otherwise the one circle the joint holds (a ball, or a circle of a rope) */
} JNode;

typedef struct BJoint
{
	int type, active, collide;
	JNode a, b;
	float lax, lay, lbx, lby; /* anchors in each body's own frame (tile units; the world's: world position) */
	float ux, uy;             /* the slide axis in A's frame */
	float ref;                /* B's angle minus A's at creation */
	float t0;                 /* the translation at creation: the spring's rest */
	int limit, motor;
	float lo, hi, speed, max_motor;
	float rest, freq, zeta, max_force;
	float tx, ty;             /* a mouse joint's target (tile units) */
	float last_phi, last_t;   /* at the end of the previous step: what the motors measure the step's motion against */
} BJoint;

typedef struct Pair
{
	unsigned key, obj, seq;
} Pair;

struct GpWorld
{
	int cap, n, max_joints, nj, nbodies, body_cap;
	float ox, oy, cell, ww, wh, gx, gy, max_radius;
	float damping, sand_damping, friction, bounce_loss;
	int iterations;
	float *pos, *prev, *rad;
	unsigned *tags, *groups;
	float *mat, *invm, *dens, *drag; /* per circle: material (5 floats: friction, bounciness, gravity scale, damping, inverse mass), inverse mass, density scale, linear drag */
	unsigned *flt;                   /* per circle: category, mask, group, body (4 words) */
	unsigned char *custom;           /* the circle's material was set (else it follows gp_set_material) */
	float *omega;                    /* the spin of each circle (radians per step, 0 for one that does not turn): tile friction's slip */
	float *ang, *prev_ang;           /* a ball's angle and last step's (Verlet, radians): balls spin */
	unsigned char *spin;             /* the circle is a whole ball (it has an angle) */
	unsigned *uflt;                  /* the filter the caller set (flt is what the solver gets: sensors collide with nothing) */
	unsigned char *sensor;           /* the circle belongs to a sensor body: overlaps are reported, nothing is pushed */
	struct Contact *ct;              /* contacts that are touching now (events on) */
	int nct, ct_cap;
	GpEvent *ev;                     /* events not yet polled */
	int nev, ev_cap, ev_dropped, events_on;
	float last_dt;
	BJoint *bj;                      /* joints between bodies (gp_add_body_joint) */
	int nbj, bj_cap, joint_iters;
	float *curves;                   /* Bezier colliders, 16 floats each (kernels/prowl_curves.rs) */
	int ncurves, curve_cap;
	int use_material;                /* a body set a material or a filter: the step uses prowl_material's kernels */
	float bounce_threshold;          /* world units / s: slower approaches do not bounce */
	float tile_bounce;               /* the tiles' bounciness (gp_set_material), 0 by default as in Unity */
	int *owner;
	unsigned char *frozen, *pinned;
	float *frozen_pos, *saved_rad, *pin_pos;
	unsigned *cell_ids, *object_ids, *chunks, *cells, uniform[4];
	Pair *tmp;
	unsigned *tiles;
	int tw, th;
	Joint *joints;
	unsigned *ji;
	float *jf;
	int *batch_first, *batch_count, nbatches;
	int joints_dirty;
	Body *bodies;
	int pairs, last_total, last_ncells;
};

static int cmp_pair( const void *pa, const void *pb )
{
	const Pair *a = pa, *b = pb;
	if ( a->key != b->key )
		return a->key < b->key ? -1 : 1;
	return a->seq < b->seq ? -1 : ( a->seq > b->seq );
}

GpWorld *gp_create( int capacity, int max_joints, float ox, float oy, float width, float height, float cell,
					float gravity_x, float gravity_y )
{
	GpWorld *w = calloc( 1, sizeof *w );
	if ( !w )
		return 0;
	w->cap = capacity;
	w->max_joints = max_joints;
	w->ox = ox;
	w->oy = oy;
	w->cell = cell;
	w->ww = width / cell;
	w->wh = height / cell;
	w->gx = gravity_x / cell;
	w->gy = gravity_y / cell;
	w->damping = 0.999f;
	w->sand_damping = 0.985f;
	w->friction = 0.15f;
	w->bounce_loss = 0.6f;
	w->iterations = 2;
	w->pairs = MAX_CELLS * capacity;
	int chunks = ( w->pairs + 3 ) / 4;
	w->pos = calloc( 2 * capacity, sizeof( float ) );
	w->prev = calloc( 2 * capacity, sizeof( float ) );
	w->rad = calloc( capacity, sizeof( float ) );
	w->tags = calloc( capacity, sizeof( unsigned ) );
	w->groups = calloc( capacity, sizeof( unsigned ) );
	w->owner = calloc( capacity, sizeof( int ) );
	w->mat = calloc( 5 * capacity, sizeof( float ) );
	w->invm = calloc( capacity, sizeof( float ) );
	w->dens = calloc( capacity, sizeof( float ) );
	w->drag = calloc( capacity, sizeof( float ) );
	w->flt = calloc( 4 * capacity, sizeof( unsigned ) );
	w->custom = calloc( capacity, 1 );
	w->ang = calloc( capacity, sizeof( float ) );
	w->omega = calloc( capacity, sizeof( float ) );
	w->prev_ang = calloc( capacity, sizeof( float ) );
	w->spin = calloc( capacity, 1 );
	w->uflt = calloc( 4 * capacity, sizeof( unsigned ) );
	w->sensor = calloc( capacity, 1 );
	w->bounce_threshold = 1.0f;
	w->joint_iters = 4;
	w->frozen = calloc( capacity, 1 );
	w->pinned = calloc( capacity, 1 );
	w->pin_pos = calloc( 2 * capacity, sizeof( float ) );
	w->frozen_pos = calloc( 2 * capacity, sizeof( float ) );
	w->saved_rad = calloc( capacity, sizeof( float ) );
	w->cell_ids = calloc( w->pairs, sizeof( unsigned ) );
	w->object_ids = calloc( w->pairs, sizeof( unsigned ) );
	w->chunks = calloc( chunks + 1, sizeof( unsigned ) );
	w->cells = calloc( w->pairs + 1, sizeof( unsigned ) );
	w->tmp = calloc( w->pairs, sizeof( Pair ) );
	w->joints = calloc( max_joints > 0 ? max_joints : 1, sizeof( Joint ) );
	w->ji = calloc( 4 * ( max_joints > 0 ? max_joints : 1 ), sizeof( unsigned ) );
	w->jf = calloc( 4 * ( max_joints > 0 ? max_joints : 1 ), sizeof( float ) );
	w->batch_first = calloc( max_joints > 0 ? max_joints : 1, sizeof( int ) );
	w->batch_count = calloc( max_joints > 0 ? max_joints : 1, sizeof( int ) );
	w->body_cap = capacity;
	w->bodies = calloc( capacity, sizeof( Body ) );
	w->max_radius = 0.0f;
	return w;
}

void gp_destroy( GpWorld *w )
{
	if ( !w )
		return;
	for ( int i = 0; i < w->nbodies; ++i )
		free( w->bodies[i].c ), free( w->bodies[i].lx ), free( w->bodies[i].ly );
	free( w->pos ), free( w->prev ), free( w->rad ), free( w->tags ), free( w->groups ), free( w->owner );
	free( w->mat ), free( w->invm ), free( w->dens ), free( w->drag ), free( w->flt ), free( w->custom ), free( w->uflt ), free( w->sensor ), free( w->ang ), free( w->omega ), free( w->prev_ang ), free( w->spin ), free( w->ct ), free( w->ev ), free( w->bj );
	free( w->frozen ), free( w->pinned ), free( w->pin_pos ), free( w->frozen_pos ), free( w->saved_rad ), free( w->cell_ids ), free( w->object_ids ), free( w->chunks );
	free( w->curves ), free( w->cells ), free( w->tmp ), free( w->tiles ), free( w->joints ), free( w->ji ), free( w->jf );
	free( w->batch_first ), free( w->batch_count ), free( w->bodies );
	free( w );
}

void gp_set_tiles( GpWorld *w, int width, int height, const unsigned *cells )
{
	free( w->tiles );
	w->tiles = malloc( sizeof( unsigned ) * (size_t)width * height + 4 );
	memcpy( w->tiles, cells, sizeof( unsigned ) * (size_t)width * height );
	w->tw = width;
	w->th = height;
}

/* ---- circles and bodies -------------------------------------------------------------------- */
static int add_circle( GpWorld *w, float x, float y, float r, float vx, float vy, float dt, unsigned tag,
					   unsigned group, int body )
{
	if ( w->n >= w->cap )
		return -1;
	int i = w->n++;
	float c = w->cell;
	float px = ( x - w->ox ) / c, py = ( y - w->oy ) / c, ri = r / c;
	w->pos[2 * i] = px;
	w->pos[2 * i + 1] = py;
	w->prev[2 * i] = px - vx * dt / c;
	w->prev[2 * i + 1] = py - vy * dt / c;
	w->rad[i] = ri;
	w->tags[i] = tag;
	w->groups[i] = group;
	w->owner[i] = body;
	w->mat[5 * i] = w->friction, w->mat[5 * i + 1] = 0.0f, w->mat[5 * i + 2] = 1.0f, w->mat[5 * i + 3] = 1.0f;
	w->mat[5 * i + 4] = 1.0f / ri;
	w->invm[i] = 1.0f / ri, w->dens[i] = 1.0f, w->drag[i] = 0.0f, w->custom[i] = 0;
	w->flt[4 * i] = 1u, w->flt[4 * i + 1] = 0xFFFFFFFFu, w->flt[4 * i + 2] = 0u, w->flt[4 * i + 3] = group;
	memcpy( &w->uflt[4 * i], &w->flt[4 * i], 4 * sizeof( unsigned ) );
	w->sensor[i] = 0;
	w->ang[i] = w->prev_ang[i] = 0.0f, w->spin[i] = 0;
	w->frozen[i] = 0;
	w->pinned[i] = 0;
	if ( ri > w->max_radius )
		w->max_radius = ri;
	return i;
}

static int new_body( GpWorld *w, int ncircles )
{
	if ( w->nbodies >= w->body_cap )
		return -1;
	Body *b = &w->bodies[w->nbodies];
	memset( b, 0, sizeof *b );
	b->c = calloc( ncircles, sizeof( int ) );
	b->active = 1;
	return w->nbodies++;
}

int gp_add_joint( GpWorld *w, int a, int b, int kind, float rest, float stiffness, float fx, float fy )
{
	if ( w->nj >= w->max_joints )
		return -1;
	Joint *j = &w->joints[w->nj];
	j->a = a;
	j->b = b;
	j->flags = kind == GP_ROPE ? 1 : kind == GP_STRUT ? 2 : 0;
	j->rest = rest / w->cell;
	j->k = stiffness;
	j->fx = ( fx - w->ox ) / w->cell;
	j->fy = ( fy - w->oy ) / w->cell;
	w->joints_dirty = 1;
	return w->nj++;
}

int gp_add_ball( GpWorld *w, float x, float y, float radius, float vx, float vy, float dt )
{
	if ( w->n >= w->cap )
		return -1;
	int id = new_body( w, 1 );
	if ( id < 0 )
		return -1;
	Body *b = &w->bodies[id];
	b->nc = 1;
	b->c[0] = add_circle( w, x, y, radius, vx, vy, dt, GP_TAG_BALL, id + 1, id );
	w->spin[b->c[0]] = 1;
	b->w = b->h = 2 * radius;
	return id;
}

/* A free circle, no body: sand. Returns its circle index (not a body id), or -1 when full. */
int gp_add_sand( GpWorld *w, float x, float y, float radius, float vx, float vy, float dt )
{
	return add_circle( w, x, y, radius, vx, vy, dt, GP_TAG_SAND, 0, -1 );
}

/* Pushes every circle within `radius` of (x, y) away from it: `force` is the speed (units / s) it gives at the centre,
 * falling off linearly to the edge. Returns how many circles were pushed. */
int gp_blast( GpWorld *w, float x, float y, float radius, float force )
{
	float c = w->cell, px = ( x - w->ox ) / c, py = ( y - w->oy ) / c, rr = radius / c;
	float dt = w->last_dt > 0 ? w->last_dt : 1.0f / 60;
	int pushed = 0;
	for ( int i = 0; i < w->n; ++i )
	{
		if ( w->frozen[i] || w->pinned[i] )
			continue;
		float dx = w->pos[2 * i] - px, dy = w->pos[2 * i + 1] - py, d = hypotf( dx, dy );
		if ( d >= rr )
			continue;
		if ( d < 1e-4f )
			dx = 0, dy = 1, d = 1;
		float k = force * ( 1.0f - d / rr ) * dt / c / d;
		w->prev[2 * i] -= dx * k;
		w->prev[2 * i + 1] -= dy * k;
		++pushed;
	}
	return pushed;
}

int gp_add_box( GpWorld *w, float x, float y, float bw, float bh, float angle, float vx, float vy, float dt,
				float stiffness )
{
	float r0 = fminf( bw, bh ) / 4.0f;
	int rows = (int)floorf( bh / ( 2 * r0 ) + 0.5f ), cols = (int)floorf( bw / ( 2 * r0 ) + 0.5f );
	if ( rows < 2 )
		rows = 2;
	if ( cols < 2 )
		cols = 2;
	float r = fminf( bw / ( 2.0f * cols ), bh / ( 2.0f * rows ) );
	if ( w->n + rows * cols > w->cap || w->nj + 6 * rows * cols > w->max_joints )
		return -1;
	int id = new_body( w, rows * cols );
	if ( id < 0 )
		return -1;
	Body *b = &w->bodies[id];
	b->nc = rows * cols;
	b->cols = cols;
	b->rows = rows;
	b->is_box = 1;
	b->w = bw;
	b->h = bh;
	b->lx = calloc( rows * cols, sizeof( float ) );
	b->ly = calloc( rows * cols, sizeof( float ) );
	float ca = cosf( angle ), sa = sinf( angle );
	for ( int j = 0; j < rows; ++j )
		for ( int i = 0; i < cols; ++i )
		{
			float lx = -bw / 2 + r + i * ( bw - 2 * r ) / ( cols - 1 );
			float ly = -bh / 2 + r + j * ( bh - 2 * r ) / ( rows - 1 );
			b->lx[j * cols + i] = lx / w->cell;
			b->ly[j * cols + i] = ly / w->cell;
			b->c[j * cols + i] =
				add_circle( w, x + lx * ca - ly * sa, y + lx * sa + ly * ca, r, vx, vy, dt, GP_TAG_CRATE, id + 1, id );
		}
	for ( int a = 0; a < b->nc; ++a )
		for ( int q = a + 1; q < b->nc; ++q )
		{
			int di = abs( a % cols - q % cols ), dj = abs( a / cols - q / cols );
			if ( ( di > dj ? di : dj ) <= 2 )
			{
				float dx = w->pos[2 * b->c[a]] - w->pos[2 * b->c[q]], dy = w->pos[2 * b->c[a] + 1] - w->pos[2 * b->c[q] + 1];
				gp_add_joint( w, b->c[a], b->c[q], GP_RIGID, sqrtf( dx * dx + dy * dy ) * w->cell, stiffness, 0, 0 );
			}
		}
	return id;
}

int gp_add_rope( GpWorld *w, float ax, float ay, float bx, float by, int segments, float radius, int pin_a, int pin_b )
{
	int n = segments + 1;
	if ( w->n + n > w->cap || w->nj + n + 2 > w->max_joints )
		return -1;
	int id = new_body( w, n );
	if ( id < 0 )
		return -1;
	Body *b = &w->bodies[id];
	b->nc = n;
	for ( int k = 0; k < n; ++k )
		b->c[k] = add_circle( w, ax + ( bx - ax ) * k / segments, ay + ( by - ay ) * k / segments, radius, 0, 0, 1,
							  GP_TAG_ROPE, id + 1, id );
	float seg = sqrtf( ( bx - ax ) * ( bx - ax ) + ( by - ay ) * ( by - ay ) ) / segments;
	for ( int k = 0; k < segments; ++k )
		gp_add_joint( w, b->c[k], b->c[k + 1], GP_RIGID, seg, 1.0f, 0, 0 );
	if ( pin_a )
		gp_add_joint( w, b->c[0], -1, GP_RIGID, 0, 1.0f, ax, ay );
	if ( pin_b )
		gp_add_joint( w, b->c[n - 1], -1, GP_RIGID, 0, 1.0f, bx, by );
	return id;
}

int gp_body_count( const GpWorld *w )
{
	return w->nbodies;
}
int gp_body_circles( const GpWorld *w, int body )
{
	return w->bodies[body].nc;
}
int gp_body_circle( const GpWorld *w, int body, int k )
{
	return w->bodies[body].c[k];
}
float gp_circle_x( const GpWorld *w, int i )
{
	return w->ox + w->pos[2 * i] * w->cell;
}
float gp_circle_y( const GpWorld *w, int i )
{
	return w->oy + w->pos[2 * i + 1] * w->cell;
}
float gp_circle_radius( const GpWorld *w, int i )
{
	return w->rad[i] * w->cell;
}
int gp_circle_count( const GpWorld *w )
{
	return w->n;
}
int gp_circle_body( const GpWorld *w, int i )
{
	return w->owner[i];
}

/* a rigid body's angle: a box from its first row, a polygon by fitting its circles to their unturned layout, a ball from its spin */
static float body_angle( const GpWorld *w, const Body *b )
{
	if ( b->is_box )
		return atan2f( w->pos[2 * b->c[1] + 1] - w->pos[2 * b->c[0] + 1], w->pos[2 * b->c[1]] - w->pos[2 * b->c[0]] );
	if ( b->is_poly && b->nc > 1 )
	{
		float cx = 0, cy = 0, sd = 0, sc = 0;
		for ( int k = 0; k < b->nc; ++k )
			cx += w->pos[2 * b->c[k]], cy += w->pos[2 * b->c[k] + 1];
		cx /= b->nc, cy /= b->nc;
		for ( int k = 0; k < b->nc; ++k )
		{
			float qx = w->pos[2 * b->c[k]] - cx, qy = w->pos[2 * b->c[k] + 1] - cy;
			sd += b->lx[k] * qx + b->ly[k] * qy;
			sc += b->lx[k] * qy - b->ly[k] * qx;
		}
		return atan2f( sc, sd );
	}
	if ( b->nc == 1 && w->spin[b->c[0]] )
		return w->ang[b->c[0]];
	return 0.0f;
}

void gp_body_pose( const GpWorld *w, int body, float *x, float *y, float *angle )
{
	const Body *b = &w->bodies[body];
	float sx = 0, sy = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		sx += w->pos[2 * b->c[k]];
		sy += w->pos[2 * b->c[k] + 1];
	}
	*x = w->ox + sx / b->nc * w->cell;
	*y = w->oy + sy / b->nc * w->cell;
	*angle = body_angle( w, b );
}

void gp_body_velocity( const GpWorld *w, int body, float dt, float *vx, float *vy )
{
	const Body *b = &w->bodies[body];
	float sx = 0, sy = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		sx += w->pos[2 * c] - w->prev[2 * c];
		sy += w->pos[2 * c + 1] - w->prev[2 * c + 1];
	}
	*vx = sx / b->nc * w->cell / dt;
	*vy = sy / b->nc * w->cell / dt;
}

void gp_body_add_velocity( GpWorld *w, int body, float dt, float dvx, float dvy )
{
	const Body *b = &w->bodies[body];
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->prev[2 * c] -= dvx * dt / w->cell;
		w->prev[2 * c + 1] -= dvy * dt / w->cell;
	}
}

void gp_body_set_velocity( GpWorld *w, int body, float dt, float vx, float vy )
{
	float cx, cy;
	gp_body_velocity( w, body, dt, &cx, &cy );
	gp_body_add_velocity( w, body, dt, vx - cx, vy - cy );
}

void gp_body_set_position( GpWorld *w, int body, float x, float y )
{
	float cx, cy, a;
	gp_body_pose( w, body, &cx, &cy, &a );
	Body *b = &w->bodies[body];
	float dx = ( x - cx ) / w->cell, dy = ( y - cy ) / w->cell;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->pos[2 * c] += dx, w->pos[2 * c + 1] += dy;
		w->prev[2 * c] += dx, w->prev[2 * c + 1] += dy;
	}
}

void gp_body_lock_rotation( GpWorld *w, int body, int lock )
{
	Body *b = &w->bodies[body];
	b->locked = lock != 0 && ( b->is_box || b->is_poly );
	if ( b->locked && b->is_poly )
		b->lock_th = body_angle( w, b );
}

void gp_body_set_active( GpWorld *w, int body, int active )
{
	Body *b = &w->bodies[body];
	if ( ( b->active != 0 ) == ( active != 0 ) )
		return;
	b->active = active != 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->frozen[c] = !active;
		if ( !active )
		{ /* out of the simulation: a speck that stays where it was */
			w->frozen_pos[2 * c] = w->pos[2 * c];
			w->frozen_pos[2 * c + 1] = w->pos[2 * c + 1];
			w->saved_rad[c] = w->rad[c];
			w->rad[c] = 1.0e-4f;
		}
		else
		{
			w->rad[c] = w->saved_rad[c];
		}
	}
}

/* ---- the step ------------------------------------------------------------------------------ */
static void upload_joints( GpWorld *w )
{
	/* greedy colouring: no circle twice in a batch */
	int *colour = malloc( sizeof( int ) * ( w->nj + 1 ) );
	unsigned long long *used = calloc( w->n + 1, sizeof( unsigned long long ) );
	int maxc = 0;
	for ( int j = 0; j < w->nj; ++j )
	{
		unsigned long long taken = used[w->joints[j].a];
		if ( w->joints[j].b >= 0 )
			taken |= used[w->joints[j].b];
		int c = 0;
		while ( c < 63 && ( taken >> c & 1ull ) )
			++c;
		colour[j] = c;
		used[w->joints[j].a] |= 1ull << c;
		if ( w->joints[j].b >= 0 )
			used[w->joints[j].b] |= 1ull << c;
		if ( c > maxc )
			maxc = c;
	}
	int k = 0;
	w->nbatches = 0;
	for ( int c = 0; c <= maxc; ++c )
	{
		int first = k;
		for ( int j = 0; j < w->nj; ++j )
			if ( colour[j] == c )
			{
				Joint *q = &w->joints[j];
				w->ji[4 * k] = q->a;
				w->ji[4 * k + 1] = q->b < 0 ? FIXED : (unsigned)q->b;
				w->ji[4 * k + 2] = q->flags;
				w->ji[4 * k + 3] = 0;
				w->jf[4 * k] = q->rest;
				w->jf[4 * k + 1] = q->k;
				w->jf[4 * k + 2] = q->fx;
				w->jf[4 * k + 3] = q->fy;
				++k;
			}
		if ( k > first )
		{
			w->batch_first[w->nbatches] = first;
			w->batch_count[w->nbatches++] = k - first;
		}
	}
	free( colour );
	free( used );
	w->joints_dirty = 0;
}

static void build_cells( GpWorld *w )
{
	int n = w->n, pairs = MAX_CELLS * n;
	float cs = w->max_radius * CELL_SIZE_MULTIPLIER;
	GP_LAUNCH( n, build_cell_ids_array( w->pos, w->cell_ids, w->object_ids, w->rad, cs, n ) );
	for ( int i = 0; i < pairs; ++i )
	{
		w->tmp[i].key = w->cell_ids[i];
		w->tmp[i].obj = w->object_ids[i];
		w->tmp[i].seq = i;
	}
	qsort( w->tmp, pairs, sizeof( Pair ), cmp_pair ); /* the GPU's radix sort, stable */
	for ( int i = 0; i < pairs; ++i )
	{
		w->cell_ids[i] = w->tmp[i].key;
		w->object_ids[i] = w->tmp[i].obj;
	}
	int chunks = ( pairs + 3 ) / 4;
	w->uniform[0] = chunks;
	w->uniform[1] = pairs;
	GP_LAUNCH( chunks, count_objects_for_each_chunk( w->chunks, w->uniform, w->cell_ids ) );
	for ( int i = 1; i < chunks; ++i ) /* the prefix sum (workgroup memory on the GPU) */
		w->chunks[i] += w->chunks[i - 1];
	unsigned indirect[3];
	GP_LAUNCH( chunks, build_collision_cells_array( w->chunks, w->cells, indirect, w->uniform, w->cell_ids ) );
	w->last_total = pairs;
	w->last_ncells = (int)w->chunks[chunks - 1];
}

/* A kinematic body's circles are held where the caller put them, after every solver pass, so
 * what touches them is pushed out instead of sharing the correction */
static void restore_pinned( GpWorld *w )
{
	for ( int i = 0; i < w->n; ++i )
		if ( w->pinned[i] )
		{
			w->pos[2 * i] = w->prev[2 * i] = w->pin_pos[2 * i];
			w->pos[2 * i + 1] = w->prev[2 * i + 1] = w->pin_pos[2 * i + 1];
		}
}

void gp_body_set_kinematic( GpWorld *w, int body, int on )
{
	Body *b = &w->bodies[body];
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->pinned[c] = on != 0;
		w->pin_pos[2 * c] = w->pos[2 * c], w->pin_pos[2 * c + 1] = w->pos[2 * c + 1];
	}
}

void gp_body_move( GpWorld *w, int body, float x, float y )
{
	Body *b = &w->bodies[body];
	float cx = 0, cy = 0;
	for ( int k = 0; k < b->nc; ++k )
		cx += w->pin_pos[2 * b->c[k]], cy += w->pin_pos[2 * b->c[k] + 1];
	float dx = ( x - w->ox ) / w->cell - cx / b->nc, dy = ( y - w->oy ) / w->cell - cy / b->nc;
	for ( int k = 0; k < b->nc; ++k )
		w->pin_pos[2 * b->c[k]] += dx, w->pin_pos[2 * b->c[k] + 1] += dy;
}

/* Box2D's per-body settings. friction is in Unity's units (0.4 default), bounciness 0..1, density scales
 * the mass (this engine's mass is proportional to a circle's radius: 1 is the plain engine's), gravity_scale
 * multiplies gravity, drag is Unity's linear drag (velocity loses drag * dt of itself a second). Returns 1,
 * or 0 when the host linked no prowl_material kernels. */
int gp_body_set_material( GpWorld *w, int body, float friction, float bounciness, float density, float gravity_scale,
						  float drag )
{
	if ( !integrate_material || !solve_collisions_filtered || !tile_material )
		return 0;
	Body *b = &w->bodies[body];
	w->use_material = 1;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->custom[c] = 1;
		w->mat[5 * c] = 0.375f * ( friction < 0 ? 0 : friction );
		w->mat[5 * c + 1] = bounciness < 0 ? 0 : ( bounciness > 1 ? 1 : bounciness );
		w->mat[5 * c + 2] = gravity_scale;
		w->dens[c] = density > 0 ? density : 1.0f;
		w->drag[c] = drag < 0 ? 0 : drag;
	}
	return 1;
}

/* A static quadratic Bezier collider p0 -> p1 (control) -> p2, in world units. side 0 is a thin wall (solid on
 * both faces: a circle stays on the face it came from), +1 / -1 one edge of a solid contour whose inside is on
 * the left / right of the direction p0 -> p2. friction is Unity's (0.4 default), bounciness 0..1; a contact
 * uses the geometric mean of the two frictions and the larger bounciness. Returns the curve's index, or -1
 * when the host linked no prowl_curves / prowl_material kernels. Sensor bodies are not filtered from curves. */
int gp_add_curve( GpWorld *w, float x0, float y0, float x1, float y1, float x2, float y2, float friction,
				  float bounciness, int side )
{
	if ( !curve_collide || !curve_material || !integrate_material || !solve_collisions_filtered || !tile_material )
		return -1;
	if ( w->ncurves == w->curve_cap )
	{
		w->curve_cap = w->curve_cap ? 2 * w->curve_cap : 16;
		w->curves = realloc( w->curves, sizeof( float ) * 16 * w->curve_cap );
	}
	float *c = &w->curves[16 * w->ncurves];
	x0 = ( x0 - w->ox ) / w->cell, y0 = ( y0 - w->oy ) / w->cell;
	x1 = ( x1 - w->ox ) / w->cell, y1 = ( y1 - w->oy ) / w->cell;
	x2 = ( x2 - w->ox ) / w->cell, y2 = ( y2 - w->oy ) / w->cell;
	c[0] = x0, c[1] = y0, c[2] = x1, c[3] = y1, c[4] = x2, c[5] = y2;
	/* the box of the control polygon holds the curve; the kernel grows it by the circle's radius */
	c[6] = fminf( x0, fminf( x1, x2 ) ), c[7] = fminf( y0, fminf( y1, y2 ) );
	c[8] = fmaxf( x0, fmaxf( x1, x2 ) ), c[9] = fmaxf( y0, fmaxf( y1, y2 ) );
	c[10] = 0.375f * ( friction < 0 ? 0 : friction );
	c[11] = bounciness < 0 ? 0 : ( bounciness > 1 ? 1 : bounciness );
	c[12] = side > 0 ? 1.0f : ( side < 0 ? -1.0f : 0.0f );
	c[13] = c[14] = c[15] = 0.0f;
	w->use_material = 1;
	return w->ncurves++;
}

void gp_clear_curves( GpWorld *w )
{
	w->ncurves = 0;
}

int gp_curve_count( const GpWorld *w )
{
	return w->ncurves;
}

/* Box2D's filter: this body is in the categories `category` (bits) and collides with those in `mask`;
 * a shared non-zero `group` collides when positive and never when negative. */
int gp_body_set_filter( GpWorld *w, int body, unsigned category, unsigned mask, int group )
{
	if ( !integrate_material || !solve_collisions_filtered || !tile_material )
		return 0;
	Body *b = &w->bodies[body];
	w->use_material = 1;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		w->uflt[4 * c] = category;
		w->uflt[4 * c + 1] = mask;
		w->uflt[4 * c + 2] = (unsigned)group;
	}
	return 1;
}

void gp_set_bounce_threshold( GpWorld *w, float speed )
{
	w->bounce_threshold = speed < 0 ? 0 : speed;
}

void gp_set_material( GpWorld *w, float friction, float bounciness )
{
	w->friction = 0.375f * friction; /* the kernel's mu: 0.15 is Unity's default friction 0.4 */
	w->bounce_loss = 1.0f - ( bounciness < 0 ? 0 : ( bounciness > 1 ? 1 : bounciness ) );
	w->tile_bounce = 1.0f - w->bounce_loss;
}


/* ---- events: contacts begin and end, sensors are entered and left --------------------------- */
#define EVENT_BEGIN_SLOP 0.01f /* touching: closer than the radii's sum plus this (tile units) */
#define EVENT_END_SLOP 0.05f   /* ...and apart again only beyond this: no flicker for a resting contact */

static int can_collide_c( unsigned ca, unsigned ma, int ga, unsigned cb, unsigned mb, int gb )
{
	if ( ga != 0 && ga == gb )
		return ga > 0;
	return ( ma & cb ) != 0 && ( mb & ca ) != 0;
}

/* may circles i and j of different bodies touch at all (the caller's filter, not the sensor override) */
static int filter_allows( const GpWorld *w, int i, int j )
{
	const unsigned *f = &w->uflt[4 * i], *g = &w->uflt[4 * j];
	return can_collide_c( f[0], f[1], (int)f[2], g[0], g[1], (int)g[2] );
}

static void push_event( GpWorld *w, int kind, int a, int b, int tile, float x, float y, float nx, float ny, float speed )
{
	if ( w->nev >= w->ev_cap )
	{
		int cap = w->ev_cap ? 2 * w->ev_cap : 64;
		if ( cap > 65536 )
		{
			w->ev_dropped++;
			return;
		}
		w->ev = realloc( w->ev, cap * sizeof( GpEvent ) );
		w->ev_cap = cap;
	}
	GpEvent *e = &w->ev[w->nev++];
	e->kind = kind, e->a = a, e->b = b, e->tile = tile;
	e->x = x, e->y = y, e->nx = nx, e->ny = ny, e->speed = speed;
}

/* the contact point (world units), the unit normal from circle i toward j, and the closing speed (world units/s) */
static void circle_contact( const GpWorld *w, int i, int j, float *x, float *y, float *nx, float *ny, float *speed )
{
	float dx = w->pos[2 * j] - w->pos[2 * i], dy = w->pos[2 * j + 1] - w->pos[2 * i + 1];
	float d = sqrtf( dx * dx + dy * dy );
	if ( d < 1.0e-6f )
		dx = 0, dy = 1, d = 1;
	*nx = dx / d, *ny = dy / d;
	float t = w->rad[i] / ( w->rad[i] + w->rad[j] ); /* where between the centres the surfaces meet */
	*x = w->ox + ( w->pos[2 * i] + dx * t ) * w->cell;
	*y = w->oy + ( w->pos[2 * i + 1] + dy * t ) * w->cell;
	float rvx = ( w->pos[2 * i] - w->prev[2 * i] ) - ( w->pos[2 * j] - w->prev[2 * j] );
	float rvy = ( w->pos[2 * i + 1] - w->prev[2 * i + 1] ) - ( w->pos[2 * j + 1] - w->prev[2 * j + 1] );
	float closing = rvx * *nx + rvy * *ny;
	*speed = w->last_dt > 0 ? ( closing > 0 ? closing : 0 ) * w->cell / w->last_dt : 0;
}

static int circles_touch( const GpWorld *w, int i, int j, float slop )
{
	float dx = w->pos[2 * i] - w->pos[2 * j], dy = w->pos[2 * i + 1] - w->pos[2 * j + 1];
	float r = w->rad[i] + w->rad[j] + slop;
	return dx * dx + dy * dy < r * r;
}

/* a tile touching circle c: its index, or -1 */
static int circle_tile( const GpWorld *w, int c, float slop, float *qx_out, float *qy_out )
{
	if ( !w->tw || !w->th )
		return -1;
	float x = w->pos[2 * c], y = w->pos[2 * c + 1], r = w->rad[c] + slop;
	for ( int ty = (int)floorf( y - r ); ty <= (int)floorf( y + r ); ++ty )
		for ( int tx = (int)floorf( x - r ); tx <= (int)floorf( x + r ); ++tx )
		{
			if ( tx < 0 || ty < 0 || tx >= w->tw || ty >= w->th || !w->tiles[ty * w->tw + tx] )
				continue;
			float qx = fminf( fmaxf( x, tx ), tx + 1.0f ), qy = fminf( fmaxf( y, ty ), ty + 1.0f );
			float dx = x - qx, dy = y - qy;
			if ( dx * dx + dy * dy < r * r )
			{
				if ( qx_out )
					*qx_out = qx, *qy_out = qy;
				return ty * w->tw + tx;
			}
		}
	return -1;
}

/* do bodies a and b touch (b < 0: the tiles)? Fills the circles that do. */
static int bodies_touch( const GpWorld *w, Contact *c, float slop )
{
	const Body *ba = &w->bodies[c->a];
	if ( !ba->active )
		return 0;
	if ( c->b < 0 )
	{
		for ( int k = 0; k < ba->nc; ++k )
		{
			int t = circle_tile( w, ba->c[k], slop, 0, 0 );
			if ( t >= 0 )
			{
				c->ca = ba->c[k], c->tile = t;
				return 1;
			}
		}
		return 0;
	}
	const Body *bb = &w->bodies[c->b];
	if ( !bb->active )
		return 0;
	for ( int k = 0; k < ba->nc; ++k )
		for ( int m = 0; m < bb->nc; ++m )
		{
			int i = ba->c[k], j = bb->c[m];
			if ( ( c->kind == 1 ) != ( w->sensor[i] || w->sensor[j] ) )
				continue;
			if ( circles_touch( w, i, j, slop ) )
			{
				c->ca = i, c->cb = j;
				return 1;
			}
		}
	return 0;
}

static int find_contact( const GpWorld *w, int kind, int a, int b )
{
	for ( int k = 0; k < w->nct; ++k )
		if ( w->ct[k].kind == kind && w->ct[k].a == a && w->ct[k].b == b )
			return k;
	return -1;
}

static void begin_contact( GpWorld *w, int kind, int a, int b, int ca, int cb, int tile )
{
	if ( w->nct >= w->ct_cap )
	{
		w->ct_cap = w->ct_cap ? 2 * w->ct_cap : 64;
		w->ct = realloc( w->ct, w->ct_cap * sizeof( Contact ) );
	}
	Contact *c = &w->ct[w->nct++];
	c->kind = kind, c->a = a, c->b = b, c->ca = ca, c->cb = cb, c->tile = tile;
	float x, y, nx, ny, speed;
	if ( b < 0 )
	{
		float qx = 0, qy = 0;
		circle_tile( w, ca, EVENT_BEGIN_SLOP, &qx, &qy );
		float dx = qx - w->pos[2 * ca], dy = qy - w->pos[2 * ca + 1], d = sqrtf( dx * dx + dy * dy );
		if ( d < 1.0e-6f )
			dx = 0, dy = -1, d = 1;
		nx = dx / d, ny = dy / d;
		x = w->ox + qx * w->cell, y = w->oy + qy * w->cell;
		float vn = ( ( w->pos[2 * ca] - w->prev[2 * ca] ) * nx + ( w->pos[2 * ca + 1] - w->prev[2 * ca + 1] ) * ny );
		speed = w->last_dt > 0 ? ( vn > 0 ? vn : 0 ) * w->cell / w->last_dt : 0;
		push_event( w, GP_EVENT_BEGIN, a, -1, tile, x, y, nx, ny, speed );
		return;
	}
	circle_contact( w, ca, cb, &x, &y, &nx, &ny, &speed );
	push_event( w, kind ? GP_EVENT_TRIGGER_ENTER : GP_EVENT_BEGIN, a, b, -1, x, y, nx, ny, speed );
}

static void update_events( GpWorld *w )
{
	/* contacts that ended (events are produced in the order: ends, then begins) */
	int keep = 0;
	for ( int k = 0; k < w->nct; ++k )
	{
		Contact c = w->ct[k];
		float slop = c.kind ? 0.0f : EVENT_END_SLOP;
		if ( bodies_touch( w, &c, slop ) )
		{
			w->ct[keep++] = c;
			continue;
		}
		push_event( w, c.kind ? GP_EVENT_TRIGGER_EXIT : GP_EVENT_END, c.a, c.b, c.b < 0 ? c.tile : -1, 0, 0, 0, 0, 0 );
	}
	w->nct = keep;

	/* contacts that began: circles that share a collision cell, then circles against the tiles */
	build_cells( w );
	for ( int cell = 0; cell < w->last_ncells; ++cell )
	{
		unsigned start = w->cells[cell], hash = w->cell_ids[start];
		for ( unsigned i = start; i < (unsigned)w->last_total && w->cell_ids[i] == hash; ++i )
			for ( unsigned j = i + 1; j < (unsigned)w->last_total && w->cell_ids[j] == hash; ++j )
			{
				int ci = (int)w->object_ids[i], cj = (int)w->object_ids[j];
				int oa = w->owner[ci], ob = w->owner[cj];
				if ( oa < 0 || ob < 0 || oa == ob || w->frozen[ci] || w->frozen[cj] || !filter_allows( w, ci, cj ) )
					continue;
				int si = w->sensor[ci], sj = w->sensor[cj];
				if ( si && sj )
					continue; /* sensors do not detect each other */
				int kind = si || sj;
				if ( !circles_touch( w, ci, cj, kind ? 0.0f : EVENT_BEGIN_SLOP ) )
					continue;
				int a = oa, b = ob, ca = ci, cb = cj;
				if ( kind ? sj : oa > ob ) /* a trigger: the sensor first; a collision: the lower body first */
					a = ob, b = oa, ca = cj, cb = ci;
				if ( find_contact( w, kind, a, b ) < 0 )
					begin_contact( w, kind, a, b, ca, cb, -1 );
			}
	}
	for ( int i = 0; i < w->n && w->tw; ++i )
	{
		if ( w->frozen[i] || w->sensor[i] )
			continue;
		int body = w->owner[i];
		if ( body >= 0 && circle_tile( w, i, EVENT_BEGIN_SLOP, 0, 0 ) >= 0 && find_contact( w, 0, body, -1 ) < 0 )
			begin_contact( w, 0, body, -1, i, -1, circle_tile( w, i, EVENT_BEGIN_SLOP, 0, 0 ) );
	}
}

void gp_enable_events( GpWorld *w, int on )
{
	w->events_on = on != 0;
	if ( !on )
		w->nct = w->nev = 0;
}

int gp_poll_events( GpWorld *w, GpEvent *out, int max )
{
	int n = w->nev < max ? w->nev : max;
	if ( n > 0 )
	{
		memcpy( out, w->ev, n * sizeof( GpEvent ) );
		memmove( w->ev, w->ev + n, ( w->nev - n ) * sizeof( GpEvent ) );
		w->nev -= n;
	}
	return n;
}

int gp_body_set_sensor( GpWorld *w, int body, int on )
{
	if ( !integrate_material || !solve_collisions_filtered || !tile_material )
		return 0;
	Body *b = &w->bodies[body];
	w->use_material = 1;
	for ( int k = 0; k < b->nc; ++k )
		w->sensor[b->c[k]] = on != 0;
	return 1;
}


/* ---- joints between bodies (Box2D's revolute, weld, prismatic, wheel, distance, mouse) -------- */
/* A body joint is solved on the CPU in position form (PBD, Mueller's "rigid body" constraints): a box is one rigid
 * body (all its circles move together, with a mass and an inertia from its circles); a ball, or one circle of a rope,
 * is a point mass that cannot turn. The solver moves positions only, so the velocities that follow are the joint's. */
static float circle_invm( const GpWorld *w, int c )
{
	return ( w->pinned[c] || w->frozen[c] ) ? 0.0f : 1.0f / ( w->rad[c] * w->dens[c] );
}

typedef struct Frame
{
	float cxA, cyA, thA, imA, iiA, cxB, cyB, thB, imB, iiB;
	float rAx, rAy, rBx, rBy; /* the anchors, from each centre (world orientation) */
	float ex, ey;             /* anchor B minus anchor A */
} Frame;

static void node_state( const GpWorld *w, JNode n, float *cx, float *cy, float *th, float *im, float *ii )
{
	*cx = *cy = *th = *im = *ii = 0.0f;
	if ( n.body < 0 )
	{
		if ( n.circ >= 0 )
		{
			*cx = w->pos[2 * n.circ], *cy = w->pos[2 * n.circ + 1], *im = circle_invm( w, n.circ );
			if ( w->spin[n.circ] ) /* a ball turns: a disc's inertia, half m r^2 */
				*th = w->ang[n.circ], *ii = 2.0f * *im / ( w->rad[n.circ] * w->rad[n.circ] );
		}
		return;
	}
	const Body *b = &w->bodies[n.body];
	float sx = 0, sy = 0, mass = 0;
	int movable = b->active;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		sx += w->pos[2 * c], sy += w->pos[2 * c + 1];
		float inv = circle_invm( w, c );
		if ( inv <= 0.0f )
			movable = 0;
		else
			mass += 1.0f / inv;
	}
	*cx = sx / b->nc, *cy = sy / b->nc;
	if ( b->is_box || b->is_poly )
		*th = body_angle( w, b );
	if ( !movable || mass <= 0.0f )
		return;
	*im = 1.0f / mass;
	float inertia = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float dx = w->pos[2 * c] - *cx, dy = w->pos[2 * c + 1] - *cy;
		inertia += ( 1.0f / circle_invm( w, c ) ) * ( dx * dx + dy * dy );
	}
	*ii = ( ( b->is_box || b->is_poly ) && b->nc > 1 && inertia > 1.0e-9f ) ? 1.0f / inertia : 0.0f;
}

static void node_apply( GpWorld *w, JNode n, float dcx, float dcy, float dth )
{
	if ( n.body < 0 )
	{
		if ( n.circ >= 0 && circle_invm( w, n.circ ) > 0.0f )
		{
			w->pos[2 * n.circ] += dcx, w->pos[2 * n.circ + 1] += dcy;
			if ( w->spin[n.circ] )
				w->ang[n.circ] += dth;
		}
		return;
	}
	const Body *b = &w->bodies[n.body];
	float cx = 0, cy = 0;
	for ( int k = 0; k < b->nc; ++k )
		cx += w->pos[2 * b->c[k]], cy += w->pos[2 * b->c[k] + 1];
	cx /= b->nc, cy /= b->nc;
	float ca = cosf( dth ), sa = sinf( dth );
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float dx = w->pos[2 * c] - cx, dy = w->pos[2 * c + 1] - cy;
		w->pos[2 * c] = cx + dcx + dx * ca - dy * sa;
		w->pos[2 * c + 1] = cy + dcy + dx * sa + dy * ca;
	}
}

static void node_add_velocity( GpWorld *w, JNode n, float dvx, float dvy )
{
	if ( n.body < 0 )
	{
		if ( n.circ >= 0 )
			w->prev[2 * n.circ] -= dvx, w->prev[2 * n.circ + 1] -= dvy;
		return;
	}
	const Body *b = &w->bodies[n.body];
	for ( int k = 0; k < b->nc; ++k )
		w->prev[2 * b->c[k]] -= dvx, w->prev[2 * b->c[k] + 1] -= dvy;
}

static float wrap_angle( float a )
{
	const float pi = 3.14159265f;
	while ( a > pi )
		a -= 2 * pi;
	while ( a < -pi )
		a += 2 * pi;
	return a;
}

static void joint_frame( const GpWorld *w, const BJoint *j, Frame *f )
{
	node_state( w, j->a, &f->cxA, &f->cyA, &f->thA, &f->imA, &f->iiA );
	node_state( w, j->b, &f->cxB, &f->cyB, &f->thB, &f->imB, &f->iiB );
	float ca = cosf( f->thA ), sa = sinf( f->thA ), cb = cosf( f->thB ), sb = sinf( f->thB );
	f->rAx = ca * j->lax - sa * j->lay, f->rAy = sa * j->lax + ca * j->lay;
	f->rBx = cb * j->lbx - sb * j->lby, f->rBy = sb * j->lbx + cb * j->lby;
	f->ex = ( f->cxB + f->rBx ) - ( f->cxA + f->rAx );
	f->ey = ( f->cyB + f->rBy ) - ( f->cyA + f->rAy );
}

/* Move the anchors along n so that C = e . n shrinks: `alpha` is the compliance (0: rigid). Returns the correction. */
static float apply_linear( GpWorld *w, const BJoint *j, const Frame *f, float nx, float ny, float C, float alpha, float max_lam )
{
	float crA = f->rAx * ny - f->rAy * nx, crB = f->rBx * ny - f->rBy * nx;
	float wA = f->imA + f->iiA * crA * crA, wB = f->imB + f->iiB * crB * crB;
	float den = wA + wB + alpha;
	if ( den <= 1.0e-12f )
		return 0.0f;
	float lam = C / den;
	if ( max_lam > 0.0f )
		lam = lam > max_lam ? max_lam : ( lam < -max_lam ? -max_lam : lam );
	if ( wA > 0 )
		node_apply( w, j->a, f->imA * lam * nx, f->imA * lam * ny, f->iiA * lam * crA );
	if ( wB > 0 )
		node_apply( w, j->b, -f->imB * lam * nx, -f->imB * lam * ny, -f->iiB * lam * crB );
	return lam;
}

/* Turn A one way and B the other so that C (an angle error) shrinks. */
static void apply_angular( GpWorld *w, const BJoint *j, const Frame *f, float C, float max_lam )
{
	float den = f->iiA + f->iiB;
	if ( den <= 1.0e-12f )
		return;
	float lam = C / den;
	if ( max_lam > 0.0f )
		lam = lam > max_lam ? max_lam : ( lam < -max_lam ? -max_lam : lam );
	if ( f->iiA > 0 )
		node_apply( w, j->a, 0, 0, f->iiA * lam );
	if ( f->iiB > 0 )
		node_apply( w, j->b, 0, 0, -f->iiB * lam );
}

static void point_constraint( GpWorld *w, const BJoint *j )
{
	Frame f;
	joint_frame( w, j, &f );
	float d = sqrtf( f.ex * f.ex + f.ey * f.ey );
	if ( d > 1.0e-7f )
		apply_linear( w, j, &f, f.ex / d, f.ey / d, d, 0.0f, 0.0f );
}

static void solve_joint( GpWorld *w, BJoint *j, float dt )
{
	Frame f;
	/* a step solves every joint `passes` times: soft constraints and motor limits are divided among the passes */
	float passes = (float)( ( w->iterations > 0 ? w->iterations : 1 ) * ( w->joint_iters > 0 ? w->joint_iters : 1 ) );
	float dtt = dt * dt, inv_iters = 1.0f / passes;
	switch ( j->type )
	{
	case GP_JOINT_REVOLUTE:
	case GP_JOINT_WELD:
	{
		if ( j->type == GP_JOINT_WELD )
		{
			joint_frame( w, j, &f );
			apply_angular( w, j, &f, wrap_angle( f.thB - f.thA - j->ref ), 0.0f );
		}
		else
		{
			joint_frame( w, j, &f );
			float phi = wrap_angle( f.thB - f.thA - j->ref );
			if ( j->limit )
			{
				float want = phi < j->lo ? j->lo : ( phi > j->hi ? j->hi : phi );
				apply_angular( w, j, &f, phi - want, 0.0f );
			}
			if ( j->motor )
			{ /* the step's relative turn should be speed * dt, with a torque limit */
				float step_turn = phi - j->last_phi, target = j->speed * dt;
				float max_lam = j->max_motor * dtt * ( f.iiA + f.iiB ) * inv_iters;
				apply_angular( w, j, &f, -( target - step_turn ), max_lam );
			}
		}
		point_constraint( w, j );
		break;
	}
	case GP_JOINT_PRISMATIC:
	case GP_JOINT_WHEEL:
	{
		joint_frame( w, j, &f );
		float ca = cosf( f.thA ), sa = sinf( f.thA );
		float ux = ca * j->ux - sa * j->uy, uy = sa * j->ux + ca * j->uy;
		if ( j->type == GP_JOINT_PRISMATIC )
		{
			apply_angular( w, j, &f, wrap_angle( f.thB - f.thA - j->ref ), 0.0f );
			joint_frame( w, j, &f );
		}
		/* off the line: the sideways offset goes */
		float px = -uy, py = ux;
		apply_linear( w, j, &f, px, py, f.ex * px + f.ey * py, 0.0f, 0.0f );
		joint_frame( w, j, &f );
		float t = f.ex * ux + f.ey * uy;
		if ( j->type == GP_JOINT_PRISMATIC && j->limit )
		{
			float want = t < j->lo ? j->lo : ( t > j->hi ? j->hi : t );
			if ( want != t )
			{
				apply_linear( w, j, &f, ux, uy, t - want, 0.0f, 0.0f );
				joint_frame( w, j, &f );
				t = f.ex * ux + f.ey * uy;
			}
		}
		if ( j->motor && j->type == GP_JOINT_PRISMATIC )
		{
			float step_move = t - j->last_t, target = j->speed * dt;
			float wsum = f.imA + f.imB;
			apply_linear( w, j, &f, ux, uy, -( target - step_move ), 0.0f, j->max_motor * dtt * wsum * inv_iters );
			joint_frame( w, j, &f );
			t = f.ex * ux + f.ey * uy;
		}
		if ( j->type == GP_JOINT_WHEEL )
		{
			if ( j->freq > 0.0f )
			{ /* the suspension: a soft constraint pulling the translation back to where it began */
				float omega = 6.2831853f * j->freq, wsum = f.imA + f.imB;
				if ( wsum > 0.0f )
				{
					float k = omega * omega / wsum, alpha = passes / ( k * dtt );
					apply_linear( w, j, &f, ux, uy, t - j->t0, alpha, 0.0f );
					float vrel = 0;
					{ /* damping: part of the relative speed along the axis goes */
						float ax0, ay0, th0, im0, ii0, bx0, by0, th1, im1, ii1;
						node_state( w, j->a, &ax0, &ay0, &th0, &im0, &ii0 );
						node_state( w, j->b, &bx0, &by0, &th1, &im1, &ii1 );
						const Body *ba = j->a.body >= 0 ? &w->bodies[j->a.body] : 0, *bb = j->b.body >= 0 ? &w->bodies[j->b.body] : 0;
						float avx = 0, avy = 0, bvx = 0, bvy = 0;
						if ( ba )
						{
							for ( int k2 = 0; k2 < ba->nc; ++k2 )
								avx += w->pos[2 * ba->c[k2]] - w->prev[2 * ba->c[k2]], avy += w->pos[2 * ba->c[k2] + 1] - w->prev[2 * ba->c[k2] + 1];
							avx /= ba->nc, avy /= ba->nc;
						}
						else if ( j->a.circ >= 0 )
							avx = w->pos[2 * j->a.circ] - w->prev[2 * j->a.circ], avy = w->pos[2 * j->a.circ + 1] - w->prev[2 * j->a.circ + 1];
						if ( bb )
						{
							for ( int k2 = 0; k2 < bb->nc; ++k2 )
								bvx += w->pos[2 * bb->c[k2]] - w->prev[2 * bb->c[k2]], bvy += w->pos[2 * bb->c[k2] + 1] - w->prev[2 * bb->c[k2] + 1];
							bvx /= bb->nc, bvy /= bb->nc;
						}
						else if ( j->b.circ >= 0 )
							bvx = w->pos[2 * j->b.circ] - w->prev[2 * j->b.circ], bvy = w->pos[2 * j->b.circ + 1] - w->prev[2 * j->b.circ + 1];
						vrel = ( bvx - avx ) * ux + ( bvy - avy ) * uy;
						float frac = 2.0f * j->zeta * omega * dt * inv_iters;
						if ( frac > 1.0f )
							frac = 1.0f;
						float d = frac * vrel, fa = f.imA / wsum, fb = f.imB / wsum;
						if ( fa > 0 )
							node_add_velocity( w, j->a, d * fa * ux, d * fa * uy );
						if ( fb > 0 )
							node_add_velocity( w, j->b, -d * fb * ux, -d * fb * uy );
					}
				}
			}
			if ( j->motor )
			{ /* spin the wheel (B) against A: only a box can turn */
				joint_frame( w, j, &f );
				float phi = wrap_angle( f.thB - f.thA - j->ref );
				float step_turn = phi - j->last_phi, target = j->speed * dt;
				apply_angular( w, j, &f, -( target - step_turn ), j->max_motor * dtt * ( f.iiA + f.iiB ) * inv_iters );
			}
		}
		break;
	}
	case GP_JOINT_DISTANCE:
	{
		joint_frame( w, j, &f );
		float d = sqrtf( f.ex * f.ex + f.ey * f.ey );
		if ( d < 1.0e-7f )
			break;
		float nx = f.ex / d, ny = f.ey / d;
		if ( j->freq > 0.0f )
		{ /* a spring toward the rest length, with hard stops at lo and hi */
			float omega = 6.2831853f * j->freq, wsum = f.imA + f.imB;
			if ( wsum > 0.0f )
				apply_linear( w, j, &f, nx, ny, d - j->rest, passes / ( omega * omega / wsum * dtt ), 0.0f );
			joint_frame( w, j, &f );
			d = sqrtf( f.ex * f.ex + f.ey * f.ey );
			if ( d < 1.0e-7f )
				break;
			nx = f.ex / d, ny = f.ey / d;
		}
		float want = d < j->lo ? j->lo : ( d > j->hi ? j->hi : d );
		if ( want != d )
			apply_linear( w, j, &f, nx, ny, d - want, 0.0f, 0.0f );
		break;
	}
	case GP_JOINT_MOUSE:
	{
		joint_frame( w, j, &f ); /* A is the world: the anchor on B is pulled toward the target */
		float ex = j->tx - ( f.cxB + f.rBx ), ey = j->ty - ( f.cyB + f.rBy );
		float d = sqrtf( ex * ex + ey * ey );
		if ( d < 1.0e-7f || f.imB <= 0.0f )
			break;
		float omega = 6.2831853f * ( j->freq > 0 ? j->freq : 5.0f ), k = omega * omega / f.imB;
		float alpha = passes / ( k * dtt );
		float crB = f.rBx * ( ey / d ) - f.rBy * ( ex / d );
		float wB = f.imB + f.iiB * crB * crB, lam = d / ( wB + alpha );
		float max_lam = j->max_force > 0 ? j->max_force * dtt * f.imB / w->cell * inv_iters : 0;
		if ( max_lam > 0 && lam > max_lam )
			lam = max_lam;
		float nx = ex / d, ny = ey / d;
		node_apply( w, j->b, f.imB * lam * nx, f.imB * lam * ny, f.iiB * lam * ( f.rBx * ny - f.rBy * nx ) );
		break;
	}
	}
}

static void solve_body_joints( GpWorld *w, float dt )
{
	for ( int pass = 0; pass < ( w->joint_iters > 0 ? w->joint_iters : 1 ); ++pass )
		for ( int k = 0; k < w->nbj; ++k )
			if ( w->bj[k].active )
				solve_joint( w, &w->bj[k], dt );
}

/* what a motor measures a step's motion against: where the joint stood when the step began */
static void joints_mark( GpWorld *w )
{
	for ( int k = 0; k < w->nbj; ++k )
	{
		BJoint *j = &w->bj[k];
		if ( !j->active )
			continue;
		Frame f;
		joint_frame( w, j, &f );
		float ca = cosf( f.thA ), sa = sinf( f.thA );
		float ux = ca * j->ux - sa * j->uy, uy = sa * j->ux + ca * j->uy;
		j->last_phi = wrap_angle( f.thB - f.thA - j->ref );
		j->last_t = f.ex * ux + f.ey * uy;
	}
}

static JNode make_node( const GpWorld *w, int body, float ax, float ay )
{
	JNode n = { -1, -1 };
	if ( body < 0 )
		return n;
	const Body *b = &w->bodies[body];
	if ( b->is_box || b->is_poly )
	{
		n.body = body;
		return n;
	}
	float best = 1.0e30f; /* a ball or a rope: the circle nearest the anchor */
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float dx = w->pos[2 * c] - ax, dy = w->pos[2 * c + 1] - ay, d = dx * dx + dy * dy;
		if ( d < best )
			best = d, n.circ = c;
	}
	return n;
}

static void share_group( GpWorld *w, JNode a, JNode b, int salt )
{
	const JNode ns[2] = { a, b };
	unsigned gid = 0;
	for ( int s = 0; s < 2 && !gid; ++s )
	{
		int c = ns[s].body >= 0 ? w->bodies[ns[s].body].c[0] : ns[s].circ;
		if ( c >= 0 && w->groups[c] )
			gid = w->groups[c];
	}
	if ( !gid )
		gid = 0x40000000u + (unsigned)salt;
	for ( int s = 0; s < 2; ++s )
	{
		if ( ns[s].body < 0 && ns[s].circ < 0 )
			continue;
		int first = ns[s].body >= 0 ? 0 : ns[s].circ, count = ns[s].body >= 0 ? w->bodies[ns[s].body].nc : 1;
		for ( int k = 0; k < count; ++k )
		{
			int c = ns[s].body >= 0 ? w->bodies[ns[s].body].c[k] : first;
			w->groups[c] = gid;
			w->flt[4 * c + 3] = w->uflt[4 * c + 3] = gid;
		}
	}
}

GpJointDef gp_joint_def( int type, int a, int b, float x, float y )
{
	GpJointDef d;
	memset( &d, 0, sizeof d );
	d.type = type, d.a = a, d.b = b, d.ax = d.bx = x, d.ay = d.by = y;
	d.axis_y = 1.0f;
	d.lower = -1.0e9f, d.upper = 1.0e9f;
	d.max_motor = 1.0e9f;
	return d;
}

int gp_add_body_joint( GpWorld *w, const GpJointDef *d )
{
	if ( d->a < 0 || d->a >= w->nbodies || d->b >= w->nbodies )
		return -1;
	if ( w->nbj >= w->bj_cap )
	{
		w->bj_cap = w->bj_cap ? 2 * w->bj_cap : 16;
		w->bj = realloc( w->bj, w->bj_cap * sizeof( BJoint ) );
	}
	float c = w->cell, ax = ( d->ax - w->ox ) / c, ay = ( d->ay - w->oy ) / c, bx = ( d->bx - w->ox ) / c, by = ( d->by - w->oy ) / c;
	BJoint *j = &w->bj[w->nbj];
	memset( j, 0, sizeof *j );
	j->type = d->type, j->active = 1, j->collide = d->collide_connected != 0;
	j->a = make_node( w, d->a, ax, ay );
	j->b = make_node( w, d->b, bx, by );
	if ( d->type == GP_JOINT_MOUSE )
	{ /* the mouse joint holds body a; there is no other body */
		j->b = j->a;
		j->a.body = j->a.circ = -1;
		bx = ax, by = ay;
	}
	Frame f;
	{ /* anchors into the bodies' frames */
		float cxA, cyA, thA, imA, iiA, cxB, cyB, thB, imB, iiB;
		node_state( w, j->a, &cxA, &cyA, &thA, &imA, &iiA );
		node_state( w, j->b, &cxB, &cyB, &thB, &imB, &iiB );
		float ca = cosf( -thA ), sa = sinf( -thA ), cb = cosf( -thB ), sb = sinf( -thB );
		float rx = ax - cxA, ry = ay - cyA, sx = bx - cxB, sy = by - cyB;
		j->lax = ca * rx - sa * ry, j->lay = sa * rx + ca * ry;
		j->lbx = cb * sx - sb * sy, j->lby = sb * sx + cb * sy;
		j->ref = wrap_angle( thB - thA );
		float len = sqrtf( d->axis_x * d->axis_x + d->axis_y * d->axis_y );
		float uxw = len > 0 ? d->axis_x / len : 0.0f, uyw = len > 0 ? d->axis_y / len : 1.0f;
		j->ux = ca * uxw - sa * uyw, j->uy = sa * uxw + ca * uyw;
	}
	float scale = ( d->type == GP_JOINT_PRISMATIC || d->type == GP_JOINT_DISTANCE ) ? 1.0f / c : 1.0f; /* lengths in tile units */
	j->limit = d->enable_limit != 0, j->motor = d->enable_motor != 0;
	j->lo = d->lower * scale, j->hi = d->upper * scale;
	j->speed = d->motor_speed * ( d->type == GP_JOINT_PRISMATIC ? 1.0f / c : 1.0f );
	j->max_motor = d->max_motor;
	j->freq = d->frequency, j->zeta = d->damping_ratio, j->max_force = d->max_force;
	joint_frame( w, j, &f );
	if ( d->type == GP_JOINT_DISTANCE )
	{
		float len = sqrtf( f.ex * f.ex + f.ey * f.ey );
		j->rest = d->length > 0 ? d->length / c : len;
		if ( !d->enable_limit )
			j->lo = j->hi = j->rest; /* a rigid rod, unless it is a spring (frequency) with stops */
		if ( d->frequency > 0 && !d->enable_limit )
			j->lo = 0, j->hi = 1.0e9f;
	}
	if ( d->type == GP_JOINT_MOUSE )
		j->tx = ax, j->ty = ay;
	{
		float ca = cosf( f.thA ), sa = sinf( f.thA );
		j->t0 = f.ex * ( ca * j->ux - sa * j->uy ) + f.ey * ( sa * j->ux + ca * j->uy );
	}
	j->last_phi = 0.0f;
	j->last_t = j->t0;
	if ( !j->collide && d->type != GP_JOINT_MOUSE )
		share_group( w, j->a, j->b, w->nbj );
	return w->nbj++;
}

void gp_joint_set_target( GpWorld *w, int joint, float x, float y )
{
	if ( joint >= 0 && joint < w->nbj )
		w->bj[joint].tx = ( x - w->ox ) / w->cell, w->bj[joint].ty = ( y - w->oy ) / w->cell;
}

void gp_joint_set_motor( GpWorld *w, int joint, int enable, float speed, float max_motor )
{
	if ( joint < 0 || joint >= w->nbj )
		return;
	BJoint *j = &w->bj[joint];
	j->motor = enable != 0;
	j->speed = speed * ( j->type == GP_JOINT_PRISMATIC ? 1.0f / w->cell : 1.0f );
	j->max_motor = max_motor;
}

void gp_joint_set_limits( GpWorld *w, int joint, int enable, float lower, float upper )
{
	if ( joint < 0 || joint >= w->nbj )
		return;
	BJoint *j = &w->bj[joint];
	float scale = ( j->type == GP_JOINT_PRISMATIC || j->type == GP_JOINT_DISTANCE ) ? 1.0f / w->cell : 1.0f;
	j->limit = enable != 0, j->lo = lower * scale, j->hi = upper * scale;
}

void gp_joint_destroy( GpWorld *w, int joint )
{
	if ( joint >= 0 && joint < w->nbj )
		w->bj[joint].active = 0;
}

void gp_set_joint_iterations( GpWorld *w, int n )
{
	w->joint_iters = n < 1 ? 1 : n;
}

/* B's angle relative to A's (radians, minus the angle at creation), and the slide of B along A's axis (world units) */
float gp_joint_angle( const GpWorld *w, int joint )
{
	const BJoint *j = &w->bj[joint];
	Frame f;
	joint_frame( w, j, &f );
	return wrap_angle( f.thB - f.thA - j->ref );
}

float gp_joint_translation( const GpWorld *w, int joint )
{
	const BJoint *j = &w->bj[joint];
	Frame f;
	joint_frame( w, j, &f );
	float ca = cosf( f.thA ), sa = sinf( f.thA );
	return ( f.ex * ( ca * j->ux - sa * j->uy ) + f.ey * ( sa * j->ux + ca * j->uy ) - j->t0 ) * w->cell;
}


/* ---- rotation: spinning balls, polygons ------------------------------------------------------- */
/* A ball's angle is kept as its own state (Verlet, like a position): it spins freely, rolls on a tile (the spin
 * is pulled toward "no slipping at the contact"), and joints and gp_body_set_angular_velocity change it.
 * A polygon is a cluster of circles kept rigid by shape matching after every step (its circles are fitted to their
 * layout by a rotation and a shift, and given the one rigid velocity that conserves linear and angular momentum), so
 * it turns, tumbles and rolls on friction at its circles' contacts without any joints. */
static void spin_begin_step( GpWorld *w )
{
	for ( int i = 0; i < w->n; ++i )
		if ( w->spin[i] && !w->frozen[i] && !w->pinned[i] )
		{
			float a = w->ang[i];
			w->ang[i] = a + ( a - w->prev_ang[i] ) * 0.9995f;
			w->prev_ang[i] = a;
		}
}

static void spin_end_step( GpWorld *w )
{
	for ( int i = 0; i < w->n && w->tw; ++i )
	{
		if ( !w->spin[i] || w->frozen[i] || w->pinned[i] )
			continue;
		float qx, qy;
		if ( circle_tile( w, i, 0.02f, &qx, &qy ) < 0 )
			continue;
		float nx = w->pos[2 * i] - qx, ny = w->pos[2 * i + 1] - qy, d = sqrtf( nx * nx + ny * ny );
		if ( d < 1.0e-6f )
			continue;
		nx /= d, ny /= d;
		float vt = ( w->pos[2 * i] - w->prev[2 * i] ) * -ny + ( w->pos[2 * i + 1] - w->prev[2 * i + 1] ) * nx;
		float roll = vt / w->rad[i], om = w->ang[i] - w->prev_ang[i];
		float mu = w->use_material ? sqrtf( w->mat[5 * i] * w->friction ) : w->friction;
		float grip = 5.0f * mu > 1.0f ? 1.0f : 5.0f * mu;
		om += grip * ( roll - om );
		w->prev_ang[i] = w->ang[i] - om;
	}
}

static void match_polygon( GpWorld *w, Body *b )
{
	if ( !b->active || b->nc < 2 )
		return;
	float cx = 0, cy = 0, mass = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		float inv = circle_invm( w, b->c[k] );
		if ( inv <= 0.0f )
			return; /* held (kinematic): nothing to match */
		mass += 1.0f / inv;
		cx += w->pos[2 * b->c[k]], cy += w->pos[2 * b->c[k] + 1];
	}
	cx /= b->nc, cy /= b->nc;
	float th = b->locked ? b->lock_th : body_angle( w, b );
	float ca = cosf( th ), sa = sinf( th ), s = b->stiff > 0.0f ? b->stiff : 1.0f;
	for ( int k = 0; k < b->nc; ++k )
	{ /* the shape's rigid placement; the correction does not count as motion */
		int c = b->c[k];
		float tx = cx + b->lx[k] * ca - b->ly[k] * sa, ty = cy + b->lx[k] * sa + b->ly[k] * ca;
		float dx = s * ( tx - w->pos[2 * c] ), dy = s * ( ty - w->pos[2 * c + 1] );
		w->pos[2 * c] += dx, w->pos[2 * c + 1] += dy;
		w->prev[2 * c] += dx, w->prev[2 * c + 1] += dy;
	}
	/* one rigid velocity: the mean and the spin that carry the circles' momentum */
	float vx = 0, vy = 0, L = 0, I = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float m = 1.0f / circle_invm( w, c );
		vx += m * ( w->pos[2 * c] - w->prev[2 * c] ), vy += m * ( w->pos[2 * c + 1] - w->prev[2 * c + 1] );
	}
	vx /= mass, vy /= mass;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float m = 1.0f / circle_invm( w, c ), rx = w->pos[2 * c] - cx, ry = w->pos[2 * c + 1] - cy;
		float ux = ( w->pos[2 * c] - w->prev[2 * c] ) - vx, uy = ( w->pos[2 * c + 1] - w->prev[2 * c + 1] ) - vy;
		L += m * ( rx * uy - ry * ux ), I += m * ( rx * rx + ry * ry );
	}
	float om = ( I > 1.0e-12f && !b->locked ) ? L / I : 0.0f;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float rx = w->pos[2 * c] - cx, ry = w->pos[2 * c + 1] - cy;
		float rvx = vx - om * ry, rvy = vy + om * rx;
		float ox = w->pos[2 * c] - w->prev[2 * c], oy = w->pos[2 * c + 1] - w->prev[2 * c + 1];
		w->prev[2 * c] = w->pos[2 * c] - ( ox + s * ( rvx - ox ) );
		w->prev[2 * c + 1] = w->pos[2 * c + 1] - ( oy + s * ( rvy - oy ) );
	}
}

static void match_polygons( GpWorld *w )
{
	for ( int bi = 0; bi < w->nbodies; ++bi )
		if ( w->bodies[bi].is_poly )
			match_polygon( w, &w->bodies[bi] );
}

/* rigid-body motion of a box or polygon: its mean velocity (tile units / step), spin (rad / step), centre */
static int rigid_motion( const GpWorld *w, const Body *b, float *cx, float *cy, float *vx, float *vy, float *om )
{
	if ( !( b->is_box || b->is_poly ) )
		return 0;
	float mass = 0;
	*cx = *cy = *vx = *vy = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float m = w->rad[c] * w->dens[c];
		mass += m;
		*cx += w->pos[2 * c], *cy += w->pos[2 * c + 1];
		*vx += m * ( w->pos[2 * c] - w->prev[2 * c] ), *vy += m * ( w->pos[2 * c + 1] - w->prev[2 * c + 1] );
	}
	*cx /= b->nc, *cy /= b->nc, *vx /= mass, *vy /= mass;
	float L = 0, I = 0;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float m = w->rad[c] * w->dens[c], rx = w->pos[2 * c] - *cx, ry = w->pos[2 * c + 1] - *cy;
		L += m * ( rx * ( ( w->pos[2 * c + 1] - w->prev[2 * c + 1] ) - *vy ) - ry * ( ( w->pos[2 * c] - w->prev[2 * c] ) - *vx ) );
		I += m * ( rx * rx + ry * ry );
	}
	*om = I > 1.0e-12f ? L / I : 0.0f;
	return 1;
}

float gp_body_angular_velocity( const GpWorld *w, int body, float dt )
{
	const Body *b = &w->bodies[body];
	if ( b->nc == 1 && w->spin[b->c[0]] )
		return ( w->ang[b->c[0]] - w->prev_ang[b->c[0]] ) / dt;
	float cx, cy, vx, vy, om;
	return rigid_motion( w, b, &cx, &cy, &vx, &vy, &om ) ? om / dt : 0.0f;
}

void gp_body_set_angular_velocity( GpWorld *w, int body, float dt, float omega )
{
	Body *b = &w->bodies[body];
	if ( b->nc == 1 && w->spin[b->c[0]] )
	{
		w->prev_ang[b->c[0]] = w->ang[b->c[0]] - omega * dt;
		return;
	}
	float cx, cy, vx, vy, om;
	if ( !rigid_motion( w, b, &cx, &cy, &vx, &vy, &om ) || b->locked )
		return;
	float o = omega * dt;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float rx = w->pos[2 * c] - cx, ry = w->pos[2 * c + 1] - cy;
		w->prev[2 * c] = w->pos[2 * c] - ( vx - o * ry );
		w->prev[2 * c + 1] = w->pos[2 * c + 1] - ( vy + o * rx );
	}
}

void gp_body_add_angular_velocity( GpWorld *w, int body, float dt, float domega )
{
	gp_body_set_angular_velocity( w, body, dt, gp_body_angular_velocity( w, body, dt ) + domega );
}

/* turn the body to `angle` about its centre, keeping its velocity */
void gp_body_set_angle( GpWorld *w, int body, float angle )
{
	Body *b = &w->bodies[body];
	if ( b->nc == 1 && w->spin[b->c[0]] )
	{
		float d = angle - w->ang[b->c[0]];
		w->ang[b->c[0]] += d, w->prev_ang[b->c[0]] += d;
		return;
	}
	if ( !( b->is_box || b->is_poly ) )
		return;
	float d = angle - body_angle( w, b ), cx = 0, cy = 0, ca = cosf( d ), sa = sinf( d );
	for ( int k = 0; k < b->nc; ++k )
		cx += w->pos[2 * b->c[k]], cy += w->pos[2 * b->c[k] + 1];
	cx /= b->nc, cy /= b->nc;
	for ( int k = 0; k < b->nc; ++k )
	{
		int c = b->c[k];
		float rx = w->pos[2 * c] - cx, ry = w->pos[2 * c + 1] - cy;
		float nx = cx + rx * ca - ry * sa, ny = cy + rx * sa + ry * ca;
		w->prev[2 * c] += nx - w->pos[2 * c], w->prev[2 * c + 1] += ny - w->pos[2 * c + 1];
		w->pos[2 * c] = nx, w->pos[2 * c + 1] = ny;
	}
	if ( b->locked )
		b->lock_th = angle;
}

static float poly_edge_distance( const float *px, const float *py, int n, float x, float y, int *inside )
{
	float best = 1.0e30f;
	int in = 0;
	for ( int i = 0, j = n - 1; i < n; j = i++ )
	{
		float ex = px[i] - px[j], ey = py[i] - py[j], len2 = ex * ex + ey * ey;
		float t = len2 > 0 ? ( ( x - px[j] ) * ex + ( y - py[j] ) * ey ) / len2 : 0.0f;
		t = t < 0 ? 0 : ( t > 1 ? 1 : t );
		float dx = x - ( px[j] + t * ex ), dy = y - ( py[j] + t * ey ), d = sqrtf( dx * dx + dy * dy );
		if ( d < best )
			best = d;
		if ( ( py[i] > y ) != ( py[j] > y ) && x < ( px[j] - px[i] ) * ( y - py[i] ) / ( py[j] - py[i] ) + px[i] )
			in = !in;
	}
	*inside = in;
	return best;
}

/* A polygon (any simple one) as circles of `radius` (0: automatic) filling it: along the inside of its edges, at its
 * corners and across the middle. The circles are the collision shape, so edges are as smooth as the circles are small
 * (the polygon is rounded by at most `radius` at a corner). verts: x0, y0, x1, y1, ... in world units relative to
 * (x, y), which is the body's origin; at `angle`. Returns the body or -1 (no room, or too few vertices). */
int gp_add_polygon( GpWorld *w, float x, float y, float angle, const float *verts, int nverts, float radius, float vx,
					float vy, float dt, float stiffness )
{
	if ( nverts < 3 || nverts > 64 )
		return -1;
	float px[64], py[64], minx = 1e30f, miny = 1e30f, maxx = -1e30f, maxy = -1e30f, area = 0;
	for ( int i = 0; i < nverts; ++i )
	{
		px[i] = verts[2 * i], py[i] = verts[2 * i + 1];
		minx = fminf( minx, px[i] ), maxx = fmaxf( maxx, px[i] ), miny = fminf( miny, py[i] ), maxy = fmaxf( maxy, py[i] );
	}
	for ( int i = 0, j = nverts - 1; i < nverts; j = i++ )
		area += px[j] * py[i] - px[i] * py[j];
	if ( fabsf( area ) < 1.0e-9f )
		return -1;
	float r = radius > 0 ? radius : fminf( maxx - minx, maxy - miny ) / 6.0f;
	enum { MAXC = 96 };
	float cx_[MAXC], cy_[MAXC];
	int nc = 0;
	for ( int attempt = 0; attempt < 6; ++attempt )
	{
		nc = 0;
		float sp = 1.3f * r;
		float orient = area > 0 ? 1.0f : -1.0f; /* the inward normal of an edge i-1 -> i is (-ey, ex) * orient */
		for ( int i = 0, j = nverts - 1; i < nverts && nc < MAXC; j = i++ )
		{
			/* a corner circle on the bisector, then the edge's own circles inset by r */
			int k = ( i + 1 ) % nverts;
			float e1x = px[i] - px[j], e1y = py[i] - py[j], l1 = sqrtf( e1x * e1x + e1y * e1y );
			float e2x = px[k] - px[i], e2y = py[k] - py[i], l2 = sqrtf( e2x * e2x + e2y * e2y );
			if ( l1 < 1e-9f || l2 < 1e-9f )
				continue;
			float n1x = -e1y / l1 * orient, n1y = e1x / l1 * orient, n2x = -e2y / l2 * orient, n2y = e2x / l2 * orient;
			float bx = n1x + n2x, by = n1y + n2y, bl = sqrtf( bx * bx + by * by );
			if ( bl > 1e-6f )
			{
				bx /= bl, by /= bl;
				float t = r / fmaxf( 0.15f, bx * n1x + by * n1y );
				float qx = px[i] + bx * t, qy = py[i] + by * t;
				int in;
				float d = poly_edge_distance( px, py, nverts, qx, qy, &in );
				int near = 0;
				for ( int m = 0; m < nc && !near; ++m )
					near = hypotf( cx_[m] - qx, cy_[m] - qy ) < 0.9f * sp;
				if ( in && d > 0.97f * r && !near && nc < MAXC )
					cx_[nc] = qx, cy_[nc++] = qy;
			}
			int steps = (int)( l2 / sp );
			for ( int s2 = 1; s2 <= steps && nc < MAXC; ++s2 )
			{
				float f = (float)s2 / ( steps + 1 ), qx = px[i] + e2x * f + n2x * r, qy = py[i] + e2y * f + n2y * r;
				int in;
				float d = poly_edge_distance( px, py, nverts, qx, qy, &in );
				int near = 0;
				for ( int m = 0; m < nc && !near; ++m )
					near = hypotf( cx_[m] - qx, cy_[m] - qy ) < 0.9f * sp;
				if ( in && d > 0.97f * r && !near )
					cx_[nc] = qx, cy_[nc++] = qy;
			}
		}
		for ( float gy = miny + r; gy <= maxy - r + 1e-6f && nc < MAXC; gy += sp )
			for ( float gx = minx + r; gx <= maxx - r + 1e-6f && nc < MAXC; gx += sp )
			{
				int in;
				float d = poly_edge_distance( px, py, nverts, gx, gy, &in );
				int near = 0;
				for ( int m = 0; m < nc && !near; ++m )
					near = hypotf( cx_[m] - gx, cy_[m] - gy ) < 0.9f * sp;
				if ( in && d > 0.97f * r && !near )
					cx_[nc] = gx, cy_[nc++] = gy;
			}
		if ( nc < MAXC )
			break;
		r *= 1.4f; /* too many: bigger circles */
	}
	if ( nc < 2 )
		return -1;
	if ( w->n + nc > w->cap )
		return -1;
	int id = new_body( w, nc );
	if ( id < 0 )
		return -1;
	Body *b = &w->bodies[id];
	b->nc = nc, b->is_poly = 1, b->stiff = stiffness > 0 ? ( stiffness > 1 ? 1 : stiffness ) : 1.0f;
	b->lx = calloc( nc, sizeof( float ) );
	b->ly = calloc( nc, sizeof( float ) );
	float mx = 0, my = 0;
	for ( int k = 0; k < nc; ++k )
		mx += cx_[k], my += cy_[k];
	mx /= nc, my /= nc;
	b->ox_l = mx, b->oy_l = my;
	b->w = maxx - minx, b->h = maxy - miny;
	float ca = cosf( angle ), sa = sinf( angle );
	for ( int k = 0; k < nc; ++k )
	{
		b->lx[k] = ( cx_[k] - mx ) / w->cell, b->ly[k] = ( cy_[k] - my ) / w->cell;
		b->c[k] = add_circle( w, x + cx_[k] * ca - cy_[k] * sa, y + cx_[k] * sa + cy_[k] * ca, r, vx, vy, dt, GP_TAG_CRATE,
							  id + 1, id );
	}
	return id;
}

/* where the polygon's origin (the point its vertices were given relative to) is now */
void gp_body_origin( const GpWorld *w, int body, float *x, float *y )
{
	const Body *b = &w->bodies[body];
	float cx, cy, a;
	gp_body_pose( w, body, &cx, &cy, &a );
	float ca = cosf( a ), sa = sinf( a );
	*x = cx - ( b->ox_l * ca - b->oy_l * sa );
	*y = cy - ( b->ox_l * sa + b->oy_l * ca );
}

void gp_step( GpWorld *w, float dt )
{
	int n = w->n;
	if ( n == 0 )
		return;
	if ( w->joints_dirty )
		upload_joints( w );
	int mat = w->use_material;
	w->last_dt = dt;
	if ( w->nbj )
		joints_mark( w );
	spin_begin_step( w );
	if ( mat )
	{ /* per-circle inputs of prowl_material's kernels: world defaults, drag, inverse mass */
		for ( int i = 0; i < n; ++i )
		{
			if ( !w->custom[i] )
				w->mat[5 * i] = w->friction;
			float sand = w->tags[i] == 1 ? w->sand_damping / w->damping : 1.0f;
			w->mat[5 * i + 3] = sand / ( 1.0f + w->drag[i] * dt );
			w->invm[i] = ( w->pinned[i] || w->frozen[i] ) ? 0.0f : 1.0f / ( w->rad[i] * w->dens[i] );
			w->mat[5 * i + 4] = w->invm[i];
			w->omega[i] = w->spin[i] ? w->ang[i] - w->prev_ang[i] : 0.0f;
			memcpy( &w->flt[4 * i], &w->uflt[4 * i], 4 * sizeof( unsigned ) );
			if ( w->sensor[i] )
				w->flt[4 * i] = w->flt[4 * i + 1] = w->flt[4 * i + 2] = 0u; /* collides with nothing */
		}
		for ( int bi = 0; bi < w->nbodies; ++bi )
		{ /* the circles of a turning body slide on a tile at the speed of the surface, not of their centres */
			const Body *b = &w->bodies[bi];
			float cx, cy, vx, vy, om;
			if ( b->active && rigid_motion( w, b, &cx, &cy, &vx, &vy, &om ) )
				for ( int k = 0; k < b->nc; ++k )
					w->omega[b->c[k]] = om;
		}
		GP_LAUNCH( n, integrate_material( w->pos, w->prev, w->rad, w->mat, dt, w->gx, w->gy, w->damping, w->ww,
										  w->wh, n ) );
	}
	else
		GP_LAUNCH( n, integrate_tagged( w->pos, w->prev, w->rad, w->tags, dt, w->gx, w->gy, w->damping,
										w->sand_damping, w->ww, w->wh, n ) );
	restore_pinned( w );
	build_cells( w );
	for ( int it = 0; it < ( w->iterations > 0 ? w->iterations : 1 ); ++it )
	{
		for ( unsigned color = 1; color <= 4; ++color )
			if ( mat )
				GP_LAUNCH( w->last_ncells,
						   solve_collisions_filtered( w->chunks, w->cells, w->cell_ids, w->object_ids, w->pos, w->rad,
													  w->uniform, w->flt, w->mat, w->prev, color,
													  w->bounce_threshold * dt / w->cell ) );
			else
			GP_LAUNCH( w->last_ncells,
					   solve_collisions_grouped( w->chunks, w->cells, w->cell_ids, w->object_ids, w->pos, w->rad,
												 w->uniform, w->groups, color ) );
		restore_pinned( w );
		if ( w->nbj )
		{
			solve_body_joints( w, dt );
			restore_pinned( w );
		}
		for ( int b = 0; b < w->nbatches; ++b )
			GP_LAUNCH( w->batch_count[b], solve_joints( w->pos, w->rad, w->ji, w->jf, w->batch_first[b], w->batch_count[b] ) );
		if ( w->tw && w->th )
			GP_LAUNCH( n, tile_collide( w->pos, w->rad, w->tiles, w->tw, w->th, n ) );
		if ( w->ncurves )
			GP_LAUNCH( n, curve_collide( w->pos, w->prev, w->rad, w->mat, w->curves, n, w->ncurves ) );
		restore_pinned( w );
	}
	if ( w->tw && w->th && mat )
		GP_LAUNCH( n, tile_material( w->pos, w->prev, w->rad, w->tiles, w->mat, w->omega, w->tw, w->th, n, w->friction,
									 w->tile_bounce, w->bounce_threshold * dt / w->cell ) );
	else if ( w->tw && w->th && ( w->friction > 0 || w->bounce_loss > 0 ) )
		GP_LAUNCH( n, tile_friction( w->pos, w->prev, w->rad, w->tiles, w->tw, w->th, n, w->friction, w->bounce_loss ) );
	if ( w->ncurves )
		GP_LAUNCH( n, curve_material( w->pos, w->prev, w->rad, w->mat, w->omega, w->curves, n, w->ncurves,
									  w->bounce_threshold * dt / w->cell ) );
	for ( int bi = 0; bi < w->nbodies; ++bi )
	{
		Body *b = &w->bodies[bi];
		if ( !b->locked || !b->active )
			continue;
		float cx = 0, cy = 0;
		for ( int k = 0; k < b->nc; ++k )
			cx += w->pos[2 * b->c[k]], cy += w->pos[2 * b->c[k] + 1];
		cx /= b->nc, cy /= b->nc;
		for ( int k = 0; k < b->nc; ++k )
		{
			int c = b->c[k];
			float nx = cx + b->lx[k], ny = cy + b->ly[k];
			w->prev[2 * c] += nx - w->pos[2 * c];
			w->prev[2 * c + 1] += ny - w->pos[2 * c + 1];
			w->pos[2 * c] = nx, w->pos[2 * c + 1] = ny;
		}
	}
	for ( int i = 0; i < n; ++i )
		if ( w->frozen[i] )
		{
			w->pos[2 * i] = w->prev[2 * i] = w->frozen_pos[2 * i];
			w->pos[2 * i + 1] = w->prev[2 * i + 1] = w->frozen_pos[2 * i + 1];
		}
	match_polygons( w );
	spin_end_step( w );
	if ( w->events_on )
		update_events( w );
}

/* ---- queries ------------------------------------------------------------------------------- */
int gp_raycast( GpWorld *w, float ox, float oy, float dx, float dy, float distance, int mask, int ignore_body,
				GpHit *hit )
{
	float len = sqrtf( dx * dx + dy * dy );
	if ( len <= 0.0f )
		return 0;
	float ray[8] = { ( ox - w->ox ) / w->cell, ( oy - w->oy ) / w->cell, dx / len, dy / len, distance / w->cell, 0, 0, 0 };
	float h[4];
	unsigned id = 0;
	/* with no tile grid the kernel must not walk one */
	GP_LAUNCH( 1, raycast( w->pos, w->rad, w->groups, w->tiles, ray, h, &id, w->tw, w->th, w->n, 1, mask,
						   ignore_body >= 0 ? (unsigned)ignore_body + 1u : 0u ) );
	if ( h[3] == 0.0f )
		return 0;
	hit->distance = h[0] * w->cell;
	hit->x = ox + dx / len * hit->distance;
	hit->y = oy + dy / len * hit->distance;
	hit->nx = h[1];
	hit->ny = h[2];
	hit->kind = (int)h[3];
	hit->body = -1;
	hit->tile = -1;
	if ( hit->kind == 2 )
		hit->body = w->owner[id];
	else
		hit->tile = (int)id;
	return 1;
}

int gp_overlap_circle( const GpWorld *w, float x, float y, float radius, int *body )
{
	for ( int i = 0; i < w->n; ++i )
	{
		float dx = gp_circle_x( w, i ) - x, dy = gp_circle_y( w, i ) - y, r = gp_circle_radius( w, i ) + radius;
		if ( dx * dx + dy * dy < r * r )
		{
			if ( body )
				*body = w->owner[i];
			return 1;
		}
	}
	return 0;
}

int gp_tile_at( const GpWorld *w, float x, float y, int *tile )
{
	int tx = (int)floorf( ( x - w->ox ) / w->cell ), ty = (int)floorf( ( y - w->oy ) / w->cell );
	if ( tx < 0 || ty < 0 || tx >= w->tw || ty >= w->th )
		return 0;
	if ( tile )
		*tile = ty * w->tw + tx;
	return w->tiles[ty * w->tw + tx] != 0;
}

int gp_body_pairs( const GpWorld *w, int *a, int *b, int max )
{
	int n = 0;
	for ( int i = 0; i < w->n; ++i )
		for ( int j = i + 1; j < w->n; ++j )
		{
			if ( w->owner[i] < 0 || w->owner[j] < 0 || w->owner[i] == w->owner[j] || w->frozen[i] || w->frozen[j] )
				continue;
			float dx = w->pos[2 * i] - w->pos[2 * j], dy = w->pos[2 * i + 1] - w->pos[2 * j + 1];
			float r = w->rad[i] + w->rad[j] + 0.03f;
			if ( dx * dx + dy * dy < r * r )
			{
				int p = w->owner[i] < w->owner[j] ? w->owner[i] : w->owner[j];
				int q = w->owner[i] < w->owner[j] ? w->owner[j] : w->owner[i];
				int dup = 0;
				for ( int k = 0; k < n && !dup; ++k )
					dup = a[k] == p && b[k] == q;
				if ( !dup && n < max )
				{
					a[n] = p;
					b[n] = q;
					++n;
				}
			}
		}
	return n;
}

int gp_tile_contacts( const GpWorld *w, int *body, int *tile, int max )
{
	int n = 0;
	for ( int i = 0; i < w->n && w->tw; ++i )
	{
		if ( w->frozen[i] || w->owner[i] < 0 )
			continue;
		float x = w->pos[2 * i], y = w->pos[2 * i + 1], r = w->rad[i] + 0.03f;
		for ( int ty = (int)floorf( y - r ); ty <= (int)floorf( y + r ); ++ty )
			for ( int tx = (int)floorf( x - r ); tx <= (int)floorf( x + r ); ++tx )
			{
				if ( tx < 0 || ty < 0 || tx >= w->tw || ty >= w->th || !w->tiles[ty * w->tw + tx] )
					continue;
				float qx = fminf( fmaxf( x, tx ), tx + 1.0f ), qy = fminf( fmaxf( y, ty ), ty + 1.0f );
				float dx = x - qx, dy = y - qy;
				if ( dx * dx + dy * dy < r * r )
				{
					int dup = 0;
					for ( int k = 0; k < n && !dup; ++k )
						dup = body[k] == w->owner[i] && tile[k] == ty * w->tw + tx;
					if ( !dup && n < max )
					{
						body[n] = w->owner[i];
						tile[n] = ty * w->tw + tx;
						++n;
					}
				}
			}
	}
	return n;
}
