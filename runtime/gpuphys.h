/* gpuphys -- the circle physics of gpu-physics-engine as a C library.
 *
 * The same kernels the GPU runs (kernels/NAME.rs, translated to C by Crust) run here once per
 * invocation on the CPU; the parts of the pipeline that need workgroup memory (the radix sort,
 * the prefix sum) are plain C. Results match the reference interpreter's.
 *
 * Space: the library works in tile units (one solid-tile cell = 1.0) so the tile collider needs
 * no scale; every call here takes and returns WORLD units (`cell` world units per tile, `ox, oy`
 * the world position of tile (0, 0)'s corner).
 *
 * Everything is circles: a ball is one; a box is a grid of at least 2 x 2 held by stiff
 * distance joints; a rope is a chain. Circles of one body never collide with each other.
 */
#ifndef GPUPHYS_H
#define GPUPHYS_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpWorld GpWorld;

enum { GP_TAG_BALL = 0, GP_TAG_SAND = 1, GP_TAG_CRATE = 2, GP_TAG_SHARD = 3, GP_TAG_ROPE = 4 };
enum { GP_RIGID = 0, GP_ROPE = 1, GP_STRUT = 2 };
enum { GP_RAY_TILES = 1, GP_RAY_CIRCLES = 2 };

typedef struct GpHit {
	float x, y, nx, ny, distance;
	int kind;  /* 1 tile, 2 circle */
	int body;  /* the body of a circle hit, -1 for a free circle */
	int tile;  /* the tile's index (ty * tiles_w + tx) */
} GpHit;

/* A world of up to `capacity` circles and `max_joints` joints. The world box is
 * [ox, ox + width] x [oy, oy + height]; gravity in world units / s^2. */
GpWorld *gp_create( int capacity, int max_joints, float ox, float oy, float width, float height, float cell,
					float gravity_x, float gravity_y );
void gp_destroy( GpWorld *w );

/* Solid tiles: width x height cells, row-major (index = ty * width + tx), non-zero is solid; tile
 * (0, 0) is the cell at (ox, oy). The cells are copied. */
void gp_set_tiles( GpWorld *w, int width, int height, const unsigned *cells );

/* Bodies. Each returns a body id (>= 0) or -1 when full. Velocities in world units / s. */
int gp_add_ball( GpWorld *w, float x, float y, float radius, float vx, float vy, float dt );
int gp_add_box( GpWorld *w, float x, float y, float width, float height, float angle, float vx, float vy,
				float dt, float stiffness );
int gp_add_curve( GpWorld *w, float x0, float y0, float x1, float y1, float x2, float y2, float friction, float bounciness, int side ); /* static quadratic Bezier collider (side 0 thin wall, +1/-1 solid contour: inside left/right); index, -1 without the prowl_curves kernels */
void gp_clear_curves( GpWorld *w );
int gp_curve_count( const GpWorld *w );
int gp_add_sand( GpWorld *w, float x, float y, float radius, float vx, float vy, float dt ); /* a free circle (tag sand): returns its circle index, -1 when full */
int gp_blast( GpWorld *w, float x, float y, float radius, float force ); /* pushes circles away: force = speed at the centre (units/s), linear to the edge */
int gp_add_rope( GpWorld *w, float ax, float ay, float bx, float by, int segments, float radius, int pin_a, int pin_b );

int gp_body_count( const GpWorld *w );
int gp_body_circles( const GpWorld *w, int body );
int gp_body_circle( const GpWorld *w, int body, int k ); /* circle index of the k-th circle */
void gp_body_pose( const GpWorld *w, int body, float *x, float *y, float *angle );
void gp_body_velocity( const GpWorld *w, int body, float dt, float *vx, float *vy );
void gp_body_set_velocity( GpWorld *w, int body, float dt, float vx, float vy );
void gp_body_add_velocity( GpWorld *w, int body, float dt, float dvx, float dvy );
void gp_body_set_position( GpWorld *w, int body, float x, float y ); /* a teleport that keeps velocity */
void gp_body_set_active( GpWorld *w, int body, int active ); /* inactive: out of the simulation */
/* A locked box keeps its orientation (shape matching after every step): Unity's freezeRotation. */
/* Kinematic: the body's circles stay where gp_body_move puts them (pushing, never pushed). */
void gp_body_set_kinematic( GpWorld *w, int body, int on );
void gp_body_move( GpWorld *w, int body, float x, float y ); /* its centre, world units */
void gp_body_lock_rotation( GpWorld *w, int body, int lock );

/* Joints between circle indices (a_circle, b_circle), or a circle and a fixed point (b < 0). */
int gp_add_joint( GpWorld *w, int a, int b, int kind, float rest, float stiffness, float fx, float fy );

/* One surface for the world: friction (Coulomb-like, 0.4 is Unity's default) and bounciness (0..1).
 * The kernels take one pair for every contact between a circle and a tile. */
void gp_set_material( GpWorld *w, float friction, float bounciness );

/* Rotation. A ball has an angle: gp_body_pose gives it, it spins freely, rolls on a tile (the spin is pulled toward not
 * slipping at the contact) and joints can turn it (so a revolute anchor on a ball may be anywhere). A polygon is a cluster of
 * circles that fill it, kept rigid by shape matching after each step (not by joints): it turns and tumbles by itself, and a
 * circle's friction at a contact spins it. A box is as before. Angles are radians, counter-clockwise.
 * gp_add_polygon: verts are x0, y0, x1, y1, ... relative to (x, y), the body's origin, turned by `angle`; `radius` is the
 * circle radius (0: a sixth of the polygon's smaller side), the polygon is rounded by about that at its corners and the
 * edges are as straight as the circles are small; `stiffness` 1 is rigid, less is a soft body. gp_body_pose gives the
 * circles' centroid (which is not the origin): gp_body_origin gives where the origin is. */
int gp_add_polygon( GpWorld *w, float x, float y, float angle, const float *verts, int nverts, float radius, float vx,
					float vy, float dt, float stiffness );
void gp_body_origin( const GpWorld *w, int body, float *x, float *y );
float gp_body_angular_velocity( const GpWorld *w, int body, float dt ); /* rad/s */
void gp_body_set_angular_velocity( GpWorld *w, int body, float dt, float omega );
void gp_body_add_angular_velocity( GpWorld *w, int body, float dt, float domega );
void gp_body_set_angle( GpWorld *w, int body, float angle );

/* Joints between bodies, as Box2D's: revolute (a pin, with angle limits and a motor), weld, prismatic (a slider with
 * translation limits and a motor), wheel (a slider with a spring, and a motor that spins the wheel), distance (a rod, rope
 * or spring) and mouse (drags a point of a body to a target).
 *
 * A BOX or POLYGON is one rigid body in a joint (all its circles move together, with a mass and inertia from them). A ball, or one
 * circle of a rope, takes part with the circle nearest the anchor (a rope's circle cannot turn), so angle limits and
 * motors work on any body that can turn: a box, a polygon or a ball (not a circle of a rope). b < 0 is the world. Anchors and the axis are in world units at the pose the bodies have
 * when the joint is made (anchor_b = anchor_a for everything but a distance joint). Angles are radians (counter-clockwise),
 * translations are world units along the axis, in body a's frame. The solver moves positions (PBD), `joint_iterations`
 * times a step (default 4): a long chain needs more. Bodies joined by a joint do not collide with each other unless
 * collide_connected is set (they then share a collision group: a chain of joints makes one group).
 * The joints move circles directly: kinematic bodies and inactive ones act as the world. */
enum { GP_JOINT_DISTANCE = 1, GP_JOINT_REVOLUTE = 2, GP_JOINT_WELD = 3, GP_JOINT_PRISMATIC = 4, GP_JOINT_WHEEL = 5, GP_JOINT_MOUSE = 6 };
typedef struct GpJointDef {
	int type, a, b;
	float ax, ay, bx, by;      /* anchors (world units) */
	float axis_x, axis_y;      /* prismatic / wheel: the slide axis (world), default up */
	int enable_limit;          /* revolute: angle in [lower, upper]; prismatic: translation; distance: length */
	float lower, upper;
	int enable_motor;          /* revolute / wheel: angular speed (rad/s); prismatic: speed along the axis */
	float motor_speed, max_motor; /* max_motor: a torque / force in (mass unit) x world units, mass = radius (tile units) x density */
	float frequency, damping_ratio; /* wheel suspension, soft distance joint, mouse joint (Hz, 0..1) */
	float length;              /* distance: the rest length, 0 = the distance at creation */
	float max_force;           /* mouse: the strongest pull (0 = no limit) */
	int collide_connected;
} GpJointDef;
GpJointDef gp_joint_def( int type, int a, int b, float x, float y ); /* defaults: no limit, no motor, axis up, bx = ax */
int gp_add_body_joint( GpWorld *w, const GpJointDef *def ); /* a joint id, or -1 */
void gp_joint_set_target( GpWorld *w, int joint, float x, float y ); /* mouse joint */
void gp_joint_set_motor( GpWorld *w, int joint, int enable, float speed, float max_motor );
void gp_joint_set_limits( GpWorld *w, int joint, int enable, float lower, float upper );
void gp_joint_destroy( GpWorld *w, int joint );
void gp_set_joint_iterations( GpWorld *w, int n );
float gp_joint_angle( const GpWorld *w, int joint );       /* B's angle relative to A's, minus what it was at creation */
float gp_joint_translation( const GpWorld *w, int joint ); /* B's slide along the axis, minus what it was at creation */

/* Contact events. gp_enable_events(w, 1) makes every gp_step compare what touches with what touched before and queue
 *   GP_EVENT_BEGIN / GP_EVENT_END           two bodies (a < b) start / stop touching; b = -1 and tile = the tile index: a body
 *                                           and the solid tiles. x, y: the contact point (world units); nx, ny: the unit normal from
 *                                           a toward b; speed: how fast they were closing along it (world units/s), for BEGIN. The speed is read
 *                                           after the solver pushed them apart, so it is lower than the true impact speed.
 *   GP_EVENT_TRIGGER_ENTER / TRIGGER_EXIT   a = a sensor body (gp_body_set_sensor), b = the body that entered / left it.
 * A contact begins when circles are within 0.01 tile units of touching and ends beyond 0.05 (no flicker for a resting
 * contact). Sensors collide with nothing; a body's filter (gp_body_set_filter) decides which bodies a sensor sees.
 * gp_poll_events copies up to `max` queued events (oldest first) and removes them. Ends come before begins within one step. */
enum { GP_EVENT_BEGIN = 1, GP_EVENT_END = 2, GP_EVENT_TRIGGER_ENTER = 3, GP_EVENT_TRIGGER_EXIT = 4 };
typedef struct GpEvent {
	int kind, a, b, tile;
	float x, y, nx, ny, speed;
} GpEvent;
void gp_enable_events( GpWorld *w, int on );
int gp_poll_events( GpWorld *w, GpEvent *out, int max );
int gp_body_set_sensor( GpWorld *w, int body, int on ); /* 1, or 0 when the host linked no prowl_material kernels */

/* Box2D's per-body settings (kernels/prowl_material.rs). The first call switches the step to those kernels; a world
 * that never calls them steps exactly as before. Both return 1, or 0 when the host linked no prowl_material kernels.
 *   friction  Unity's units (0.4 default); a pair uses the geometric mean, a tile the same with gp_set_material's friction
 *   bounciness 0..1; a pair uses the larger; approaches slower than gp_set_bounce_threshold (default 1 unit/s) do not bounce
 *   density   scales the mass (mass is proportional to a circle's radius here; 1 is the plain engine)
 *   gravity_scale, drag  as Unity's Rigidbody2D
 *   category, mask, group  Box2D's: collide when each mask holds the other's category; a shared non-zero group
 *             collides when positive and never when negative. Circles of one body never collide with each other. */
int gp_body_set_material( GpWorld *w, int body, float friction, float bounciness, float density, float gravity_scale,
						  float drag );
int gp_body_set_filter( GpWorld *w, int body, unsigned category, unsigned mask, int group );
void gp_set_bounce_threshold( GpWorld *w, float speed );

void gp_step( GpWorld *w, float dt );

float gp_circle_x( const GpWorld *w, int circle );
float gp_circle_y( const GpWorld *w, int circle );
float gp_circle_radius( const GpWorld *w, int circle );
int gp_circle_count( const GpWorld *w );
int gp_circle_body( const GpWorld *w, int circle );

/* Ray cast (direction need not be unit). Returns 1 and fills *hit, or 0. ignore_body < 0: none. */
int gp_raycast( GpWorld *w, float ox, float oy, float dx, float dy, float distance, int mask, int ignore_body,
				GpHit *hit );
/* A circle (a body's circle) overlapping the disc: returns its body id (or -1 / not found: 0). */
int gp_overlap_circle( const GpWorld *w, float x, float y, float radius, int *body );
/* Tile solid at the world point, 1/0; also its index in *tile when non-null. */
int gp_tile_at( const GpWorld *w, float x, float y, int *tile );

/* Pairs of bodies whose circles touch (distance < r1 + r2 + slop) after the last step, each pair
 * once with a < b; and body-tile contacts as (body, tile). Returns how many were written. */
int gp_body_pairs( const GpWorld *w, int *a, int *b, int max );
int gp_tile_contacts( const GpWorld *w, int *body, int *tile, int max );

#ifdef __cplusplus
}
#endif
#endif
