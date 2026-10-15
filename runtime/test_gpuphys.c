/* A known-answer test of gpuphys.c's per-body material and filter (kernels/prowl_material.rs run as C).
 * python3 runtime/check.py builds and runs it. Exit status 0: all checks passed. */
#include "gpuphys.h"
#include <math.h>
#include <stdio.h>

static int failed;
static void check( const char *what, int ok )
{
	printf( "%s %s\n", ok ? "ok  " : "FAIL", what );
	if ( !ok )
		failed++;
}

static GpWorld *world( float gy )
{
	GpWorld *w = gp_create( 256, 512, 0, 0, 20, 10, 1.0f, 0, gy );
	unsigned tiles[20 * 10] = { 0 };
	for ( int x = 0; x < 20; ++x )
		tiles[x] = 1; /* the floor: its top is at y = 1 */
	gp_set_tiles( w, 20, 10, tiles );
	return w;
}

static float y_of( GpWorld *w, int body )
{
	float x, y, a;
	gp_body_pose( w, body, &x, &y, &a );
	return y;
}

/* the highest the ball gets after its first landing */
static float rebound( float bounciness, int use_material )
{
	GpWorld *w = world( -9.81f );
	int b = gp_add_ball( w, 10, 6, 0.5f, 0, 0, 1.0f / 60 );
	if ( use_material )
		gp_body_set_material( w, b, 0.4f, bounciness, 1.0f, 1.0f, 0.0f );
	float lowest = 100, top_after = 0;
	for ( int s = 0; s < 400; ++s )
	{
		gp_step( w, 1.0f / 60 );
		float y = y_of( w, b );
		if ( y < lowest )
			lowest = y;
		if ( lowest < 1.53f && y > top_after )
			top_after = y;
	}
	gp_destroy( w );
	return top_after - 1.5f; /* above the resting height (floor top 1 + radius 0.5) */
}


/* ---- events */
static int count_kind( const GpEvent *e, int n, int kind )
{
	int c = 0;
	for ( int i = 0; i < n; ++i )
		c += e[i].kind == kind;
	return c;
}

static void events_test( void )
{
	GpEvent ev[256];
	/* a ball lands on the floor: one BEGIN against the tiles, none while it rests, an END when it is lifted */
	{
		GpWorld *w = world( -9.81f );
		gp_enable_events( w, 1 );
		int b = gp_add_ball( w, 10, 4, 0.5f, 0, 0, 1.0f / 60 );
		int begins = 0, other = 0, n;
		GpEvent first = { 0 };
		for ( int s = 0; s < 240; ++s )
		{
			gp_step( w, 1.0f / 60 );
			n = gp_poll_events( w, ev, 256 );
			for ( int k = 0; k < n; ++k )
			{
				if ( ev[k].kind == GP_EVENT_BEGIN && !begins++ )
					first = ev[k];
				else
					other++;
			}
		}
		check( "landing on the floor gives exactly one BEGIN", begins == 1 && other == 0 );
		check( "it is against the tiles (b = -1, a tile index)", first.a == b && first.b == -1 && first.tile >= 0 );
		check( "the normal points down into the floor", fabsf( first.nx ) < 0.2f && first.ny < -0.9f );
		check( "the contact point is on the floor's top", fabsf( first.y - 1.0f ) < 0.15f && fabsf( first.x - 10 ) < 0.6f );
		printf( "     closing speed at the landing %.2f (the fall itself is ~7: it is read after the solver pushed them apart)\n", first.speed );
		check( "the closing speed is a fair fraction of the fall speed", first.speed > 3.0f && first.speed < 9.0f );
		gp_body_set_position( w, b, 10, 6 );
		gp_step( w, 1.0f / 60 );
		n = gp_poll_events( w, ev, 256 );
		check( "lifting it off gives an END", count_kind( ev, n, GP_EVENT_END ) == 1 && ev[0].b == -1 );
		gp_destroy( w );
	}
	/* two balls: one BEGIN for the pair, normal from the lower index (a) toward b, and an END when they part */
	{
		GpWorld *w = world( -9.81f );
		gp_enable_events( w, 1 );
		int low = gp_add_ball( w, 10, 1.5f, 0.5f, 0, 0, 1.0f / 60 ), high = gp_add_ball( w, 10, 4, 0.5f, 0, 0, 1.0f / 60 );
		int pair_begins = 0, n;
		GpEvent pb = { 0 };
		for ( int s = 0; s < 240; ++s )
		{
			gp_step( w, 1.0f / 60 );
			n = gp_poll_events( w, ev, 256 );
			for ( int k = 0; k < n; ++k )
				if ( ev[k].kind == GP_EVENT_BEGIN && ev[k].b == high && !pair_begins++ )
					pb = ev[k];
		}
		check( "two balls: one BEGIN for the pair", pair_begins == 1 && pb.a == low );
		check( "its normal runs from the lower body up to the upper", pb.ny > 0.9f );
		gp_body_set_position( w, high, 14, 6 );
		gp_step( w, 1.0f / 60 );
		n = gp_poll_events( w, ev, 256 );
		int ended = 0;
		for ( int k = 0; k < n; ++k )
			ended += ev[k].kind == GP_EVENT_END && ev[k].a == low && ev[k].b == high;
		check( "moving one away gives an END for the pair", ended == 1 );
		gp_destroy( w );
	}
	/* filtered pairs make no events; a sensor is entered and left and never pushes */
	{
		GpWorld *w = world( -9.81f );
		gp_enable_events( w, 1 );
		int low = gp_add_ball( w, 10, 1.5f, 0.5f, 0, 0, 1.0f / 60 ), high = gp_add_ball( w, 10, 4, 0.5f, 0, 0, 1.0f / 60 );
		gp_body_set_filter( w, low, 1u, 1u, 0 );
		gp_body_set_filter( w, high, 2u, 2u, 0 );
		int pair = 0, n;
		for ( int s = 0; s < 240; ++s )
		{
			gp_step( w, 1.0f / 60 );
			n = gp_poll_events( w, ev, 256 );
			for ( int k = 0; k < n; ++k )
				pair += ev[k].b == high || ev[k].a == high && ev[k].b >= 0;
		}
		check( "a pair that the masks exclude makes no event", pair == 0 );
		gp_destroy( w );
	}
	{
		GpWorld *w = world( -9.81f );
		gp_enable_events( w, 1 );
		int zone = gp_add_ball( w, 10, 4, 1.0f, 0, 0, 1.0f / 60 ), ball = gp_add_ball( w, 10, 8, 0.5f, 0, 0, 1.0f / 60 );
		gp_body_set_material( w, zone, 0.4f, 0, 1, 0.0f, 0 ); /* no gravity: it stays where it is */
		check( "a body can become a sensor", gp_body_set_sensor( w, zone, 1 ) == 1 );
		int enter = 0, exit_ = 0, n;
		GpEvent in = { 0 };
		for ( int s = 0; s < 240; ++s )
		{
			gp_step( w, 1.0f / 60 );
			n = gp_poll_events( w, ev, 256 );
			for ( int k = 0; k < n; ++k )
			{
				if ( ev[k].kind == GP_EVENT_TRIGGER_ENTER && !enter++ )
					in = ev[k];
				exit_ += ev[k].kind == GP_EVENT_TRIGGER_EXIT;
			}
		}
		check( "a ball falling through a sensor enters and leaves it once", enter == 1 && exit_ == 1 );
		check( "the sensor is a, the ball is b", in.a == zone && in.b == ball );
		check( "the ball was not pushed by the sensor and lands on the floor", fabsf( y_of( w, ball ) - 1.5f ) < 0.05f );
		check( "the sensor did not move", fabsf( y_of( w, zone ) - 4.0f ) < 0.01f );
		gp_destroy( w );
	}
}

/* ---- joints */
static void pose( GpWorld *w, int body, float *x, float *y, float *a )
{
	gp_body_pose( w, body, x, y, a );
}

static GpWorld *space( void ) /* no gravity, no floor in the way */
{
	return world( 0.0f );
}

/* B is a 2 x 0.5 plank hinged at its left end (10.5, 8) to a kinematic 1 x 1 block */
static int hinge_scene( GpWorld *w, int type, int *plank, int limit )
{
	int block = gp_add_box( w, 10, 8, 1, 1, 0, 0, 0, 1.0f / 60, 1.0f );
	*plank = gp_add_box( w, 11.5f, 8, 2, 0.5f, 0, 0, 0, 1.0f / 60, 1.0f );
	gp_body_set_kinematic( w, block, 1 );
	GpJointDef d = gp_joint_def( type, block, *plank, 10.5f, 8.0f );
	d.enable_limit = limit, d.lower = 0.0f, d.upper = 0.3f;
	return gp_add_body_joint( w, &d );
}

static void joints_test( void )
{
	const float dt = 1.0f / 60;
	/* a pendulum: a ball pinned to the world keeps its distance and swings */
	{
		GpWorld *w = world( -9.81f );
		int ball = gp_add_ball( w, 12, 8, 0.25f, 0, 0, dt );
		GpJointDef d = gp_joint_def( GP_JOINT_DISTANCE, ball, -1, 12, 8 ); /* a rod from the ball's centre to a pin */
		d.bx = 10, d.by = 8;
		check( "a distance joint to the world is made", gp_add_body_joint( w, &d ) >= 0 );
		float lo = 99, hi = 0, lowest = 99;
		for ( int s = 0; s < 180; ++s )
		{
			gp_step( w, dt );
			float x, y, a;
			pose( w, ball, &x, &y, &a );
			float r = hypotf( x - 10, y - 8 );
			if ( r < lo )
				lo = r;
			if ( r > hi )
				hi = r;
			if ( y < lowest )
				lowest = y;
		}
		printf( "     pendulum: length %.3f..%.3f (2 at rest), lowest y %.2f\n", lo, hi, lowest );
		check( "a pendulum keeps its length", lo > 1.9f && hi < 2.1f );
		check( "...and swings down", lowest < 6.3f );
		gp_destroy( w );
	}
	/* a hinge: the plank stays on its pin; limits hold it up; a weld does not turn at all; a motor turns it */
	{
		float ang[4];
		float pin_err = 0;
		for ( int c = 0; c < 4; ++c )
		{
			GpWorld *w = world( c == 3 ? 0.0f : -9.81f );
			int plank, j = hinge_scene( w, c == 2 ? GP_JOINT_WELD : GP_JOINT_REVOLUTE, &plank, c == 1 );
			if ( c == 3 )
				gp_joint_set_motor( w, j, 1, 2.0f, 1.0e9f );
			for ( int s = 0; s < ( c == 3 ? 60 : 90 ); ++s )
			{
				gp_step( w, dt );
				if ( c == 0 )
				{
					float x, y, a;
					pose( w, plank, &x, &y, &a );
					float ex = x - cosf( a ) - 10.5f, ey = y - sinf( a ) - 8.0f, e = hypotf( ex, ey );
					if ( e > pin_err )
						pin_err = e;
				}
			}
			ang[c] = gp_joint_angle( w, j );
			gp_destroy( w );
		}
		printf( "     plank angle after 1.5 s: free %.2f, limited %.2f, welded %.2f; motor 1 s at 2 rad/s: %.2f; pin error %.3f\n",
				ang[0], ang[1], ang[2], ang[3], pin_err );
		check( "a free hinge swings the plank down", ang[0] < -1.0f );
		check( "the plank stays on its pin", pin_err < 0.1f );
		check( "an angle limit holds the plank up", fabsf( ang[1] ) < 0.15f );
		check( "a weld does not turn", fabsf( ang[2] ) < 0.1f );
		check( "a revolute motor turns the plank at about its speed", ang[3] > 1.4f && ang[3] < 2.6f );
	}
	/* prismatic: slides on its axis to the limit; a motor drives it; wheel: a spring that sags and settles */
	{
		GpWorld *w = world( -9.81f );
		int block = gp_add_box( w, 10, 8, 1, 1, 0, 0, 0, dt, 1.0f ), slider = gp_add_box( w, 10, 6.5f, 1, 1, 0, 0, 0, dt, 1.0f );
		gp_body_set_kinematic( w, block, 1 );
		GpJointDef d = gp_joint_def( GP_JOINT_PRISMATIC, block, slider, 10, 6.5f );
		d.enable_limit = 1, d.lower = -1.0f, d.upper = 0.5f;
		int j = gp_add_body_joint( w, &d );
		for ( int s = 0; s < 120; ++s )
			gp_step( w, dt );
		float x, y, a;
		pose( w, slider, &x, &y, &a );
		printf( "     slider: translation %.3f (limit -1), x %.3f, angle %.3f\n", gp_joint_translation( w, j ), x, a );
		check( "a prismatic joint stops at its lower limit", fabsf( gp_joint_translation( w, j ) + 1.0f ) < 0.12f );
		check( "...and stays on its line, unturned", fabsf( x - 10 ) < 0.05f && fabsf( a ) < 0.05f );
		gp_destroy( w );
	}
	{
		GpWorld *w = space();
		int block = gp_add_box( w, 10, 8, 1, 1, 0, 0, 0, dt, 1.0f ), slider = gp_add_box( w, 10, 6.5f, 1, 1, 0, 0, 0, dt, 1.0f );
		gp_body_set_kinematic( w, block, 1 );
		GpJointDef d = gp_joint_def( GP_JOINT_PRISMATIC, block, slider, 10, 6.5f );
		d.enable_motor = 1, d.motor_speed = 2.0f, d.max_motor = 1.0e9f;
		int j = gp_add_body_joint( w, &d );
		for ( int s = 0; s < 60; ++s )
			gp_step( w, dt );
		printf( "     slider motor, 1 s at 2 units/s: translation %.3f\n", gp_joint_translation( w, j ) );
		check( "a prismatic motor slides at about its speed", fabsf( gp_joint_translation( w, j ) - 2.0f ) < 0.4f );
		gp_destroy( w );
	}
	{
		GpWorld *w = world( -9.81f );
		int block = gp_add_box( w, 10, 8, 1, 1, 0, 0, 0, dt, 1.0f ), wheel = gp_add_box( w, 10, 6.5f, 1, 1, 0, 0, 0, dt, 1.0f );
		gp_body_set_kinematic( w, block, 1 );
		GpJointDef d = gp_joint_def( GP_JOINT_WHEEL, block, wheel, 10, 6.5f );
		d.frequency = 1.0f, d.damping_ratio = 0.7f;
		int j = gp_add_body_joint( w, &d );
		float before = 0;
		for ( int s = 0; s < 360; ++s )
		{
			if ( s == 359 )
				before = gp_joint_translation( w, j );
			gp_step( w, dt );
		}
		float t = gp_joint_translation( w, j ), x, y, a;
		pose( w, wheel, &x, &y, &a );
		printf( "     suspension at 1 Hz: sags to %.3f (g / omega^2 = 0.25), last step moved %.4f\n", t, fabsf( t - before ) );
		check( "a wheel's suspension sags under gravity", t < -0.1f && t > -0.5f );
		check( "...settles, and stays on its axis", fabsf( t - before ) < 0.003f && fabsf( x - 10 ) < 0.05f );
		gp_destroy( w );
	}
	/* distance: a rod keeps its length, a rope stops a fall at its length, a spring stretches */
	{
		GpWorld *w = world( -9.81f );
		int a = gp_add_ball( w, 10, 8, 0.25f, 0, 0, dt ), b = gp_add_ball( w, 13, 8, 0.25f, 0, 0, dt );
		gp_body_set_kinematic( w, a, 1 );
		GpJointDef d = gp_joint_def( GP_JOINT_DISTANCE, a, b, 10, 8 );
		d.bx = 13, d.by = 8;
		gp_add_body_joint( w, &d );
		float lo = 99, hi = 0;
		for ( int s = 0; s < 120; ++s )
		{
			gp_step( w, dt );
			float x, y, an;
			pose( w, b, &x, &y, &an );
			float r = hypotf( x - 10, y - 8 );
			lo = r < lo ? r : lo, hi = r > hi ? r : hi;
		}
		check( "a distance joint is a rod: the length is kept", lo > 2.9f && hi < 3.1f );
		gp_destroy( w );
	}
	{
		GpWorld *w = world( -9.81f );
		int a = gp_add_ball( w, 10, 8, 0.25f, 0, 0, dt ), b = gp_add_ball( w, 10, 7, 0.25f, 0, 0, dt );
		gp_body_set_kinematic( w, a, 1 );
		GpJointDef d = gp_joint_def( GP_JOINT_DISTANCE, a, b, 10, 8 );
		d.bx = 10, d.by = 7, d.enable_limit = 1, d.lower = 0.0f, d.upper = 3.0f; /* a rope: at most 3 long */
		gp_add_body_joint( w, &d );
		float top = 99;
		for ( int s = 0; s < 120; ++s )
		{
			gp_step( w, dt );
			float x, y, an;
			pose( w, b, &x, &y, &an );
			top = y < top ? y : top;
		}
		printf( "     rope of 3: the ball hangs at y %.3f (5 is the end of the rope)\n", top );
		check( "a rope joint lets the ball fall to its length and holds it", top > 4.85f && top < 5.2f );
		gp_destroy( w );
	}
	/* mouse: pulls a ball toward a target; a weak force barely moves it */
	for ( int weak = 0; weak < 2; ++weak )
	{
		GpWorld *w = space();
		int ball = gp_add_ball( w, 10, 6, 0.5f, 0, 0, dt );
		GpJointDef d = gp_joint_def( GP_JOINT_MOUSE, ball, -1, 10, 6 );
		d.frequency = 5.0f, d.damping_ratio = 0.7f, d.max_force = weak ? 0.05f : 0.0f;
		int j = gp_add_body_joint( w, &d );
		gp_joint_set_target( w, j, 15, 6 );
		for ( int s = 0; s < 120; ++s )
			gp_step( w, dt );
		float x, y, a;
		pose( w, ball, &x, &y, &a );
		if ( !weak )
			check( "a mouse joint brings the ball to the target", fabsf( x - 15 ) < 0.3f );
		else
			check( "a mouse joint's force limit holds it back", x > 10.0f && x < 13.0f );
		gp_destroy( w );
	}
	/* joined bodies do not collide with each other */
	{
		GpWorld *w = space();
		int a = gp_add_ball( w, 10, 6, 0.5f, 0, 0, dt ), b = gp_add_ball( w, 10.4f, 6, 0.5f, 0, 0, dt );
		gp_body_set_kinematic( w, a, 1 );
		GpJointDef d = gp_joint_def( GP_JOINT_DISTANCE, a, b, 10, 6 );
		d.bx = 10.4f, d.by = 6;
		gp_add_body_joint( w, &d );
		for ( int s = 0; s < 60; ++s )
			gp_step( w, dt );
		float x, y, an;
		pose( w, b, &x, &y, &an );
		check( "overlapping bodies joined together are not pushed apart", fabsf( x - 10.4f ) < 0.05f );
		gp_destroy( w );
	}
}

/* ---- rotation */
static float body_speed( GpWorld *w, int body )
{
	float vx, vy;
	gp_body_velocity( w, body, 1.0f / 60, &vx, &vy );
	return hypotf( vx, vy );
}

static float pair_distance( GpWorld *w, int body, int i, int j )
{
	int a = gp_body_circle( w, body, i ), b = gp_body_circle( w, body, j );
	return hypotf( gp_circle_x( w, a ) - gp_circle_x( w, b ), gp_circle_y( w, a ) - gp_circle_y( w, b ) );
}

static int regular( float *v, int n, float r )
{
	for ( int i = 0; i < n; ++i )
		v[2 * i] = r * cosf( 6.2831853f * i / n ), v[2 * i + 1] = r * sinf( 6.2831853f * i / n );
	return n;
}

static void rotation_test( void )
{
	const float dt = 1.0f / 60;
	/* balls spin: freely, and rolling on the floor */
	{
		GpWorld *w = space();
		int b = gp_add_ball( w, 10, 6, 0.5f, 0, 0, dt );
		gp_body_set_angular_velocity( w, b, dt, 5.0f );
		for ( int s = 0; s < 60; ++s )
			gp_step( w, dt );
		float x, y, a;
		gp_body_pose( w, b, &x, &y, &a );
		float om = gp_body_angular_velocity( w, b, dt );
		printf( "     free spin at 5 rad/s for 1 s: angle %.2f, speed %.2f\n", a, om );
		check( "a ball keeps spinning in the air", fabsf( a - 4.9f ) < 0.3f && fabsf( om - 4.85f ) < 0.3f );
		gp_body_set_angle( w, b, 1.0f );
		gp_body_pose( w, b, &x, &y, &a );
		check( "gp_body_set_angle turns it", fabsf( a - 1.0f ) < 1e-4f );
		gp_destroy( w );
	}
	{
		GpWorld *w = world( -9.81f );
		int b = gp_add_ball( w, 5, 1.5f, 0.5f, 3.0f, 0, dt );
		gp_body_set_material( w, b, 0.4f, 0.0f, 1.0f, 1.0f, 0.0f ); /* rolling without a slowdown needs the material path */
		for ( int s = 0; s < 90; ++s )
			gp_step( w, dt );
		float vx, vy, x, y, a;
		gp_body_velocity( w, b, dt, &vx, &vy );
		float om = gp_body_angular_velocity( w, b, dt );
		gp_body_pose( w, b, &x, &y, &a );
		printf( "     rolling: speed %.2f, spin %.2f (rolling would be %.2f), angle %.2f\n", vx, om, -vx / 0.5f, a );
		check( "a ball rolls on the floor: the spin matches the speed", vx > 0.5f && fabsf( om + vx / 0.5f ) < 0.25f * vx / 0.5f );
		check( "...and its angle keeps turning", a < -3.0f );
		gp_destroy( w );
	}
	/* a ball pinned off-centre now pivots and can be turned by a motor */
	{
		GpWorld *w = world( -9.81f );
		int ball = gp_add_ball( w, 12, 8, 0.25f, 0, 0, dt );
		GpJointDef d = gp_joint_def( GP_JOINT_REVOLUTE, ball, -1, 10, 8 );
		gp_add_body_joint( w, &d );
		float lowest = 99, worst = 0, ang_low = 0;
		for ( int s = 0; s < 150; ++s )
		{
			gp_step( w, dt );
			float x, y, a;
			gp_body_pose( w, ball, &x, &y, &a );
			worst = fmaxf( worst, fabsf( hypotf( x - 10, y - 8 ) - 2.0f ) );
			if ( y < lowest )
				lowest = y, ang_low = a;
		}
		printf( "     ball pendulum on a revolute joint: lowest y %.2f, arm error %.3f, angle at the bottom %.2f\n", lowest, worst, ang_low );
		check( "a ball pinned off-centre swings about the pin", lowest < 6.2f && worst < 0.05f );
		check( "...turning with its arm", ang_low < -1.0f );
		gp_destroy( w );
	}
	{
		GpWorld *w = space();
		int hub = gp_add_ball( w, 10, 8, 0.25f, 0, 0, dt ), ball = gp_add_ball( w, 10, 8, 0.5f, 0, 0, dt );
		gp_body_set_kinematic( w, hub, 1 );
		GpJointDef d = gp_joint_def( GP_JOINT_REVOLUTE, hub, ball, 10, 8 ); /* a wheel on its axle */
		d.enable_motor = 1, d.motor_speed = 2.0f, d.max_motor = 1.0e9f;
		int j = gp_add_body_joint( w, &d );
		for ( int s = 0; s < 60; ++s )
			gp_step( w, dt );
		float x, y, a;
		gp_body_pose( w, ball, &x, &y, &a );
		printf( "     a motor on a wheel: joint angle %.2f after 1 s at 2 rad/s, off its axle by %.4f\n", gp_joint_angle( w, j ), hypotf( x - 10, y - 8 ) );
		check( "a revolute motor spins a ball on its axle", fabsf( gp_joint_angle( w, j ) - 2.0f ) < 0.3f && hypotf( x - 10, y - 8 ) < 0.02f );
		gp_destroy( w );
	}
	/* polygons */
	{
		float tri[6] = { -1.0f, -0.5f, 1.0f, -0.5f, 0.0f, 1.0f };
		GpWorld *w = world( -9.81f );
		int b = gp_add_polygon( w, 10, 5, 0.3f, tri, 3, 0.0f, 0, 0, dt, 1.0f );
		check( "a triangle is made of circles", b >= 0 && gp_body_circles( w, b ) >= 4 );
		float ox, oy;
		gp_body_origin( w, b, &ox, &oy );
		check( "gp_body_origin gives back the origin it was made at", fabsf( ox - 10 ) < 0.01f && fabsf( oy - 5 ) < 0.01f );
		float d0 = pair_distance( w, b, 0, 1 ), d1 = pair_distance( w, b, 1, 2 ), worst = 0;
		for ( int s = 0; s < 360; ++s )
		{
			gp_step( w, dt );
			worst = fmaxf( worst, fabsf( pair_distance( w, b, 0, 1 ) - d0 ) / d0 );
			worst = fmaxf( worst, fabsf( pair_distance( w, b, 1, 2 ) - d1 ) / d1 );
		}
		float low = 99;
		for ( int k = 0; k < gp_body_circles( w, b ); ++k )
		{
			int c = gp_body_circle( w, b, k );
			low = fminf( low, gp_circle_y( w, c ) - gp_circle_radius( w, c ) );
		}
		float x, y, a;
		gp_body_pose( w, b, &x, &y, &a );
		printf( "     triangle (%d circles): rest speed %.3f, spin %.3f, lowest point %.3f (floor at 1), shape drift %.4f, angle %.2f\n",
				gp_body_circles( w, b ), body_speed( w, b ), gp_body_angular_velocity( w, b, dt ), low, worst, a );
		check( "a polygon keeps its shape while it tumbles and lands", worst < 0.02f );
		check( "...comes to rest on the floor", body_speed( w, b ) < 0.12f && fabsf( gp_body_angular_velocity( w, b, dt ) ) < 0.3f && low > 0.93f && low < 1.08f );
		gp_destroy( w );
	}
	{
		float sq[8] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f };
		GpWorld *w = world( -9.81f );
		int b = gp_add_polygon( w, 10, 5, 0.2f, sq, 4, 0.0f, 0, 0, dt, 1.0f );
		for ( int s = 0; s < 480; ++s )
			gp_step( w, dt );
		float x, y, a;
		gp_body_pose( w, b, &x, &y, &a );
		float flat = fabsf( remainderf( a, 1.5707963f ) );
		printf( "     square polygon: angle %.3f (flat is a multiple of 1.571), centre y %.3f\n", a, y );
		check( "a square polygon settles flat like a box", flat < 0.12f && y > 1.35f && y < 1.7f );
		gp_destroy( w );
	}
	{
		float sq[8] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f };
		GpWorld *w = space();
		int b = gp_add_polygon( w, 10, 6, 0.0f, sq, 4, 0.0f, 0, 0, dt, 1.0f );
		float d0 = pair_distance( w, b, 0, 1 ), x0, y0, a0;
		gp_body_pose( w, b, &x0, &y0, &a0 );
		gp_body_set_angular_velocity( w, b, dt, 3.0f );
		for ( int s = 0; s < 60; ++s )
			gp_step( w, dt );
		float x, y, a;
		gp_body_pose( w, b, &x, &y, &a );
		printf( "     polygon at 3 rad/s for 1 s: turned %.2f, spin %.2f, drift %.3f, shape %.4f\n", a - a0, gp_body_angular_velocity( w, b, dt ),
				hypotf( x - x0, y - y0 ), fabsf( pair_distance( w, b, 0, 1 ) - d0 ) / d0 );
		check( "a polygon turns at the spin it is given (less the world's damping)", fabsf( ( a - a0 ) - 2.9f ) < 0.25f && fabsf( gp_body_angular_velocity( w, b, dt ) - 2.7f ) < 0.35f );
		check( "...about its centre, rigidly", hypotf( x - x0, y - y0 ) < 0.05f && fabsf( pair_distance( w, b, 0, 1 ) - d0 ) / d0 < 0.02f );
		gp_destroy( w );
	}
	{
		float plank[8] = { -1.0f, -0.25f, 1.0f, -0.25f, 1.0f, 0.25f, -1.0f, 0.25f };
		GpWorld *w = world( -9.81f );
		int block = gp_add_box( w, 10, 8, 1, 1, 0, 0, 0, dt, 1.0f ), p = gp_add_polygon( w, 11.5f, 8, 0.0f, plank, 4, 0.0f, 0, 0, dt, 1.0f );
		gp_body_set_kinematic( w, block, 1 );
		GpJointDef d = gp_joint_def( GP_JOINT_REVOLUTE, block, p, 10.5f, 8.0f );
		int j = gp_add_body_joint( w, &d );
		float pin = 0;
		for ( int s = 0; s < 90; ++s )
		{
			gp_step( w, dt );
			float x, y, a;
			gp_body_pose( w, p, &x, &y, &a );
			/* the hinge is the plank's left end: -1 along its own axis from the centre */
			pin = fmaxf( pin, hypotf( x - cosf( a ) - 10.5f, y - sinf( a ) - 8.0f ) );
		}
		printf( "     a polygon plank on a hinge: angle %.2f after 1.5 s, pin error %.3f\n", gp_joint_angle( w, j ), pin );
		check( "a polygon hangs from a hinge and swings", gp_joint_angle( w, j ) < -1.0f && pin < 0.12f );
		gp_destroy( w );
	}
	{
		float hex[12];
		regular( hex, 6, 0.6f );
		GpWorld *w = world( -9.81f );
		int b = gp_add_polygon( w, 4, 1.7f, 0.0f, hex, 6, 0.0f, 3.0f, 0, dt, 1.0f );
		gp_body_set_material( w, b, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f ); /* full friction */
		float x0, y0, a0, x, y, a;
		gp_body_pose( w, b, &x0, &y0, &a0 );
		for ( int s = 0; s < 900; ++s )
			gp_step( w, dt );
		float rx, ry, ra;
		gp_body_pose( w, b, &rx, &ry, &ra );
		for ( int k = 0; k < 30; ++k )
			gp_step( w, dt );
		float drift = 0;
		{
			float qx, qy, qa;
			gp_body_pose( w, b, &qx, &qy, &qa );
			drift = hypotf( qx - rx, qy - ry ) + fabsf( qa - ra );
		}
		gp_body_pose( w, b, &x, &y, &a );
		/* vertex 0 sits at angle 0, so the flat edges are at pi/6 + k*pi/3 */
		float flat = fabsf( remainderf( a - 0.5235988f, 1.0471976f ) );
		printf( "     hexagon thrown along the floor: travelled %.2f, turned %.2f rad, rests at %.2f off a flat edge, drift over 0.5 s %.3f\n", x - x0, a - a0, flat, drift );
		check( "a hexagon thrown along the floor tips over on its friction", x - x0 > 1.0f && fabsf( a - a0 ) > 0.5f );
		check( "...and comes to rest on a flat edge", flat < 0.1f && drift < 0.05f );
		gp_destroy( w );
	}
}

static float x_of( GpWorld *w, int body )
{
	float x, y, a;
	gp_body_pose( w, body, &x, &y, &a );
	return x;
}

/* Bezier colliders (kernels/prowl_curves.rs) */
static void curves_test( void )
{
	/* a thin bowl: a ball dropped inside it settles at the lowest point, r above the curve */
	{
		GpWorld *w = gp_create( 256, 512, 0, 0, 20, 10, 1.0f, 0, -9.81f );
		check( "a curve is accepted", gp_add_curve( w, 4, 6, 10, 2, 16, 6, 0.4f, 0, 0 ) == 0 );
		int b = gp_add_ball( w, 8, 8, 0.5f, 0, 0, 1.0f / 60 );
		for ( int s = 0; s < 600; ++s )
			gp_step( w, 1.0f / 60 );
		float x = x_of( w, b ), y = y_of( w, b );
		printf( "     bowl: rests at (%.2f, %.2f)\n", x, y );
		check( "a ball comes to rest in the bowl", fabsf( x - 10 ) < 1.2f && y > 4.45f && y < 4.75f );
		gp_destroy( w );
	}
	/* the same bowl from underneath: a ball thrown up at it is stopped by the thin wall */
	{
		GpWorld *w = gp_create( 256, 512, 0, 0, 20, 10, 1.0f, 0, 0 );
		gp_add_curve( w, 4, 6, 10, 2, 16, 6, 0.4f, 0, 0 );
		int b = gp_add_ball( w, 10, 0.8f, 0.4f, 0, 6.0f, 1.0f / 60 );
		for ( int s = 0; s < 120; ++s )
			gp_step( w, 1.0f / 60 );
		check( "a thin wall stops a ball from below", y_of( w, b ) < 3.75f && y_of( w, b ) > 2.0f );
		gp_destroy( w );
	}
	/* a solid hill on the floor: balls dropped on its flanks roll off, and none ends up inside or on top */
	{
		GpWorld *w = world( -9.81f );
		gp_add_curve( w, 2, 1, 6, 9, 10, 1, 0.4f, 0, -1 );     /* apex at (6, 5) */
		const float xs[6] = { 3.5f, 4.5f, 5.0f, 7.0f, 7.5f, 8.5f };
		int b[6];
		for ( int i = 0; i < 6; ++i )
			b[i] = gp_add_ball( w, xs[i], 9.0f - 1.2f * ( i % 3 ), 0.4f, 0, 0, 1.0f / 60 );
		for ( int s = 0; s < 1500; ++s )
			gp_step( w, 1.0f / 60 );
		int low = 0, inside = 0;
		for ( int i = 0; i < 6; ++i )
		{
			float x = x_of( w, b[i] ), y = y_of( w, b[i] );
			low += y < 4.0f;
			float t = ( x - 2 ) / 8, top = 1 + 16 * t * ( 1 - t );
			inside += x > 2 && x < 10 && y < top - 0.2f;
		}
		printf( "     hill: %d of 6 balls came down, %d inside\n", low, inside );
		check( "balls roll down a solid hill", low == 6 );
		check( "no ball ends up inside the solid hill", inside == 0 );
		gp_destroy( w );
	}
	/* bounciness and friction of a curve */
	for ( int e = 0; e < 2; ++e )
	{
		GpWorld *w = gp_create( 256, 512, 0, 0, 20, 10, 1.0f, 0, -9.81f );
		gp_add_curve( w, 2, 3, 10, 3, 18, 3, 0.4f, e ? 0.8f : 0.0f, 0 );
		int b = gp_add_ball( w, 10, 8, 0.5f, 0, 0, 1.0f / 60 );
		float lowest = 100, top = 0;
		for ( int s = 0; s < 400; ++s )
		{
			gp_step( w, 1.0f / 60 );
			float y = y_of( w, b );
			if ( y < lowest )
				lowest = y;
			if ( y > lowest + 0.2f && y > top )
				top = y;
		}
		printf( "     curve bounciness %.1f: rebound to %.2f\n", e ? 0.8f : 0.0f, top );
		check( e ? "a bouncy curve throws the ball back up" : "a dead curve does not", e ? top > 4.0f : top < 3.9f );
		gp_destroy( w );
	}
	{
		float reach[2];
		for ( int f = 0; f < 2; ++f )
		{
			GpWorld *w = gp_create( 256, 512, 0, 0, 40, 10, 1.0f, 0, -9.81f );
			gp_add_curve( w, 1, 3, 20, 3, 39, 3, f ? 2.0f : 0.0f, 0, 0 );
			int b = gp_add_ball( w, 2, 3.5f, 0.5f, 8.0f, 0, 1.0f / 60 );
			for ( int s = 0; s < 90; ++s )
				gp_step( w, 1.0f / 60 );
			reach[f] = x_of( w, b );
			gp_destroy( w );
		}
		printf( "     sliding 1.5 s from x=2 at 8 units/s: frictionless x %.2f, friction 2 x %.2f\n", reach[0], reach[1] );
		check( "curve friction slows a sliding ball", reach[1] < reach[0] - 0.3f );
	}
}

int main( void )
{
	setvbuf( stdout, 0, _IONBF, 0 );
	/* material mode with defaults is the plain engine for a single ball on the floor */
	{
		GpWorld *a = world( -9.81f ), *b = world( -9.81f );
		int ba = gp_add_ball( a, 10, 6, 0.5f, 1.0f, 0, 1.0f / 60 ), bb = gp_add_ball( b, 10, 6, 0.5f, 1.0f, 0, 1.0f / 60 );
		int ok = gp_body_set_filter( b, bb, 1u, 0xFFFFFFFFu, 0 );
		check( "the prowl_material kernels are linked", ok == 1 );
		float worst = 0;
		for ( int s = 0; s < 240; ++s )
		{
			gp_step( a, 1.0f / 60 );
			gp_step( b, 1.0f / 60 );
			float d = fabsf( y_of( a, ba ) - y_of( b, bb ) );
			if ( d > worst )
				worst = d;
		}
		check( "default material reproduces the plain step on a floor", worst < 2e-3f );
		gp_destroy( a ), gp_destroy( b );
	}
	/* bounciness */
	{
		float soft = rebound( 0.0f, 1 ), bouncy = rebound( 0.8f, 1 );
		printf( "     rebound height: bounciness 0 -> %.3f, 0.8 -> %.3f\n", soft, bouncy );
		check( "a bouncy ball rebounds clearly higher than a dead one", bouncy > 0.5f && bouncy > 4.0f * soft );
	}
	/* gravity scale and drag */
	{
		GpWorld *w = world( -9.81f );
		int still = gp_add_ball( w, 4, 6, 0.5f, 0, 0, 1.0f / 60 ), up = gp_add_ball( w, 8, 6, 0.5f, 0, 0, 1.0f / 60 );
		int fall = gp_add_ball( w, 12, 6, 0.5f, 0, 0, 1.0f / 60 );
		gp_body_set_material( w, still, 0.4f, 0, 1, 0.0f, 0 );
		gp_body_set_material( w, up, 0.4f, 0, 1, -1.0f, 0 );
		gp_body_set_material( w, fall, 0.4f, 0, 1, 1.0f, 0 );
		for ( int s = 0; s < 30; ++s )
			gp_step( w, 1.0f / 60 );
		check( "gravity scale 0 does not fall", fabsf( y_of( w, still ) - 6 ) < 1e-3f );
		check( "gravity scale -1 rises", y_of( w, up ) > 6.3f );
		check( "gravity scale 1 falls", y_of( w, fall ) < 5.8f );
		gp_destroy( w );
	}
	{
		GpWorld *w = world( 0 );
		int free_ = gp_add_ball( w, 3, 5, 0.5f, 6, 0, 1.0f / 60 ), slow = gp_add_ball( w, 3, 7, 0.5f, 6, 0, 1.0f / 60 );
		gp_body_set_material( w, free_, 0.4f, 0, 1, 1, 0.0f );
		gp_body_set_material( w, slow, 0.4f, 0, 1, 1, 3.0f );
		for ( int s = 0; s < 60; ++s )
			gp_step( w, 1.0f / 60 );
		float x0, y0, a0, x1, y1, a1;
		gp_body_pose( w, free_, &x0, &y0, &a0 );
		gp_body_pose( w, slow, &x1, &y1, &a1 );
		printf( "     after 1 s at 6 units/s: free x %.2f, drag 3 x %.2f\n", x0, x1 );
		check( "drag slows a ball down", x1 < x0 - 1.0f && x0 > 8.5f );
		gp_destroy( w );
	}
	/* filtering: a ball dropped on another rests on it, unless the masks let it pass */
	for ( int pass = 0; pass < 2; ++pass )
	{
		GpWorld *w = world( -9.81f );
		int low = gp_add_ball( w, 10, 1.5f, 0.5f, 0, 0, 1.0f / 60 ), high = gp_add_ball( w, 10, 4, 0.5f, 0, 0, 1.0f / 60 );
		gp_body_set_filter( w, low, 1u, 0xFFFFFFFFu, 0 );
		gp_body_set_filter( w, high, pass ? 2u : 1u, pass ? 2u : 0xFFFFFFFFu, 0 ); /* pass 1: only meets category 2 */
		for ( int s = 0; s < 300; ++s )
			gp_step( w, 1.0f / 60 );
		float y = y_of( w, high );
		if ( !pass )
			check( "a ball comes to rest on another", y > 2.3f );
		else
			check( "masks that exclude each other let a ball fall through", y < 1.7f );
		gp_destroy( w );
	}
	/* a positive shared group collides although the masks exclude each other; a negative one never does */
	for ( int g = 0; g < 2; ++g )
	{
		GpWorld *w = world( -9.81f );
		int low = gp_add_ball( w, 10, 1.5f, 0.5f, 0, 0, 1.0f / 60 ), high = gp_add_ball( w, 10, 4, 0.5f, 0, 0, 1.0f / 60 );
		gp_body_set_filter( w, low, 1u, 0u, g ? -4 : 4 );
		gp_body_set_filter( w, high, 1u, 0u, g ? -4 : 4 );
		for ( int s = 0; s < 300; ++s )
			gp_step( w, 1.0f / 60 );
		float y = y_of( w, high );
		check( g ? "a shared negative group never collides" : "a shared positive group collides despite the masks",
			   g ? y < 1.7f : y > 2.3f );
		gp_destroy( w );
	}
	events_test();
	joints_test();
	rotation_test();
	curves_test();
	printf( failed ? "%d FAILED\n" : "all checks passed\n", failed );
	return failed ? 1 : 0;
}
