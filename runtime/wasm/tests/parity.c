/* The scenario tests/test.mjs also runs in WebAssembly: both print the same lines. */
#include "gpuphys.h"
#include <math.h>
#include <stdio.h>
int main( void )
{
	GpWorld *w = gp_create( 512, 1024, 0, 0, 20, 10, 1.0f, 0, -18.0f );
	unsigned tiles[200] = { 0 };
	for ( int x = 0; x < 20; ++x )
		tiles[x] = 1;
	tiles[8 + 3 * 20] = tiles[9 + 3 * 20] = 1; /* a ledge */
	gp_set_tiles( w, 20, 10, tiles );
	gp_enable_events( w, 1 );
	int ball = gp_add_ball( w, 5, 6, 0.35f, 1.0f, 0, 1 / 60.0f );
	int box = gp_add_box( w, 10, 6, 1.5f, 1.0f, 0.3f, 0, 0, 1 / 60.0f, 1.0f );
	float hex[12];
	for ( int i = 0; i < 6; ++i )
		hex[2 * i] = 0.6f * cosf( 6.2831853f * i / 6 ), hex[2 * i + 1] = 0.6f * sinf( 6.2831853f * i / 6 );
	int poly = gp_add_polygon( w, 14, 6, 0.2f, hex, 6, 0, 0, 0, 1 / 60.0f, 1.0f );
	int rope = gp_add_rope( w, 3, 8, 7, 8, 10, 0.08f, 1, 0 );
	int bouncy = gp_add_ball( w, 16, 7, 0.35f, 0, 0, 1 / 60.0f );
	gp_body_set_material( w, bouncy, 0.4f, 0.8f, 1.0f, 1.0f, 0.0f );
	int plank = gp_add_box( w, 12.5f, 8, 2, 0.3f, 0, 0, 0, 1 / 60.0f, 1.0f );
	GpJointDef d = gp_joint_def( GP_JOINT_REVOLUTE, plank, -1, 11.5f, 8 );
	int hinge = gp_add_body_joint( w, &d );
	for ( int s = 0; s < 240; ++s )
		gp_step( w, 1 / 60.0f );
	int ids[7] = { ball, box, poly, rope, bouncy, plank, -1 };
	for ( int i = 0; i < 6; ++i )
	{
		float x, y, a;
		gp_body_pose( w, ids[i], &x, &y, &a );
		printf( "body %d %.3f %.3f %.3f\n", ids[i], x, y, a );
	}
	printf( "hinge %.3f\n", gp_joint_angle( w, hinge ) );
	GpEvent ev[64];
	printf( "events %d\n", gp_poll_events( w, ev, 64 ) );
	GpHit h;
	printf( "ray %d\n", gp_raycast( w, 2, 5, 0, -1, 100, 3, -1, &h ) );
	printf( "circles %d\n", gp_circle_count( w ) );
	return 0;
}
