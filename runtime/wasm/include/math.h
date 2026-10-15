/* Just what gpuphys.c needs, for the freestanding wasm build (see build.py). */
#ifndef GP_WASM_MATH_H
#define GP_WASM_MATH_H
#define sqrtf( x ) __builtin_sqrtf( x )
#define fabsf( x ) __builtin_fabsf( x )
#define floorf( x ) __builtin_floorf( x )
#define ceilf( x ) __builtin_ceilf( x )
#define fminf( a, b ) __builtin_fminf( a, b )
#define fmaxf( a, b ) __builtin_fmaxf( a, b )
#define sqrt( x ) __builtin_sqrt( x )
#define fabs( x ) __builtin_fabs( x )
#define floor( x ) __builtin_floor( x )
float sinf( float );
float cosf( float );
float atan2f( float, float );
float hypotf( float, float );
float fmodf( float, float );
float powf( float, float );
float atanf( float );
float acosf( float );
float asinf( float );
float tanf( float );
float expf( float );
float remainderf( float, float );
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#endif
