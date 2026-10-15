/* A tiny libc for the freestanding wasm32 build of gpuphys.c: a heap, mem*, qsort, and the trigonometry
 * that WebAssembly has no instruction for (imported from JavaScript's Math, see gpuphys.js). */
#include <stddef.h>

#define IMPORT( name ) __attribute__( ( import_module( "env" ), import_name( name ) ) )
IMPORT( "sinf" ) float sinf( float );
IMPORT( "cosf" ) float cosf( float );
IMPORT( "atan2f" ) float atan2f( float, float );
IMPORT( "powf" ) float powf( float, float );
IMPORT( "atanf" ) float atanf( float );
IMPORT( "acosf" ) float acosf( float );
IMPORT( "asinf" ) float asinf( float );
IMPORT( "tanf" ) float tanf( float );
IMPORT( "expf" ) float expf( float );

float hypotf( float x, float y )
{
	return __builtin_sqrtf( x * x + y * y );
}
float fmodf( float x, float y )
{
	return x - y * __builtin_truncf( x / y );
}
float remainderf( float x, float y )
{
	return x - y * __builtin_rintf( x / y );
}
int abs( int x )
{
	return x < 0 ? -x : x;
}

void *memcpy( void *d, const void *s, size_t n )
{
	unsigned char *a = d;
	const unsigned char *b = s;
	while ( n-- )
		*a++ = *b++;
	return d;
}
void *memmove( void *d, const void *s, size_t n )
{
	unsigned char *a = d;
	const unsigned char *b = s;
	if ( a < b )
		while ( n-- )
			*a++ = *b++;
	else
		while ( n-- )
			a[n] = b[n];
	return d;
}
void *memset( void *d, int c, size_t n )
{
	unsigned char *a = d;
	while ( n-- )
		*a++ = (unsigned char)c;
	return d;
}
int memcmp( const void *x, const void *y, size_t n )
{
	const unsigned char *a = x, *b = y;
	for ( ; n; --n, ++a, ++b )
		if ( *a != *b )
			return *a - *b;
	return 0;
}
size_t strlen( const char *s )
{
	size_t n = 0;
	while ( s[n] )
		++n;
	return n;
}

/* ---- heap: first fit over a free list, grown a page at a time; blocks carry their size ------------------- */
extern unsigned char __heap_base;
typedef struct Blk
{
	size_t size; /* payload bytes */
	struct Blk *next;
} Blk;
#define HDR 16u /* keeps payloads 16-aligned */
static Blk *free_list;
static unsigned char *brk;
static unsigned char *limit;

static size_t round16( size_t n )
{
	return ( n + 15u ) & ~(size_t)15u;
}
static void *morecore( size_t bytes )
{
	if ( !brk )
	{
		brk = (unsigned char *)round16( (size_t)&__heap_base );
		limit = (unsigned char *)( (size_t)__builtin_wasm_memory_size( 0 ) * 65536u );
	}
	if ( brk + bytes > limit )
	{
		size_t need = ( (size_t)( brk + bytes - limit ) + 65535u ) / 65536u;
		if ( __builtin_wasm_memory_grow( 0, need ) == (size_t)-1 )
			return 0;
		limit += need * 65536u;
	}
	void *p = brk;
	brk += bytes;
	return p;
}
void *malloc( size_t n )
{
	n = round16( n ? n : 1 );
	Blk **pp = &free_list;
	for ( Blk *b = free_list; b; pp = &b->next, b = b->next )
		if ( b->size >= n )
		{
			*pp = b->next;
			return (unsigned char *)b + HDR;
		}
	Blk *b = morecore( HDR + n );
	if ( !b )
		return 0;
	b->size = n;
	return (unsigned char *)b + HDR;
}
void free( void *p )
{
	if ( !p )
		return;
	Blk *b = (Blk *)( (unsigned char *)p - HDR );
	b->next = free_list;
	free_list = b;
}
void *calloc( size_t a, size_t b )
{
	size_t n = a * b;
	void *p = malloc( n );
	if ( p )
		memset( p, 0, n );
	return p;
}
void *realloc( void *p, size_t n )
{
	if ( !p )
		return malloc( n );
	Blk *b = (Blk *)( (unsigned char *)p - HDR );
	if ( b->size >= n )
		return p;
	void *q = malloc( n );
	if ( q )
	{
		memcpy( q, p, b->size );
		free( p );
	}
	return q;
}

void qsort( void *base, size_t n, size_t sz, int ( *cmp )( const void *, const void * ) )
{
	unsigned char *a = base, tmp[64];
	if ( sz > sizeof tmp )
		return;
	for ( size_t i = 1; i < n; ++i ) /* insertion sort: the lists are short */
	{
		memcpy( tmp, a + i * sz, sz );
		size_t j = i;
		while ( j > 0 && cmp( a + ( j - 1 ) * sz, tmp ) > 0 )
		{
			memcpy( a + j * sz, a + ( j - 1 ) * sz, sz );
			--j;
		}
		memcpy( a + j * sz, tmp, sz );
	}
}

float fminf( float a, float b )
{
	return a < b ? a : b;
}
float fmaxf( float a, float b )
{
	return a > b ? a : b;
}
