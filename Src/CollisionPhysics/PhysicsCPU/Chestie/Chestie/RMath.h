#pragma once

#include "Vector2.h"
#include "Vector3.h"
#include "Matrix.h"
#include "MatrixStack.h"
#include "Quaternion.h"
#include "Plane.h"
#include "BoundingFrustum.h"
#include "BoundingSphere.h"
#include "BoundingBox.h"
#include "MathOperators.h"
#include "CRSpline.h"


class Math
{
public:
	static const float	E;
	static const float	Log2E;
	static const float	Log10E;
	
	static const float	Pi;
	static const float	PiOver2;
	static const float	PiOver3;
	static const float	PiOver4;
	static const float	PiOver6;
	static const float	TwoPi;
	static const float	TwoPiOver3;
	static const float	FourPiOver3;
    static const float  Sqrt2;

	static const float			MaxFloat;
	static const float			MinFloat;
    static const float          MinPositiveFloat;
    static const float          MaxNegativeFloat;
	
	static const unsigned int	MaxUint;
	static const unsigned int	MinUint;
	static const unsigned short	MaxUshort;
	static const unsigned short	MinUshort;
	static const unsigned char	MaxUchar;
	static const unsigned char	MinUchar;
	
	static const int			MaxInt;
	static const int			MinInt;
	static const short			MaxShort;
	static const short			MinShort;
	static const char			MaxChar;
	static const char			MinChar;
	
	static float	Sin(const float rad);
	static float	Cos(const float rad);
	static float	Tan(const float rad);
	
	static float	Asin(const float rad);
	static float	Acos(const float rad);
	static float	Atan(const float rad);
	static float	Atan2(const float y, const float x);
	
	static float	Abs(const float val);
	static int		Abs(const int val);
	static short	Abs(const short val);
	static char		Abs(const char val);
	
	static float	Sqrt(const float val);
	static float	Rsqrt(const float val);
	static float	Square(const float val);
	
	static float	Ceil(const float val);
	static float	Floor(const float val);
	static float	Round(const float val);
	static float	Trunc(const float val);
	static int		Sign(const float val);
	static int		Sign0(const float val);
	
	static float	Pow(const float x, const float y);
	static float	Exp(const float power);
	static float	Expm1(const float power);
	static float	Log(const float val);
	static float	Log(const float val, const float base);
	static float	Log2(const float val);
	static float	Log10(const float val);
	
	static float	Rem(const float x, const float y);
	static float	Mod(const float val, float *const intpart);
	static float	Fmod(const float x, const float y);
	
	inline static float				Max(const float x, const float y)					{ return x > y ? x : y; }
	inline static int				Max(const int x, const int y)						{ return x > y ? x : y; }
	inline static unsigned int		Max(const unsigned int x, const unsigned int y)		{ return x > y ? x : y; }
	inline static short				Max(const short x, const short y)					{ return x > y ? x : y; }
	inline static unsigned short	Max(const unsigned short x, const unsigned short y) { return x > y ? x : y; }
	inline static char				Max(const char x, const char y)						{ return x > y ? x : y; }
	inline static unsigned char		Max(const unsigned char x, const unsigned char y)	{ return x > y ? x : y; }
	
	inline static float				Min(const float x, const float y)					{ return x > y ? y : x; }
	inline static int				Min(const int x, const int y)						{ return x > y ? y : x; }
	inline static unsigned int		Min(const unsigned int x, const unsigned int y)		{ return x > y ? y : x; }
	inline static short				Min(const short x, const short y)					{ return x > y ? y : x; }
	inline static unsigned short	Min(const unsigned short x, const unsigned short y) { return x > y ? y : x; }
	inline static char				Min(const char x, const char y)						{ return x > y ? y : x; }
	inline static unsigned char		Min(const unsigned char x, const unsigned char y)	{ return x > y ? y : x; }
	
	inline static float	Clamp(const float val, const float min, const float max)		{ return val > min ? (val < max ? val : max) : min; }
	inline static float	Lerp(const float val1, const float val2, const float ammount)	{ return val1 + (val2 - val1) * ammount; }
	inline static float Distance(const float val1, const float val2)					{ return Abs(val1 - val2); }
	
	inline static float ToDegrees(const float rad)										{ return rad * 57.295779513082325f; }
	inline static float	ToRadians(const float deg)										{ return deg * 0.017453292519943f;	}
	static float WrapAngle(float rad);
	static void  WrapAngle(float *const angle);	

	static void ShellSort(void *array, const unsigned int numElements, const unsigned int sizeElement, int (*fcmp)(const void *const, const void *const));
    static void ShellSortArrayOfPointers(unsigned int *const array, const unsigned int numElements, int (*fcmp)(const void *const, const void *const));
	static void QuickSort(void *array, const unsigned int numElements, const unsigned int sizeElement, int (*fcmp)(const void *const, const void *const));

	static void		SeedRand(unsigned int seed);
	static int		Rand();
	
	static float	Rand(const float min, const float max);		//random between (min, max)
	static float	Rand(float max);							//random between (0, N)
	static float	RandAngle();								//random between (-Pi,Pi)
	static float	Rand01();									//random between (0, 1)
	static float	Rand11();									//random between (-1, 1)
	static float	RandNN(const float n);						//random between (-N, N)
	static float	RandNNInt(const float min, const float max);	//random between (-max, -min) or (min, max)
	static Vector3	RandNN(const float nx, const float ny, const float nz);
	static Vector3	RandNN(const Vector3 *const nval);
	static Vector2	RandNN(const float nx, const float ny);
	static Vector2	RandNN(const Vector2 *const nval);

	static bool Equals(float a, float b, float e=0.01f);
};

#ifndef NULL
#define NULL 0
#endif

#ifndef null
#define null 0
#endif
