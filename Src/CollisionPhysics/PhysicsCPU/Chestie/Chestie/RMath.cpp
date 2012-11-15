#include "RMath.h"
#include "platform.h"
#include <math.h>
#include <stdlib.h>
#include <memory.h>

const float	Math::E				= 2.718281828459045f;
const float	Math::Log2E			= 1.442695040888963f;
const float	Math::Log10E		= 0.434294481903252f;

const float	Math::Pi			= 3.141592653589793f;
const float	Math::PiOver2		= 1.570796326794897f;
const float	Math::PiOver3		= 1.047197551196598f;
const float	Math::PiOver4		= 0.785398163397448f;
const float	Math::PiOver6		= 0.523598775598299f;
const float	Math::TwoPi			= 6.283185307179586f;
const float	Math::TwoPiOver3	= 2.094395102393195f;
const float	Math::FourPiOver3	= 4.188790204786391f;
const float Math::Sqrt2         = 1.414213562373095f;

const float				Math::MaxFloat          =  1E+37f;
const float				Math::MinFloat          = -1E+37f;
const float				Math::MinPositiveFloat	=  1E-37f;
const float				Math::MaxNegativeFloat	= -1E-37f;

const unsigned int		Math::MaxUint	= 4294967295u;
const unsigned int		Math::MinUint	= 0u;
const unsigned short	Math::MaxUshort	= 65535u;
const unsigned short	Math::MinUshort = 0u;
const unsigned char		Math::MaxUchar  = 255u;
const unsigned char		Math::MinUchar  = 0u;

const int				Math::MaxInt    = +2147483647;
const int				Math::MinInt    = -2147483647;
const short				Math::MaxShort  = +32767;
const short				Math::MinShort  = -32767;
const char				Math::MaxChar   = +127;
const char				Math::MinChar   = -127;

float	Math::Sin(const float rad)						{ return sinf(rad); }
float	Math::Cos(const float rad)						{ return cosf(rad); }
float	Math::Tan(const float rad)						{ return tanf(rad); }
float	Math::Asin(const float val)						{ return asinf(val); }
float	Math::Acos(const float val)						{ return acosf(val); }
float	Math::Atan(const float val)						{ return atanf(val); }
float	Math::Atan2(const float y, const float x)		{ return atan2f(y, x); }

float	Math::Abs(const float val)						{ return fabsf(val); }
float	Math::Ceil(const float val)						{ return ceilf(val); }
float	Math::Floor(const float val)					{ return floorf(val); }
float	Math::Round(const float val)					{ return roundf(val); }
float	Math::Trunc(const float val)					{ return truncf(val); }

float	Math::Pow(const float x, const float y)			{ return powf(x, y); }
float	Math::Exp(const float power)					{ return expf(power); }
float	Math::Expm1(const float power)					{ return expm1f(power); }
float	Math::Log(const float val)						{ return logf(val); }
float	Math::Log(const float val, const float base)	{ return logf(val) / logf(base); }
float	Math::Log2(const float val)						{ return log2f(val); }
float	Math::Log10(const float val)					{ return log10f(val); }

float	Math::Rem(const float x, const float y)			{ return remainderf(x, y); }
float	Math::Mod(const float val, float *const intpart){ return modff(val, intpart); }
float	Math::Fmod(const float x, const float y)		{ return fmodf(x, y); }

float Math::Square(const float val)
{
	return val*val;
}
int		Math::Abs(const int val)
{
	int t = val >> 31; 
	return t ^ (val + t);
}

short	Math::Abs(const short val)
{
	short t = val >> 15; 
	return t ^ (val + t);
}

char	Math::Abs(const char val)
{
	char t = val >> 7; 
	return t ^ (val + t);
}

float	Math::Sqrt(const float val)
{
	const float halfx = 0.5f * val;
	float _x;
	
	int carmack = *(const int *) &val;
	carmack = 0x5F3759DF - (carmack >> 1);
	_x = *(float *) &carmack;
	
	_x = _x * (1.5f - halfx * _x * _x);
	_x = _x * (1.5f - halfx * _x * _x);
	return _x * val;
}

float	Math::Rsqrt(const float val)
{
	const float halfx = 0.5f * val;
	float _x;
	
	int carmack = *(const int *) &val;
	carmack = 0x5F3759DF - (carmack >> 1);
	_x = *(float *) &carmack;
	
	_x = _x * (1.5f - halfx * _x * _x);
	_x = _x * (1.5f - halfx * _x * _x);
	return _x;
}

int		Math::Sign(const float val)
{
	return 1 + (((*(const int*const)&val) >> 31) << 1);
}

int		Math::Sign0(const float val)
{
	return val != 0.f ? (1 + (((*(const int*const)&val) >> 31) << 1)) : 0;
}

float	Math::WrapAngle(float rad)
{
	rad = fmodf(rad, TwoPi);
	
	if (rad > Pi)
        rad -= TwoPi;
	else if (rad <= -Pi)  
        rad += TwoPi;
	
	return rad;
}


void	Math::WrapAngle(float *const rad)
{
	*rad = fmodf(*rad, TwoPi);
	
	if (*rad > Pi)
        *rad -= TwoPi;
	else if (*rad <= -Pi)  
        *rad += TwoPi;
}

void Math::QuickSort(void *array, const unsigned int numElements, const unsigned int sizeElement, int (*fcmp)(const void *const, const void *const))
{
	qsort(array, numElements, sizeElement, fcmp);
}

void Math::ShellSort(void *array, const unsigned int numElements, const unsigned int sizeElement, int (*fcmp)(const void *const, const void *const))
{
	const char *const pend = ((char *) array) + (numElements * sizeElement);
	unsigned int gap = numElements >> 1, gapsz = gap * sizeElement;
	char *pi, *pj, *pgap = ((char *) array) + gapsz, *pjgap;
	char *temp = new char[sizeElement];
	
	while(gap)
	{
		for(pi = pgap; pi < pend; pi += sizeElement)
		{
			memcpy(temp, pi, sizeElement);
			pj = pi;
			pjgap = pj - gapsz;
			
			while(pj >= pgap && (*fcmp)(pjgap, temp) > 0)
			{
				memcpy(pj, pjgap, sizeElement);
				pj = pjgap;
				pjgap -= gapsz;
			}
			memcpy(pj, temp, sizeElement);
		}
		gap = Math::Round(0.454545454545455f * gap);
		gapsz = gap * sizeElement;
		pgap = ((char *) array) + gapsz;
	}
	delete[] temp;
}

void Math::ShellSortArrayOfPointers(unsigned int *const array, const unsigned int numElements, int (*fcmp)(const void *const, const void *const))
{
    const unsigned int *const pend = array + numElements;
	unsigned int gap = numElements >> 1, gapsz = gap;
	unsigned int *pi, *pj, *pgap = array + gapsz, *pjgap;
	unsigned int temp;
	
	while(gap)
	{
		for(pi = pgap; pi < pend; ++pi)
		{
            temp = *pi;
			pj = pi;
			pjgap = pj - gapsz;
			
			while(pj >= pgap && (*fcmp)(pjgap, &temp) > 0)
			{
                *pj = *pjgap;
				pj = pjgap;
				pjgap -= gapsz;
			}
            *pj = temp;
		}
		gap = Math::Round(0.454545454545455f * gap);
		gapsz = gap;
		pgap = array + gapsz;
	}
}

void Math::SeedRand(unsigned int seed)
{
	srand(seed);
}

int	Math::Rand()
{
	return rand();
}

float Math::Rand(const float min, const float max)
{
	return ((max - min) * float(rand()) / float(RAND_MAX)) + min; 
}

float Math::Rand(const float max)
{
	return max * float(rand()) / float(RAND_MAX);
}

float Math::Rand01()
{
	return float(rand()) / float(RAND_MAX);
}

float Math::Rand11()
{
	return (2.f * float(rand()) / float(RAND_MAX)) - 1.f;
}

float Math::RandNNInt(const float min, const float max)
{
	float v = Rand01() * (max-min) + min;

	if (rand() & 0x0001)
		return v;
	else
		return -v;
}

float Math::RandAngle()
{
	return (TwoPi * float(rand()) / float(RAND_MAX)) - Pi;
}

float Math::RandNN(const float n)
{
	return n * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f);
}

Vector3 Math::RandNN(const float nx, const float ny, const float nz)
{
	return Vector3(nx * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f),
				   ny * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f),
				   nz * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f));
}

Vector3 Math::RandNN(const Vector3 *const nval)
{
	return Vector3(nval->x * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f),
				   nval->y * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f),
				   nval->z * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f));
}

Vector2 Math::RandNN(const float nx, const float ny)
{
	return Vector2(nx * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f),
				   ny * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f));
}

Vector2 Math::RandNN(const Vector2 *const nval)
{
	return Vector2(nval->x * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f),
				   nval->y * ((2.f * float(rand()) / float(RAND_MAX)) - 1.f));
}

bool Math::Equals(float a, float b, float e)
{
	return (Abs(a - b) < e);
}
