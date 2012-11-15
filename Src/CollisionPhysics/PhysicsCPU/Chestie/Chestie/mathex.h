#ifndef MATHEX_H
#define MATHEX_H

#include <math.h>

#define EPSILON 1.0e-8
#define ZERO EPSILON
#define M_PI 3.1415926535

#define VC 1
#if VC
	#define bool int
	#define false 0
	#define true !false
#endif
template <class T> inline T limit(const T &x, const T &lower, const T &upper)
{
	if (x < lower)
		return lower;
	if (x > upper)
		return upper;
	return x;
}
template <class T> inline T sqr(const T &x)
{
	return x*x;
}
template <class T> inline T RadToDeg(const T &rad)
{
}

template <class T> inline T DegToRad(const T &deg)
{
	return (deg * M_PI) / 180.0;
}

#endif