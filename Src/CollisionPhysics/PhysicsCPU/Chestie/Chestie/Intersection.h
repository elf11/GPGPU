
#ifndef __INTERSECTION_H_DEF__
#define __INTERSECTION_H_DEF__

#include "stdinc.h"
#include "RMath.h"
#include "tvector.h"

class BaseObject;

class Intersection
{
public:
	static bool SphereSphere(const TVector * const pos1, const float _rad1, const TVector * const pos2, const float _rad2);
	static bool PointSphere(const TVector *const pos, const TVector *const spherePos, const float sphereRadius);
	
};

#endif // __INTERSECTION_H_DEF__
