#pragma once

#include "Vector3.h"

class BoundingSphere
{
public:
	Vector3 center;
	float	radius;
	
	inline BoundingSphere() {}
	inline BoundingSphere(const Vector3 *const Center, const float Radius) : center(*Center), radius(Radius) {}
	~BoundingSphere()	{}
	
	inline float * Ptr() { return &center.x; }
	inline const float * Ptr() const { return &center.x; }
};

