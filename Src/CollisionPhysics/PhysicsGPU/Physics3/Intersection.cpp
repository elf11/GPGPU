

#include "Intersection.h"

bool Intersection::SphereSphere(const TVector *const pos1, const float _rad1, const TVector *const pos2, const float _rad2)
{
	TVector l = *pos2 - *pos1;
	float dst2 = TVector::dot(l, l);
	float mindst = _rad1 + _rad2;
	
	return (dst2 <= mindst * mindst);
}


bool Intersection::PointSphere(const TVector *const pos, const TVector *const spherePos, const float sphereRadius)
{
	TVector l = *spherePos - *pos;
	float dst2 = TVector::dot(l, l);
	
	return (dst2 <= sphereRadius * sphereRadius);
}