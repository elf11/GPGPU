#include "Plane.h"
#include "RMath.h"

const Plane Plane::YOZ		= Plane(1.f, 0.f, 0.f, 0.f);
const Plane Plane::ZOX		= Plane(0.f, 1.f, 0.f, 0.f);
const Plane Plane::XOY		= Plane(0.f, 0.f, 1.f, 0.f);
const Plane Plane::YOZNeg	= Plane(-1.f, 0.f, 0.f, 0.f);
const Plane Plane::ZOXNeg	= Plane(0.f, -1.f, 0.f, 0.f);
const Plane Plane::XOYNeg	= Plane(0.f, 0.f, -1.f, 0.f);

Plane::Plane(const Vector3 *const position1, const Vector3 *const position2, const Vector3 *const position3)
{
	const float x12 = position1->x - position2->x;
	const float y12 = position1->y - position2->y;
	const float z12 = position1->z - position2->z;
	
	const float x23 = position2->x - position3->x;
	const float y23 = position2->y - position3->y;
	const float z23 = position2->z - position3->z;
	
	const float x31 = position3->x - position1->x;
	const float y31 = position3->y - position1->y;
	const float z31 = position3->z - position1->z;
	
	normal.x = position1->y * z23 + position2->y * z31 + position3->y * z12;
	normal.y = position1->z * x23 + position2->z * x31 + position3->z * x12;
	normal.z = position1->x * y23 + position2->x * y31 + position3->x * y12;
	
	d = position1->x * (position3->y * position2->z - position2->y * position3->z) +
		position2->x * (position1->y * position3->z - position3->y * position1->z) +
		position3->x * (position2->y * position1->z - position1->y * position2->z);
}

void Plane::Normalize()
{
	const float rnorm = Math::Rsqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
	normal.x	*= rnorm;
	normal.y	*= rnorm;
	normal.z	*= rnorm;
	d			*= rnorm;
}

void Plane::Transform(const Plane *const plane, const Matrix *const mat, Plane *const result)
{
	result->normal.x = mat->m11 * plane->normal.x + mat->m12 * plane->normal.y + mat->m13 * plane->normal.z + plane->d * mat->m14;
	result->normal.y = mat->m21 * plane->normal.x + mat->m22 * plane->normal.y + mat->m23 * plane->normal.z + plane->d * mat->m24;
	result->normal.z = mat->m31 * plane->normal.x + mat->m32 * plane->normal.y + mat->m33 * plane->normal.z + plane->d * mat->m34;
	result->d        = mat->m41 * plane->normal.x + mat->m42 * plane->normal.y + mat->m43 * plane->normal.z + plane->d * mat->m44;
}


float Plane::Distance(const Plane* const plane, const Vector3 * const point)
{
	return plane->normal.x*point->x + plane->normal.y*point->y + plane->normal.z*point->z + plane->d;
}