#pragma once
#include "Vector3.h"

class Plane
{
public:
	Vector3 normal;
	float	d;
	
	inline Plane() {}
	inline Plane(const Vector3 *const Normal, const float D) : normal(*Normal), d(D) {}
	inline Plane(const float normalX, const float normalY, const float normalZ, const float D) : d(D) { normal.x = normalX; normal.y = normalY; normal.z = normalZ; }
	Plane(const Vector3 *const position1, const Vector3 *const position2, const Vector3 *const position3);
	~Plane()	{}
	
	inline float * Ptr() { return &normal.x; }
	inline const float * Ptr() const { return &normal.x; }
	
	static const Plane YOZ;
	static const Plane ZOX;
	static const Plane XOY;
	static const Plane YOZNeg;
	static const Plane ZOXNeg;
	static const Plane XOYNeg;
	
	inline float	DotCoordinate(const Vector3 *const value) const	{ return d + Vector3::Dot(&normal, value); }
	inline float	DotNormal(const Vector3 *const value) const		{ return Vector3::Dot(&normal, value); }
	void			Normalize();
	
	static void		Transform(const Plane *const plane, const Matrix *const mat, Plane *const result);
	static float	Distance(const Plane* const plane, const Vector3 * const point);
};
