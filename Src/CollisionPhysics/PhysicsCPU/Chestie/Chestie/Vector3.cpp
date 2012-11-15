#include "Vector3.h"
#include "Vector2.h"
#include "RMath.h"
#include "Matrix.h"
#include <math.h>

const Vector3 Vector3::Left		= Vector3(-1.f, 0.f, 0.f);
const Vector3 Vector3::Right	= Vector3(+1.f, 0.f, 0.f);
const Vector3 Vector3::Down		= Vector3(0.f, -1.f, 0.f);
const Vector3 Vector3::Up		= Vector3(0.f, +1.f, 0.f);
const Vector3 Vector3::Forward	= Vector3(0.f, 0.f, -1.f);
const Vector3 Vector3::Backward	= Vector3(0.f, 0.f, +1.f);

const Vector3 Vector3::UnitX	= Vector3(1.f, 0.f, 0.f);
const Vector3 Vector3::UnitY	= Vector3(0.f, 1.f, 0.f);
const Vector3 Vector3::UnitZ	= Vector3(0.f, 0.f, 1.f);
const Vector3 Vector3::Zero		= Vector3(0.f, 0.f, 0.f);
const Vector3 Vector3::One		= Vector3(1.f, 1.f, 1.f);

const Vector3 Vector3::Luma     = Vector3(0.299f, 0.587f, 0.114f);

void Vector3::Normalize()
{
	const float rlen = Math::Rsqrt(x*x + y*y + z*z);
	x *= rlen;
	y *= rlen;
	z *= rlen;
}

void Vector3::Normalize(const Vector3 *const v, Vector3 *const result)
{
	const float rlen = Math::Rsqrt(v->x*v->x + v->y*v->y + v->z*v->z);
	result->x = v->x * rlen;
	result->y = v->y * rlen;
	result->z = v->z * rlen;
}

Vector3	Vector3::Normalize(const Vector3 *const v)
{
	const float rlen = Math::Rsqrt(v->x*v->x + v->y*v->y + v->z*v->z);
	return Vector3(v->x * rlen, v->y * rlen, v->z * rlen);
}

void  Vector3::Negate()
{ 
	x = -x;
	y = -y;
	z = -z;
}

float Vector3::Length()			const	{ return Math::Sqrt(x*x + y*y + z*z); }
float Vector3::LengthReversed()	const	{ return Math::Rsqrt(x*x + y*y + z*z); }
float Vector3::LengthSquared()	const	{ return x*x + y*y + z*z; }
float Vector3::Yaw()			const	{ return atan2f(-x, -z); }
float Vector3::Pitch()			const	{ return asinf(y); }

float Vector3::Dot(const Vector3 *const v1, const Vector3 *const v2)
{
	return v1->x*v2->x + v1->y*v2->y + v1->z*v2->z;
}

float Vector3::DotNorm(const Vector3 *const v1, const Vector3 *const p1v2, const Vector3 *const p2v2)
{
    const float dx = p2v2->x - p1v2->x;
    const float dy = p2v2->y - p1v2->y;
    const float dz = p2v2->z - p1v2->z;
    const float dot = v1->x * dx + v1->y * dy + v1->z * dz;
    return dot * Math::Rsqrt(dx*dx + dy*dy + dz*dz);
}

void Vector3::Cross(const Vector3 *const v1, const Vector3 * const v2, Vector3 * const result)
{
	result->x = v1->y*v2->z - v1->z*v2->y;
	result->y = v1->z*v2->x - v1->x*v2->z;
	result->z = v1->x*v2->y - v1->y*v2->x;
}

void Vector3::Add(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result)
{
	result->x = v1->x + v2->x;
	result->y = v1->y + v2->y;
	result->z = v1->z + v2->z;
}

void Vector3::Add(const Vector3 *const v, const float val, Vector3 *const result)
{
	result->x = v->x + val;
	result->y = v->y + val;
	result->z = v->z + val;
}

void Vector3::Subtract(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result)
{
	result->x = v1->x - v2->x;
	result->y = v1->y - v2->y;
	result->z = v1->z - v2->z;
}

void Vector3::Subtract(const Vector3 *const v, const float val, Vector3 *const result)
{
	result->x = v->x - val;
	result->y = v->y - val;
	result->z = v->z - val;
}

void Vector3::Divide(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result)
{
	result->x = v1->x / v2->x;
	result->y = v1->y / v2->y;
	result->z = v1->z / v2->z;
}

void Vector3::Divide(const Vector3 *const v, const float val, Vector3 *const result)
{
	result->x = v->x / val;
	result->y = v->y / val;
	result->z = v->z / val;
}

void Vector3::Multiply(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result)
{
	result->x = v1->x * v2->x;
	result->y = v1->y * v2->y;
	result->z = v1->z * v2->z;
}

void Vector3::Multiply(const Vector3 *const v, const float val, Vector3 *const result)
{
	result->x = v->x * val;
	result->y = v->y * val;
	result->z = v->z * val;
}

void Vector3::Negate(const Vector3 *const v, Vector3 *const result)
{
	result->x = -v->x;
	result->y = -v->y;
	result->z = -v->z;
}

float Vector3::Distance2(const Vector3 *const v1, const Vector3 *const v2)
{
	const float dx = v1->x - v2->x;
	const float dy = v1->y - v2->y;
	const float dz = v1->z - v2->z;
	return dx*dx + dy*dy + dz*dz;
}

float Vector3::Distance(const Vector3 *const v1, const Vector3 *const v2)
{
	const float dx = v1->x - v2->x;
	const float dy = v1->y - v2->y;
	const float dz = v1->z - v2->z;
	return Math::Sqrt(dx*dx + dy*dy + dz*dz);
}

float Vector3::Distance2(const Vector3 & v1, const Vector3 & v2)
{
	const float dx = v1.x - v2.x;
	const float dy = v1.y - v2.y;
	const float dz = v1.z - v2.z;
	return dx*dx + dy*dy + dz*dz;
}

float Vector3::Distance(const Vector3 & v1, const Vector3 & v2)
{
	const float dx = v1.x - v2.x;
	const float dy = v1.y - v2.y;
	const float dz = v1.z - v2.z;
	return Math::Sqrt(dx*dx + dy*dy + dz*dz);
}


Vector3 Vector3::Add(const Vector3 *const v1, const Vector3 *const v2)
{
	return Vector3(v1->x + v2->x, v1->y + v2->y, v1->z + v2->z);
}

Vector3 Vector3::Add(const Vector3 *const v, const float val)
{
	return Vector3(v->x + val, v->y + val, v->z + val);
}

Vector3 Vector3::Subtract(const Vector3 *const v1, const Vector3 *const v2)
{
	return Vector3(v1->x - v2->x, v1->y - v2->y, v1->z - v2->z);
}

Vector3 Vector3::Subtract(const Vector3 *const v, const float val)
{
	return Vector3(v->x - val, v->y - val, v->z - val);
}

Vector3 Vector3::Divide(const Vector3 *const v1, const Vector3 *const v2)
{
	return Vector3(v1->x / v2->x, v1->y / v2->y, v1->z / v2->z);
}

Vector3 Vector3::Divide(const Vector3 *const v, const float val)
{
	return Vector3(v->x / val, v->y / val, v->z / val);
}

Vector3 Vector3::Multiply(const Vector3 *const v1, const Vector3 *const v2)
{
	return Vector3(v1->x * v2->x, v1->y * v2->y, v1->z * v2->z);
}

Vector3 Vector3::Multiply(const Vector3 *const v, const float val)
{
	return Vector3(v->x * val, v->y * val, v->z * val);
}

Vector3 Vector3::Negate(const Vector3 *const v)
{
	return Vector3(-v->x, -v->y, -v->z);
}


void Vector3::Lerp(const Vector3 *const v1, const Vector3 *const v2, const float ammount, Vector3 *const result)
{
	if(ammount <= 0.f)
	{
		result->x = v1->x;
		result->y = v1->y;
		result->z = v1->z;
	}
	else if(ammount >= 1.f)
	{
		result->x = v2->x;
		result->y = v2->y;
		result->z = v2->z;
	}
	else
	{
		result->x = v1->x + (v2->x - v1->x) * ammount;
		result->y = v1->y + (v2->y - v1->y) * ammount;
		result->z = v1->z + (v2->z - v1->z) * ammount;
	}
}

void Vector3::Clamp(const Vector3 *const v, const Vector3 *const min, const Vector3 *const max, Vector3 *const result)
{
	result->x = v->x > min->x ? (v->x < max->x ? v->x : max->x) : min->x;
	result->y = v->y > min->y ? (v->y < max->y ? v->y : max->y) : min->y;
	result->z = v->z > min->z ? (v->z < max->z ? v->z : max->z) : min->z;
}

void Vector3::Clamp(const Vector3 *const v, const float min, const float max, Vector3 *const result)
{
	result->x = v->x > min ? (v->x < max ? v->x : max) : min;
	result->y = v->y > min ? (v->y < max ? v->y : max) : min;
	result->z = v->z > min ? (v->z < max ? v->z : max) : min;
}

void Vector3::Max(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result)
{
	result->x = v1->x > v2->x ? v1->x : v2->x;
	result->y = v1->y > v2->y ? v1->y : v2->y;
	result->z = v1->z > v2->z ? v1->z : v2->z;
}

void Vector3::Min(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result)
{
	result->x = v1->x > v2->x ? v2->x : v1->x;
	result->y = v1->y > v2->y ? v2->y : v1->y;
	result->z = v1->z > v2->z ? v2->z : v1->z;
}

void Vector3::Reflect(const Vector3 *const v, const Vector3 *const normal, Vector3 *const result)
{
	const float dot2 = (v->x*normal->x + v->y*normal->y + v->z*normal->z) * 2.f;
	result->x = v->x - normal->x*dot2;
	result->y = v->y - normal->y*dot2;
	result->z = v->z - normal->z*dot2;
}

void Vector3::Transform(const Vector3 *const position, const Matrix *const mat, Vector3 *const result)
{
	result->x = mat->m11 * position->x + mat->m12 * position->y + mat->m13 * position->z + mat->m14;
	result->y = mat->m21 * position->x + mat->m22 * position->y + mat->m23 * position->z + mat->m24;
	result->z = mat->m31 * position->x + mat->m32 * position->y + mat->m33 * position->z + mat->m34;
}

void Vector3::TransformNormal(const Vector3 *const direction, const Matrix *const mat, Vector3 *const result)
{
	result->x = mat->m11 * direction->x + mat->m12 * direction->y + mat->m13 * direction->z;
	result->y = mat->m21 * direction->x + mat->m22 * direction->y + mat->m23 * direction->z;
	result->z = mat->m31 * direction->x + mat->m32 * direction->y + mat->m33 * direction->z;
}

void Vector3::TransformNormal(const Vector3 *const direction, const Quaternion *const quaternion, Vector3 *const result)
{		
	const float tx = quaternion->w*direction->x + quaternion->y*direction->z - quaternion->z*direction->y;
	const float ty = quaternion->w*direction->y - quaternion->x*direction->z + quaternion->z*direction->x;
	const float tz = quaternion->w*direction->z + quaternion->x*direction->y - quaternion->y*direction->x;
	const float tw = -quaternion->x*direction->x - quaternion->y*direction->y - quaternion->z*direction->z;
	
	result->x = -tw*quaternion->x + tx*quaternion->w - ty*quaternion->z + tz*quaternion->y;
	result->y = -tw*quaternion->y + ty*quaternion->w - tz*quaternion->x + tx*quaternion->z;
	result->z = -tw*quaternion->z + tz*quaternion->w - tx*quaternion->y + ty*quaternion->x;
}

void Vector3::TransformInverseNormal(const Vector3 *const direction, const Matrix *const mat, Vector3 *const result)
{
	result->x = mat->m11 * direction->x + mat->m21 * direction->y + mat->m31 * direction->z;
	result->y = mat->m12 * direction->x + mat->m22 * direction->y + mat->m32 * direction->z;
	result->z = mat->m13 * direction->x + mat->m23 * direction->y + mat->m33 * direction->z;
}

void Vector3::TransformInverseNormal(const Vector3 *const direction, const Quaternion *const quaternion, Vector3 *const result)
{
	const float tx = quaternion->w*direction->x - quaternion->y*direction->z + quaternion->z*direction->y;
	const float ty = quaternion->w*direction->y + quaternion->x*direction->z - quaternion->z*direction->x;
	const float tz = quaternion->w*direction->z - quaternion->x*direction->y + quaternion->y*direction->x;
	const float tw = quaternion->x*direction->x + quaternion->y*direction->y + quaternion->z*direction->z;
	
	result->x = tw*quaternion->x + tx*quaternion->w + ty*quaternion->z - tz*quaternion->y;
	result->y = tw*quaternion->y + ty*quaternion->w + tz*quaternion->x - tx*quaternion->z;
	result->z = tw*quaternion->z + tz*quaternion->w + tx*quaternion->y - ty*quaternion->x;
}

void Vector3::operator += (const Vector3 &v)
{
	x += v.x;
	y += v.y;
	z += v.z;
}

void Vector3::operator += (float val)
{
	x += val;
	y += val;
	z += val;
}

void Vector3::operator -= (const Vector3 &v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
}

void Vector3::operator -= (float val)
{
	x -= val;
	y -= val;
	z -= val;
}

void Vector3::operator *= (const Vector3 &v)
{
	x *= v.x;
	y *= v.y;
	z *= v.z;
}

void Vector3::operator *= (float val)
{
	x *= val;
	y *= val;
	z *= val;
}

void Vector3::operator /= (const Vector3 &v)
{
	x /= v.x;
	y /= v.y;
	z /= v.z;
}

void Vector3::operator /= (float val)
{
	x /= val;
	y /= val;
	z /= val;
}

Vector3& Vector3::operator  = (const Vector3 &v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	return *this;
}

Vector3& Vector3::operator  = (const Vector2 &v)
{
	x = v.x;
	y = v.y;
	return *this;
}

void Vector3::Randomize(float rx, float ry, float rz)
{
	x = Math::Rand11() * rx;
	y = Math::Rand11() * ry;
	z = Math::Rand11() * rz;
}

void Vector3::RandomizeNNInt(float minx, float miny, float minz, float maxx, float maxy, float maxz)
{
	x = Math::RandNNInt(minx, maxx);
	y = Math::RandNNInt(miny, maxy);
	z = Math::RandNNInt(minz, maxz);
}
