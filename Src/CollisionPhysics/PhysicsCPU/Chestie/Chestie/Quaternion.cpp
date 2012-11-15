#include "Quaternion.h"
#include "RMath.h"
#include "Matrix.h"
#include "Vector3.h"
#include "platform.h"
#include <math.h>

const Quaternion Quaternion::Identity	= Quaternion(0.f, 0.f, 0.f, 1.f);
const Quaternion Quaternion::Left		= Quaternion(0.f,  0.707106781186548f, 0.f, 0.707106781186548f);
const Quaternion Quaternion::Right		= Quaternion(0.f, -0.707106781186548f, 0.f, 0.707106781186548f);
const Quaternion Quaternion::Up			= Quaternion( 0.707106781186548f, 0.f, 0.f, 0.707106781186548f);
const Quaternion Quaternion::Down		= Quaternion(-0.707106781186548f, 0.f, 0.f, 0.707106781186548f);
const Quaternion Quaternion::Backward	= Quaternion(0.f, 1.f, 0.f, 0.f);

Quaternion & Quaternion::RotateX(const float angle)
{
	const Quaternion tmp = (*this);
	RotateX(&tmp, angle, this);
	return (*this);
}

Quaternion & Quaternion::RotateY(const float angle)
{
	const Quaternion tmp = (*this);
	RotateY(&tmp, angle, this);
	return (*this);
}

Quaternion & Quaternion::RotateZ(const float angle)
{
	const Quaternion tmp = (*this);
	RotateZ(&tmp, angle, this);
	return (*this);
}

Quaternion & Quaternion::Rotate(const Vector3 *const axis, const float angle)
{
	const Quaternion tmp = (*this);
	Rotate(&tmp, axis, angle, this);
	return (*this);
}

void Quaternion::Normalize()
{
	float mag = x*x + y*y + z*z + w*w;
	if(fabsf(mag) < 0.99999f)
	{
		mag = Math::Rsqrt(mag);
		x *= mag;
		y *= mag;
		z *= mag;
		w *= mag;
	}
}

float	Quaternion::Yaw() const		{ return atan2f(2.f*(z*x + y*w), 1.f - 2.f*(x*x + y*y)); }
float	Quaternion::Pitch() const	{ return asinf(Math::Clamp(2.f*(x*w - y*z), -1.f, 1.f)); }
float	Quaternion::Roll()	const	{ return atan2f(2.f * (x*y + w*z), 1.f - 2.f*(x*x + z*z)); }
Vector3	Quaternion::Forward() const	{ return Vector3(-2.f*(z*x + y*w), 2.f*(x*w - y*z), 2.f*(x*x + y*y) - 1.f); }
Vector3 Quaternion::RightDir() const	{ return Vector3(1.f - 2.f*(y*y + z*z), 2.f*(x*y + w*z), 2.f*(x*z - w*y));}
Vector3 Quaternion::UpDir() const		{ return Vector3(2.f*(x*y - w*z), 1.f - 2.f*(x*x + z*z), 2.f*(y*z + w*x));}

Quaternion	Quaternion::CreateFromAxisAngle(const Vector3 *const axis, const float angle)
{
	const float half_angle = angle * 0.5f;
	const float s = sinf(half_angle);
	
	return Quaternion
	(
		axis->x * s,
		axis->y * s,
		axis->z * s,
		cosf(half_angle)
	);
}

void Quaternion::CreateFromAxisAngle(const Vector3 *const axis, const float angle, Quaternion *const result)
{
	const float half_angle = angle * 0.5f;
	const float s = sinf(half_angle);
	
	result->x = axis->x * s;
	result->y = axis->y * s;
	result->z = axis->z * s;
	result->w = cosf(half_angle);
}

Quaternion Quaternion::CreateFromYawPitchRoll(const float yaw, const float pitch, const float roll)
{
	const float half_yaw = yaw * 0.5f;
	const float half_pitch = pitch * 0.5f;
	const float half_roll = roll * 0.5f;
	const float sy = sinf(half_yaw);
	const float cy = cosf(half_yaw);
	const float sp = sinf(half_pitch);
	const float cp = cosf(half_pitch);
	const float sr = sinf(half_roll);
	const float cr = cosf(half_roll);
	
	return Quaternion
	(
		((cy * sp) * cr) + ((sy * cp) * sr),
		((sy * cp) * cr) - ((cy * sp) * sr),
		((cy * cp) * sr) - ((sy * sp) * cr),
		((cy * cp) * cr) + ((sy * sp) * sr)
	);
}

void Quaternion::CreateFromYawPitchRoll(const float yaw, const float pitch, const float roll, Quaternion *const result)
{
	const float half_yaw = yaw * 0.5f;
	const float half_pitch = pitch * 0.5f;
	const float half_roll = roll * 0.5f;
	const float sy = sinf(half_yaw);
	const float cy = cosf(half_yaw);
	const float sp = sinf(half_pitch);
	const float cp = cosf(half_pitch);
	const float sr = sinf(half_roll);
	const float cr = cosf(half_roll);
	
	result->x = ((cy * sp) * cr) + ((sy * cp) * sr);
	result->y = ((sy * cp) * cr) - ((cy * sp) * sr);
	result->z = ((cy * cp) * sr) - ((sy * sp) * cr);
	result->w = ((cy * cp) * cr) + ((sy * sp) * sr);
}

Quaternion	Quaternion::CreateFromRotationMatrix(const Matrix *const matrix)
{
	Quaternion q;

	q.w = Math::Sqrt(matrix->m11 + matrix->m22 + matrix->m33 + 1.0f) * 0.5f;

	if(q.w > 0.0001f)
	{
		const float cw = 1.0f / (4.0f * q.w);
		q.x = (matrix->m32 - matrix->m23) * cw;
		q.y = (matrix->m13 - matrix->m31) * cw;
		q.z = (matrix->m21 - matrix->m12) * cw;
	}
	else
	{
		q.x = Math::Sqrt(matrix->m11*0.5f + 0.5f);
		q.y = Math::Sqrt(matrix->m22*0.5f + 0.5f);
		q.z = Math::Sqrt(matrix->m33*0.5f + 0.5f);
		if(q.x > 0.0001f)
		{
			q.y = copysign(q.y, matrix->m12);
			q.z = copysign(q.z, matrix->m13);
		}
		else if(q.y > 0.0001f)
		{
			q.x = copysign(q.x, matrix->m12);
			q.z = copysign(q.z, matrix->m23);
		}
		else if(q.z > 0.0001f)
		{
			q.y = copysign(q.y, matrix->m23);
			q.x = copysign(q.x, matrix->m13);
		}
	}
	return q;
}

void Quaternion::CreateFromRotationMatrix(const Matrix *const matrix, Quaternion *const result)
{
	result->w = Math::Sqrt(matrix->m11 + matrix->m22 + matrix->m33 + 1.0f) * 0.5f;

	if(result->w > 0.0001f)
	{
		const float cw = 1.0f / (4.0f * result->w);
		result->x = (matrix->m32 - matrix->m23) * cw;
		result->y = (matrix->m13 - matrix->m31) * cw;
		result->z = (matrix->m21 - matrix->m12) * cw;
	}
	else
	{
		result->x = Math::Sqrt(matrix->m11*0.5f + 0.5f);
		result->y = Math::Sqrt(matrix->m22*0.5f + 0.5f);
		result->z = Math::Sqrt(matrix->m33*0.5f + 0.5f);
		if(result->x > 0.0001f)
		{
			result->y = copysign(result->y, matrix->m12);
			result->z = copysign(result->z, matrix->m13);
		}
		else if(result->y > 0.0001f)
		{
			result->x = copysign(result->x, matrix->m12);
			result->z = copysign(result->z, matrix->m23);
		}
		else if(result->z > 0.0001f)
		{
			result->y = copysign(result->y, matrix->m23);
			result->x = copysign(result->x, matrix->m13);
		}
	}
}

Quaternion	Quaternion::CreateFromDirection(const Vector3 *const direction)
{
	const float y_half_angle = atan2f(-direction->x, -direction->z) * 0.5f;
	const float y_sin = sinf(y_half_angle);
	const float y_cos = cosf(y_half_angle);
	
	const float x_half_angle = asinf(direction->y) * 0.5f;
	const float x_sin = sinf(x_half_angle);
	const float x_cos = cosf(x_half_angle);
	
	return Quaternion
	(
		y_cos * x_sin,
		y_sin * x_cos,
		-y_sin * x_sin,
		y_cos * x_cos
	);
}

void Quaternion::CreateFromDirection(const Vector3 *const direction, Quaternion *const result)
{
	const float y_half_angle = atan2f(-direction->x, -direction->z) * 0.5f;
	const float y_sin = sinf(y_half_angle);
	const float y_cos = cosf(y_half_angle);
	
	const float x_half_angle = asinf(direction->y) * 0.5f;
	const float x_sin = sinf(x_half_angle);
	const float x_cos = cosf(x_half_angle);
	
	result->x =	y_cos * x_sin;
	result->y =	y_sin * x_cos;
	result->z = -y_sin * x_sin;
	result->w = y_cos * x_cos;
}

Quaternion	Quaternion::CreateRotationX(const float angle)
{
	const float half_angle = angle * 0.5f;
	return Quaternion
	(
		sinf(half_angle),
		0.f,
		0.f,
		cosf(half_angle)
	);
}

void Quaternion::CreateRotationX(const float angle, Quaternion *const result)
{
	const float half_angle = angle * 0.5f;
	
	result->x = sinf(half_angle);
	result->y = 0.f;
	result->z = 0.f;
	result->w = cosf(half_angle);
}

Quaternion	Quaternion::CreateRotationY(const float angle)
{
	const float half_angle = angle * 0.5f;
	return Quaternion
	(
		0.f,
		sinf(half_angle),
		0.f,
		cosf(half_angle)
	);
}

void Quaternion::CreateRotationY(const float angle, Quaternion *const result)
{
	const float half_angle = angle * 0.5f;
	
	result->x = 0.f;
	result->y = sinf(half_angle);
	result->z = 0.f;
	result->w = cosf(half_angle);
}

Quaternion	Quaternion::CreateRotationZ(const float angle)
{
	const float half_angle = angle * 0.5f;
	return Quaternion
	(
		0.f,
		0.f,
		sinf(half_angle),
		cosf(half_angle)
	);
}

void Quaternion::CreateRotationZ(const float angle, Quaternion *const result)
{
	const float half_angle = angle * 0.5f;
	
	result->x = 0.f;
	result->y = 0.f;
	result->z = sinf(half_angle);
	result->w = cosf(half_angle);
}

Quaternion	Quaternion::Concatenate(const Quaternion *const q1, const Quaternion *const q2)
{
	return Quaternion
	(
		q1->w*q2->x + q1->x*q2->w + q1->y*q2->z - q1->z*q2->y,
		q1->w*q2->y + q1->y*q2->w + q1->z*q2->x - q1->x*q2->z,
		q1->w*q2->z + q1->z*q2->w + q1->x*q2->y - q1->y*q2->x,
		q1->w*q2->w - q1->x*q2->x - q1->y*q2->y - q1->z*q2->z
	);
}

void Quaternion::Concatenate(const Quaternion *const q1, const Quaternion *const q2, Quaternion *const result)
{
	result->x = q1->w*q2->x + q1->x*q2->w + q1->y*q2->z - q1->z*q2->y;
	result->y = q1->w*q2->y + q1->y*q2->w + q1->z*q2->x - q1->x*q2->z;
	result->z = q1->w*q2->z + q1->z*q2->w + q1->x*q2->y - q1->y*q2->x;
	result->w = q1->w*q2->w - q1->x*q2->x - q1->y*q2->y - q1->z*q2->z;
}

void Quaternion::operator *= (const Quaternion &q)
{
	const float rx = w*q.x + x*q.w + y*q.z - z*q.y;
	const float ry = w*q.y + y*q.w + z*q.x - x*q.z;
	const float rz = w*q.z + z*q.w + x*q.y - y*q.x;
	const float rw = w*q.w - x*q.x - y*q.y - z*q.z;
	this->x = rx;
	this->y = ry;
	this->z = rz;
	this->w = rw;
}

Quaternion	Quaternion::Conjugate(const Quaternion *const q)
{
	return Quaternion(-q->x, -q->y, -q->z, q->w);
}

void Quaternion::Conjugate(const Quaternion *const q, Quaternion *const result)
{
	result->x = -q->x;
	result->y = -q->y;
	result->z = -q->z;
	result->w = q->w;
}

Quaternion	Quaternion::Slerp(const Quaternion *const q1, const Quaternion *const q2, const float ammount)
{
	if(ammount <= 0.f)	return *q1;
	if(ammount >= 1.f)	return *q2;
		
	Quaternion _q2 = *q2;
	float cosOmega = q1->x*q2->x + q1->y*q2->y + q1->z*q2->z + q1->w*q2->w;
	float k0, k1;
	
	if(cosOmega < 0.f)
	{
		_q2.x = -_q2.x;
		_q2.y = -_q2.y;
		_q2.z = -_q2.z;
		_q2.w = -_q2.w;
		cosOmega = -cosOmega;
	}	
	
	if(cosOmega > 0.9999f)
	{
		k0 = 1.f - ammount;
		k1 = ammount;
	}
	else 
	{
		const float omega = acosf(cosOmega);
		const float sinOmega = sinf(omega);
		k0 = sinf((1.f - ammount) * omega) / sinOmega;
		k1 = sinf(ammount * omega) / sinOmega;
	}
	return Quaternion(k0*q1->x + k1*_q2.x,
					  k0*q1->y + k1*_q2.y,
					  k0*q1->z + k1*_q2.z,
					  k0*q1->w + k1*_q2.w);
}

void Quaternion::Slerp(const Quaternion *const q1, const Quaternion *const q2, const float ammount, Quaternion *const result)
{
	if(ammount <= 0.f)
	{	
		*result = *q1;
		return;
	}
	if(ammount >= 1.f)
	{
		*result = *q2;
		return;
	}
	
	Quaternion _q2 = *q2;
	float cosOmega = q1->x*q2->x + q1->y*q2->y + q1->z*q2->z + q1->w*q2->w;
	float k0, k1;
	
	if(cosOmega < 0.f)
	{
		_q2.x = -_q2.x;
		_q2.y = -_q2.y;
		_q2.z = -_q2.z;
		_q2.w = -_q2.w;
		cosOmega = -cosOmega;
	}	
	
	if(cosOmega > 0.9999f)
	{
		k0 = 1.f - ammount;
		k1 = ammount;
	}
	else 
	{
		const float omega = acosf(cosOmega);
		const float sinOmega = sinf(omega);
		k0 = sinf((1.f - ammount) * omega) / sinOmega;
		k1 = sinf(ammount * omega) / sinOmega;
	}
	result->x = k0*q1->x + k1*_q2.x;
	result->y = k0*q1->y + k1*_q2.y;
	result->z = k0*q1->z + k1*_q2.z;
	result->w = k0*q1->w + k1*_q2.w;
}

float Quaternion::Dot(const Quaternion *const q1, const Quaternion *const q2)
{
	return q1->x*q2->x + q1->y*q2->y + q1->z*q2->z + q1->w*q2->w;
}

Quaternion Quaternion::RotateX(const Quaternion *const q, const float angle)
{
	const float half_angle = angle * 0.5f;
	const float _x = sinf(half_angle);
	const float _w = cosf(half_angle);
	
	return Quaternion
	(
		q->w * _x + q->x * _w,
		q->y * _w + q->z * _x,
		q->z * _w - q->y * _x,
		q->w * _w - q->x * _x
	);
}

void Quaternion::RotateX(const Quaternion *const q, const float angle, Quaternion *const result)
{
	const float half_angle = angle * 0.5f;
	const float _x = sinf(half_angle);
	const float _w = cosf(half_angle);
	
	result->x = q->w * _x + q->x * _w;
	result->y = q->y * _w + q->z * _x;
	result->z = q->z * _w - q->y * _x;
	result->w = q->w * _w - q->x * _x;
}

Quaternion Quaternion::RotateY(const Quaternion *const q, const float angle)
{
	const float half_angle = angle * 0.5f;
	const float _y = sinf(half_angle);
	const float _w = cosf(half_angle);
	
	return Quaternion
	(
		q->x * _w - q->z * _y,
		q->w * _y + q->y * _w,
		q->z * _w + q->x * _y,
		q->w * _w - q->y * _y
	);
}

void Quaternion::RotateY(const Quaternion *const q, const float angle, Quaternion *const result)
{
	const float half_angle = angle * 0.5f;
	const float _y = sinf(half_angle);
	const float _w = cosf(half_angle);
	
	result->x = q->x * _w - q->z * _y;
	result->y = q->w * _y + q->y * _w;
	result->z = q->z * _w + q->x * _y;
	result->w = q->w * _w - q->y * _y;
}

Quaternion Quaternion::RotateZ(const Quaternion *const q, const float angle)
{
	const float half_angle = angle * 0.5f;
	const float _z = sinf(half_angle);
	const float _w = cosf(half_angle);
	
	return Quaternion
	(
		q->x * _w + q->y * _z,
		q->y * _w - q->x * _z,
		q->w * _z + q->z * _w,
		q->w * _w - q->z * _z
	);
}

void Quaternion::RotateZ(const Quaternion *const q, const float angle, Quaternion *const result)
{
	const float half_angle = angle * 0.5f;
	const float _z = sinf(half_angle);
	const float _w = cosf(half_angle);
	
	result->x = q->x * _w + q->y * _z;
	result->y = q->y * _w - q->x * _z;
	result->z = q->w * _z + q->z * _w;
	result->w = q->w * _w - q->z * _z;	
}

Quaternion Quaternion::Rotate(const Quaternion *const q, const Vector3 *const axis, const float angle)
{
	const float half_angle = angle * 0.5f;
	const float s = sinf(half_angle);
	
	const float q2x = axis->x * s;
	const float q2y = axis->y * s;
	const float q2z = axis->z * s;
	const float q2w = cosf(half_angle);
	
	return Quaternion
	(
		q->w*q2x + q->x*q2w + q->y*q2z - q->z*q2y,
		q->w*q2y + q->y*q2w + q->z*q2x - q->x*q2z,
		q->w*q2z + q->z*q2w + q->x*q2y - q->y*q2x,
		q->w*q2w - q->x*q2x - q->y*q2y - q->z*q2z
	);
}

void Quaternion::Rotate(const Quaternion *const q, const Vector3 *const axis, const float angle, Quaternion *const result)
{
	const float half_angle = angle * 0.5f;
	const float s = sinf(half_angle);
	
	const float q2x = axis->x * s;
	const float q2y = axis->y * s;
	const float q2z = axis->z * s;
	const float q2w = cosf(half_angle);
	
	result->x = q->w*q2x + q->x*q2w + q->y*q2z - q->z*q2y;
	result->y = q->w*q2y + q->y*q2w + q->z*q2x - q->x*q2z;
	result->z = q->w*q2z + q->z*q2w + q->x*q2y - q->y*q2x;
	result->w = q->w*q2w - q->x*q2x - q->y*q2y - q->z*q2z;
}

