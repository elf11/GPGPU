#pragma once

class Matrix;
class Quaternion;
class Vector2;

//#pragma pack(1)
class Vector3
{
public:
	float x;
	float y;
	float z;
	
	static const Vector3 Left;
	static const Vector3 Right;
	static const Vector3 Up;
	static const Vector3 Down;
	static const Vector3 Backward;
	static const Vector3 Forward;
	static const Vector3 UnitX;
	static const Vector3 UnitY;
	static const Vector3 UnitZ;
	static const Vector3 Zero;
	static const Vector3 One;
    static const Vector3 Luma;
	
	inline Vector3()												{}
	inline Vector3(const float X, const float Y, const float Z)	: x(X), y(Y), z(Z) {}
	~Vector3() {}
	
	inline float * Ptr() { return &x; }
	inline const float * Ptr() const { return &x; }
	
	void	Normalize();
	void	Negate();
	
	float	Length()		const;
	float	LengthReversed()const;
	float	LengthSquared() const;
	float	Yaw()			const;
	float	Pitch()			const;
	
	static float	Dot(const Vector3 *const v1, const Vector3 *const v2);
    static float    DotNorm(const Vector3 *const v1, const Vector3 *const p1v2, const Vector3 *const p2v2);
    
	static void		Cross(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result);
	
	static float	Distance(const Vector3 *const v1, const Vector3 *const v2);
	static float	Distance2(const Vector3 *const v1, const Vector3 *const v2);

    static float	Distance(const Vector3 & v1, const Vector3 & v2);
	static float	Distance2(const Vector3 & v1, const Vector3 & v2);
	
	static void		Add(const Vector3 *const v, const float val, Vector3 *const result);
	static Vector3	Add(const Vector3 *const v, const float val);
	static void		Add(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result);
	static Vector3	Add(const Vector3 *const v1, const Vector3 *const v2);
	static void		Subtract(const Vector3 *const v, const float val, Vector3 *const result);
	static Vector3	Subtract(const Vector3 *const v, const float val);
	static void		Subtract(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result);
	static Vector3	Subtract(const Vector3 *const v1, const Vector3 *const v2);
	static void		Divide(const Vector3 *const v, const float val, Vector3 *const result);
	static Vector3	Divide(const Vector3 *const v, const float val);
	static void		Divide(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result);
	static Vector3	Divide(const Vector3 *const v1, const Vector3 *const v2);
	static void		Multiply(const Vector3 *const v, const float val, Vector3 *const result);
	static Vector3	Multiply(const Vector3 *const v, const float val);
	static void		Multiply(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result);
	static Vector3	Multiply(const Vector3 *const v1, const Vector3 *const v2);
	static void		Negate(const Vector3 *const v, Vector3 *const result);
	static Vector3	Negate(const Vector3 *const v);
	static void		Normalize(const Vector3 *const v, Vector3 *const result);
	static Vector3	Normalize(const Vector3 *const v);
	
	static void		Lerp(const Vector3 *const v1, const Vector3 *const v2, const float ammount, Vector3 *const result);
	static void		Clamp(const Vector3 *const v, const Vector3 *const min, const Vector3 *const max, Vector3 *const result);
	static void		Clamp(const Vector3 *const v, const float min, const float max, Vector3 *const result);
	static void		Max(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result);
	static void		Min(const Vector3 *const v1, const Vector3 *const v2, Vector3 *const result);
	
	static void		Reflect(const Vector3 *const v, const Vector3 *const normal, Vector3 *const result);
	
	static void		Transform(const Vector3 *const position, const Matrix *const mat, Vector3 *const result);
	static void		TransformNormal(const Vector3 *const direction, const Matrix *const mat, Vector3 *const result);
	static void		TransformNormal(const Vector3 *const direction, const Quaternion *const quaternion, Vector3 *const result);
	static void		TransformInverseNormal(const Vector3 *const direction, const Matrix *const mat, Vector3 *const result);
	static void		TransformInverseNormal(const Vector3 *const direction, const Quaternion *const quaternion, Vector3 *const result);

	void Randomize(float rx, float ry, float rz);
	void RandomizeNNInt(float minx, float miny, float minz, float maxx, float maxy, float maxz);
	static Vector3 Random(float rx, float ry, float rz) {Vector3 r; r.Randomize(rx, ry, rz); return r;}
	static Vector3 RandomNNInt(float minx, float miny, float minz, float maxx, float maxy, float maxz) {Vector3 r; r.RandomizeNNInt(minx, miny, minz, maxx, maxy, maxz); return r;}
	static Vector3 Random(const Vector3* vec) {Vector3 r; r.Randomize(vec->x, vec->y, vec->z); return r;}
	
	void operator += (float val);
	void operator += (const Vector3 &v);
	void operator -= (float val);
	void operator -= (const Vector3 &v);
	void operator *= (float val);
	void operator *= (const Vector3 &v);
	void operator /= (float val);
	void operator /= (const Vector3 &v);
	Vector3& operator  = (const Vector3 &v);
    Vector3& operator  = (const Vector2 &v);
	
	inline void Set(const float x, const float y, const float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}
	
	inline void Set(const float v)
	{
		this->x = v;
		this->y = v;
		this->z = v;
	}
};

//#pragma pack()
