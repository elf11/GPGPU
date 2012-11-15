#pragma once

class Matrix;
class Vector3;

class Vector2
{
public:
	float x;
	float y;
	
	static const Vector2 UnitX;
	static const Vector2 UnitY;
	static const Vector2 Zero;
	static const Vector2 One;
	
	inline Vector2()								{}
	inline Vector2(const float X, const float Y)	: x(X), y(Y) {}
	~Vector2() {}
	
	inline float * Ptr() { return &x; }
	inline const float * Ptr() const { return &x; }
	
	void	Normalize();
	void	Negate();
	
	float	Length()		const;
	float	LengthReversed()const;
	float	LengthSquared() const;
	float	Angle()			const;
	
	static float	Distance(const Vector2 *const v1, const Vector2 *const v2);
	static float	Distance2(const Vector2 *const v1, const Vector2 *const v2);
	
	static float	Dot(const Vector2 *const v1, const Vector2 *const v2);
	static float    DotNorm(const Vector2 *const v1, const Vector2 *const p1v2, const Vector2 *const p2v2);
    
	static void		Add(const Vector2 *const v, const float val, Vector2 *const result);
	static Vector2	Add(const Vector2 *const v, const float val);
	static void		Add(const Vector2 *const v1, const Vector2 *const v2, Vector2 *const result);
	static Vector2	Add(const Vector2 *const v1, const Vector2 *const v2);
	static void		Subtract(const Vector2 *const v, const float val, Vector2 *const result);
	static Vector2	Subtract(const Vector2 *const v, const float val);
	static void		Subtract(const Vector2 *const v1, const Vector2 *const v2, Vector2 *const result);
	static Vector2	Subtract(const Vector2 *const v1, const Vector2 *const v2);
	static void		Divide(const Vector2 *const v, const float val, Vector2 *const result);
	static Vector2	Divide(const Vector2 *const v, const float val);
	static void		Divide(const Vector2 *const v1, const Vector2 *const v2, Vector2 *const result);
	static Vector2	Divide(const Vector2 *const v1, const Vector2 *const v2);
	static void		Multiply(const Vector2 *const v, const float val, Vector2 *const result);
	static Vector2	Multiply(const Vector2 *const v, const float val);
	static void		Multiply(const Vector2 *const v1, const Vector2 *const v2, Vector2 *const result);
	static Vector2	Multiply(const Vector2 *const v1, const Vector2 *const v2);
	static void		Negate(const Vector2 *const v, Vector2 *const result);
	static Vector2	Negate(const Vector2 *const v);
	static void		Normalize(const Vector2 *const v, Vector2 *const result);
	static Vector2	Normalize(const Vector2 *const v);
	
	static void		Lerp(const Vector2 *const v1, const Vector2 *const v2, const float ammount, Vector2 *const result);
	static void		Clamp(const Vector2 *const v, const Vector2 *const min, const Vector2 *const max, Vector2 *const result);
	static void		Clamp(const Vector2 *const v, const float min, const float max, Vector2 *const result);
	
	static void		Max(const Vector2 *const v1, const Vector2 *const v2, Vector2 *const result);
	static void		Min(const Vector2 *const v1, const Vector2 *const v2, Vector2 *const result);
	
	static void		Reflect(const Vector2 *const v, const Vector2 *const normal, Vector2 *const result);
	static void		Transform(const Vector2 *const position, const Matrix *const mat, Vector2 *const result);
	static void		TransformNormal(const Vector2 *const direction, const Matrix *const mat, Vector2 *const result);
	
	void operator += (float val);
	void operator += (const Vector2 &v);
	void operator -= (float val);
	void operator -= (const Vector2 &v);
	void operator *= (float val);
	void operator *= (const Vector2 &v);
	void operator /= (float val);
	void operator /= (const Vector2 &v);
    
    Vector2& operator  = (const Vector2 &v);
	Vector2& operator  = (const Vector3 &v);

	inline void Set(const float x, const float y)
	{
		this->x = x;
		this->y = y;
	}
	
	inline void Set(const float v)
	{
		this->x = v;
		this->y = v;
	}

	void Randomize(float rx, float ry);
	static Vector2 Random(float rx, float ry) {Vector2 r; r.Randomize(rx, ry); return r;}
	static Vector2 Random(const Vector2* vec) {Vector2 r; r.Randomize(vec->x, vec->y); return r;}

};


