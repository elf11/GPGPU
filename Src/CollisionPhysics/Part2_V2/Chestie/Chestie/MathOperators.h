#pragma once

struct RotationX
{
	const float angle;
	RotationX(const float rad) : angle(rad) {}
};

struct RotationY
{
	const float angle;
	RotationY(const float rad) : angle(rad) {}
};

struct RotationZ
{
	float const angle;
	RotationZ(const float rad) : angle(rad) {}
};

struct Translation
{
	const float x;
	const float y;
	const float z;
	Translation(const float X, const float Y, const float Z) : x(X), y(Y), z(Z) {} 
};

struct Scale
{
	const float x;
	const float y;
	const float z;
	Scale(const float X, const float Y, const float Z) : x(X), y(Y), z(Z) {} 
};

inline Vector2 operator + (const Vector2 &v1, const Vector2 &v2)	{ return Vector2::Add(&v1, &v2); }
inline Vector2 operator + (const Vector2 &v1, const float val)		{ return Vector2::Add(&v1, val); }
inline Vector2 operator + (const float val, const Vector2 &v1)		{ return Vector2::Add(&v1, val); }
inline Vector2 operator - (const Vector2 &v1, const Vector2 &v2)	{ return Vector2::Subtract(&v1, &v2); }
inline Vector2 operator - (const Vector2 &v1, const float val)		{ return Vector2::Subtract(&v1, val); }
inline Vector2 operator - (const float val, const Vector2 &v1)		{ return Vector2::Subtract(&v1, val); }
inline Vector2 operator * (const Vector2 &v1, const Vector2 &v2)	{ return Vector2::Multiply(&v1, &v2); }
inline Vector2 operator * (const Vector2 &v1, const float val)		{ return Vector2::Multiply(&v1, val); }
inline Vector2 operator * (const float val, const Vector2 &v1)		{ return Vector2::Multiply(&v1, val); }
inline Vector2 operator / (const Vector2 &v1, const Vector2 &v2)	{ return Vector2::Divide(&v1, &v2); }
inline Vector2 operator / (const Vector2 &v1, const float val)		{ return Vector2::Divide(&v1, val); }
inline Vector2 operator / (const float val, const Vector2 &v1)		{ return Vector2::Divide(&v1, val); }
inline Vector2 operator - (const Vector2 &v)						{ return Vector2::Negate(&v); }

inline Vector3 operator + (const Vector3 &v1, const Vector3 &v2)	{ return Vector3::Add(&v1, &v2); }
inline Vector3 operator + (const Vector3 &v1, const float val)		{ return Vector3::Add(&v1, val); }
inline Vector3 operator + (const float val, const Vector3 &v1)		{ return Vector3::Add(&v1, val); }
inline Vector3 operator - (const Vector3 &v1, const Vector3 &v2)	{ return Vector3::Subtract(&v1, &v2); }
inline Vector3 operator - (const Vector3 &v1, const float val)		{ return Vector3::Subtract(&v1, val); }
inline Vector3 operator - (const float val, const Vector3 &v1)		{ return Vector3::Subtract(&v1, val); }
inline Vector3 operator * (const Vector3 &v1, const Vector3 &v2)	{ return Vector3::Multiply(&v1, &v2); }
inline Vector3 operator * (const Vector3 &v1, const float val)		{ return Vector3::Multiply(&v1, val); }
inline Vector3 operator * (const float val, const Vector3 &v1)		{ return Vector3::Multiply(&v1, val); }
inline Vector3 operator / (const Vector3 &v1, const Vector3 &v2)	{ return Vector3::Divide(&v1, &v2); }
inline Vector3 operator / (const Vector3 &v1, const float val)		{ return Vector3::Divide(&v1, val); }
inline Vector3 operator / (const float val, const Vector3 &v1)		{ return Vector3::Divide(&v1, val); }
inline Vector3 operator - (const Vector3 &v)						{ return Vector3::Negate(&v); }

inline Matrix operator *(const Matrix &mat1, const Matrix &mat2)		{ return Matrix::Multiply(&mat1, &mat2); }
inline Matrix operator *(const Matrix &mat, const RotationX &rot)		{ return Matrix::RotateX(&mat, rot.angle); }
inline Matrix operator *(const Matrix &mat, const RotationY &rot)		{ return Matrix::RotateY(&mat, rot.angle); }
inline Matrix operator *(const Matrix &mat, const RotationZ &rot)		{ return Matrix::RotateZ(&mat, rot.angle); }
inline Matrix operator *(const Matrix &mat, const Translation &trans)	{ return Matrix::Translate(&mat, (const Vector3 *) &trans); }
inline Matrix operator *(const Matrix &mat, const Scale &scales)		{ return Matrix::Scale(&mat, (const Vector3 *) &scales); }

inline Vector3 operator *(const Matrix &mat1, const Vector3 &v1)		{ return Matrix::Multiply(&mat1, &v1); }

inline Matrix operator *(const MatrixStack &mat1, const Matrix &mat2)		{ return Matrix::Multiply(&mat1, &mat2); }
inline Matrix operator *(const Matrix &mat1, const MatrixStack &mat2)		{ return Matrix::Multiply(&mat1, &mat2); }
inline Matrix operator *(const MatrixStack &mat1, const MatrixStack &mat2)		{ return Matrix::Multiply(&mat1, &mat2); }
inline Matrix operator *(const MatrixStack &mat, const RotationX &rot)		{ return Matrix::RotateX(&mat, rot.angle); }
inline Matrix operator *(const MatrixStack &mat, const RotationY &rot)		{ return Matrix::RotateY(&mat, rot.angle); }
inline Matrix operator *(const MatrixStack &mat, const RotationZ &rot)		{ return Matrix::RotateZ(&mat, rot.angle); }
inline Matrix operator *(const MatrixStack &mat, const Translation &trans)	{ return Matrix::Translate(&mat, (const Vector3 *) &trans); }
inline Matrix operator *(const MatrixStack &mat, const Scale &scales)		{ return Matrix::Scale(&mat, (const Vector3 *) &scales); }

inline Quaternion operator *(const Quaternion &q1, const Quaternion &q2)	{ return Quaternion::Concatenate(&q1, &q2); }
inline Quaternion operator *(const Quaternion &q, const RotationX &rot)		{ return Quaternion::RotateX(&q, rot.angle); }
inline Quaternion operator *(const Quaternion &q, const RotationY &rot)		{ return Quaternion::RotateY(&q, rot.angle); }
inline Quaternion operator *(const Quaternion &q, const RotationZ &rot)		{ return Quaternion::RotateZ(&q, rot.angle); }
