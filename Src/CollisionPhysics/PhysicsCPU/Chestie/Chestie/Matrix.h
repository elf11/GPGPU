#pragma once

class Vector3;
class Quaternion;
class Plane;

class Matrix
{
public:
	//Members
	float m11; float m21; float m31; float m41;
	float m12; float m22; float m32; float m42;
	float m13; float m23; float m33; float m43;
	float m14; float m24; float m34; float m44;
	
	//Constructors
	inline Matrix() { /*(*this) = Identity;*/ }
	inline Matrix(float M11, float M12, float M13, float M14,
				  float M21, float M22, float M23, float M24,
				  float M31, float M32, float M33, float M34,
				  float M41, float M42, float M43, float M44) :
		m11(M11), m12(M12), m13(M13), m14(M14),
		m21(M21), m22(M22), m23(M23), m24(M24),
		m31(M31), m32(M32), m33(M33), m34(M34),
		m41(M41), m42(M42), m43(M43), m44(M44) {}
	~Matrix() {}
	
	inline float * Ptr() { return &m11; }
	inline const float * Ptr() const { return &m11; }
	
	//Member functions - Properties
	float					Determinant() const;
	
	inline Vector3 &		Right()					{ return (Vector3 &) m11; }
	inline Vector3 &		Up()					{ return (Vector3 &) m12; }
	inline Vector3 &		Backward()				{ return (Vector3 &) m13; }
	inline Vector3 &		Translation()			{ return (Vector3 &) m14; }
	
	inline const Vector3 &	Right()			const	{ return (const Vector3 &) m11; }
	inline const Vector3 &	Up()			const	{ return (const Vector3 &) m12; }
	inline const Vector3 &	Backward()		const	{ return (const Vector3 &) m13; }
	inline const Vector3 &	Translation()	const	{ return (const Vector3 &) m14; }
	
	//Member functions - In-place transforms
	Matrix &				Translate(const float xPosition, const float yPosition, const float zPosition);
	Matrix &				Translate(const Vector3 *const position);
	inline Matrix &			Translate(const Vector3 &position)
							{ return Translate(&position); }
	
	Matrix &				RotateX(const float angle);
	Matrix &				RotateY(const float angle);
	Matrix &				RotateZ(const float angle);
	Matrix &				Rotate(const float axisX, const float axisY, const float axisZ, const float angle);
	Matrix &				Rotate(const Vector3 *const axis, const float angle);
	inline Matrix &			Rotate(const Vector3 &axis, const float angle)
							{ return Rotate(&axis, angle); }
	
	Matrix &				Scale(const float xScale, const float yScale, const float zScale);
	Matrix &				Scale(const float scale);
	Matrix &				Scale(const Vector3 *const scales);
	inline Matrix &			Scale(const Vector3 &scales)
							{ return Scale(&scales); }
	
	Matrix &				Transpose();
	Matrix &				Invert();
	Matrix &				Invert3x3();
	
	//Statics values
	static const Matrix Identity;
	static const Matrix Bias;
	
	//Static - Constructors
	static Matrix	CreateFromAxisAngle(const Vector3 *const axis, const float angle);
	static void		CreateFromAxisAngle(const Vector3 *const axis, const float angle, Matrix *const result);
	static Matrix	CreateFromQuaternion(const Quaternion *const quaternion);
	static void		CreateFromQuaternion(const Quaternion *const quaternion, Matrix *const result);
	static Matrix	CreateFromQuaternionTranslation(const Quaternion*const quaternion, const Vector3*const pos);
	static void		CreateFromQuaternionTranslation(const Quaternion*const quaternion, const Vector3*const pos, Matrix *const result);
	static Matrix	CreateFromQuaternionTranslationScale(const Quaternion*const quaternion, const Vector3*const pos, const Vector3 *const scale);
	static void		CreateFromQuaternionTranslationScale(const Quaternion*const quaternion, const Vector3*const pos, const Vector3 *const scale, Matrix *const result);
	static Matrix	CreateFromDirection(const Vector3 *const direction);
	static void		CreateFromDirection(const Vector3 *const direction, Matrix *const result);
	static Matrix	CreateFromYawPitchRoll(const float yaw, const float pitch, const float roll);
	static void		CreateFromYawPitchRoll(const float yaw, const float pitch, const float roll, Matrix *const result);
	static Matrix	CreateTranslation(const float xPosition, const float yPosition, const float zPosition);
	static void		CreateTranslation(const float xPosition, const float yPosition, const float zPosition, Matrix *const result);
	static Matrix	CreateTranslation(const Vector3 *const position);
	static void		CreateTranslation(const Vector3 *const position, Matrix *const result);
	static Matrix	CreateRotationX(const float angle);
	static void		CreateRotationX(const float angle, Matrix *const result);
	static Matrix	CreateRotationY(const float angle);
	static void		CreateRotationY(const float angle, Matrix *const result);
	static Matrix	CreateRotationZ(const float angle);
	static void		CreateRotationZ(const float angle, Matrix *const result);
	static Matrix	CreateScale(const float scale);
	static void		CreateScale(const float scale, Matrix *const result);
	static Matrix	CreateScale(const float xScale, const float yScale, const float zScale);
	static void		CreateScale(const float xScale, const float yScale, const float zScale, Matrix *const result);
	static Matrix	CreateScale(const Vector3 *const scales);
	static void		CreateScale(const Vector3 *const scales, Matrix *const result);
    static Matrix   CreateBias(const float scale, const float offset);
    static void     CreateBias(const float scale, const float offset, Matrix *const result);
    static Matrix   CreateBias(const Vector3 *const scale, const Vector3 *const offset);
    static void     CreateBias(const Vector3 *const scale, const Vector3 *const offset, Matrix *const result);
	static Matrix	CreateWorld(const Vector3 *const position, const Vector3 *const forward, const Vector3 *const up);
	static void		CreateWorld(const Vector3 *const position, const Vector3 *const forward, const Vector3 *const up, Matrix *const result);
	static Matrix	CreateWorldN(const Vector3 *const position, const Vector3 *const forward, const Vector3 *const up);
	static Matrix	CreateLookAt(const Vector3 *const eye, const Vector3 *const center, const Vector3 *const up);
	static void		CreateLookAt(const Vector3 *const eye, const Vector3 *const center, const Vector3 *const up, Matrix *const result);
	static Matrix	CreatePerspective(const float fieldOfView, const float aspectRatio, const float nearPlaneDistance, const float farPlaneDistance);
	static void		CreatePerspective(const float fieldOfView, const float aspectRatio, const float nearPlaneDistance, const float farPlaneDistance, Matrix *const result);
	static Matrix	CreatePerspectiveInfinite(const float fieldOfView, const float aspectRatio);
	static void		CreatePerspectiveInfinite(const float fieldOfView, const float aspectRatio, Matrix *const result);
	static Matrix	CreatePerspectiveObliqueClipPlane(const Matrix *const projection, const Matrix *const view, const Plane *const clipPlaneWorldSpace);
	static void		CreatePerspectiveObliqueClipPlane(const Matrix *const projection, const Matrix *const view, const Plane *const clipPlaneWorldSpace, Matrix *const result);
	static Matrix	CreateOrthographic(const float left, const float right, const float bottom, const float top, const float zNearPlane, const float zFarPlane);
	static void		CreateOrthographic(const float left, const float right, const float bottom, const float top, const float zNearPlane, const float zFarPlane, Matrix *const result);
	static Matrix	CreatePlaneProjection(const Plane *const plane, const Vector3 *const v3, const float v3w);
	static void		CreatePlaneProjection(const Plane *const plane, const Vector3 *const v3, const float v3w, Matrix *const result);
	
	//Static - Transforms
	static Matrix	Transpose(const Matrix *const mat);
	static void		Transpose(const Matrix *const mat, Matrix *const result);
	static bool		Invert(const Matrix *const mat, Matrix *const result);
	static bool		Invert3x3(const Matrix *const mat, Matrix *const result);
	static bool		InverseTranslation(const Matrix *const mat, Vector3 *const result);
	
	static Matrix	Multiply(const Matrix *const mat1, const Matrix *const mat2);
	static void		Multiply(const Matrix *const mat1, const Matrix *const mat2, Matrix *const result);
	static Vector3  Multiply(const Matrix *const mat1, const Vector3 *const v);
	static void     Multiply(const Matrix *const mat1, const Vector3 *const v, Vector3 *const result);
	
	static Matrix	Translate(const Matrix *const mat, const float xPosition, const float yPosition, const float zPosition);
	static void		Translate(const Matrix *const mat, const float xPosition, const float yPosition, const float zPosition, Matrix *const result);
	static Matrix	Translate(const Matrix *const mat, const Vector3 *const position);
	static void		Translate(const Matrix *const mat, const Vector3 *const position, Matrix *const result);
	static Matrix	RotateX(const Matrix *const mat, const float angle);
	static void		RotateX(const Matrix *const mat, const float angle, Matrix *const result);
	static Matrix	RotateY(const Matrix *const mat, const float angle);
	static void		RotateY(const Matrix *const mat, const float angle, Matrix *const result);
	static Matrix	RotateZ(const Matrix *const mat, const float angle);
	static void		RotateZ(const Matrix *const mat, const float angle, Matrix *const result);
	static Matrix	Scale(const Matrix *const mat, const float scale);
	static void		Scale(const Matrix *const mat, const float scale, Matrix *const result);
	static Matrix	Scale(const Matrix *const mat, const float xScale, const float yScale, const float zScale);
	static void		Scale(const Matrix *const mat, const float xScale, const float yScale, const float zScale, Matrix *const result);
	static Matrix	Scale(const Matrix *const mat, const Vector3 *const scales);
	static void		Scale(const Matrix *const mat, const Vector3 *const scales, Matrix *const result);
};
