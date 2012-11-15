#include "RMath.h"
#include "Matrix.h"
#include "Vector3.h"
#include "Plane.h"
#include "Quaternion.h"
#include <math.h>

const Matrix Matrix::Identity = Matrix(1.f, 0.f, 0.f, 0.f,
									   0.f, 1.f, 0.f, 0.f,
									   0.f, 0.f, 1.f, 0.f,
									   0.f, 0.f, 0.f, 1.f);

const Matrix Matrix::Bias = Matrix(0.5f, 0.0f, 0.0f, 0.5f,
								   0.0f, 0.5f, 0.0f, 0.5f,
								   0.0f, 0.0f, 0.5f, 0.5f,
								   0.0f, 0.0f, 0.0f, 1.0f);

float Matrix::Determinant() const
{
	return	m11 * (+m22*m33*m44 - m22*m43*m34 - m23*m32*m44
				   +m23*m42*m34 + m24*m32*m43 - m24*m42*m33) +
			m21 * (-m12*m33*m44 + m12*m43*m34 + m13*m32*m44
				   -m13*m42*m34 - m14*m32*m43 + m14*m42*m33) +
			m31 * (+m12*m23*m44 - m12*m43*m24 - m13*m22*m44
				   +m13*m42*m24 + m14*m22*m43 - m14*m42*m23) +
			m41 * (-m12*m23*m34 + m12*m33*m24 + m13*m22*m34
				   -m13*m32*m24 - m14*m22*m33 + m14*m32*m23);
}

bool Matrix::Invert(const Matrix *const mat, Matrix *const result)
{
	float det;
	
	result->m11 = mat->m22*mat->m33*mat->m44 - mat->m22*mat->m43*mat->m34 - mat->m23*mat->m32*mat->m44
				+ mat->m23*mat->m42*mat->m34 + mat->m24*mat->m32*mat->m43 - mat->m24*mat->m42*mat->m33;
	result->m12 =-mat->m12*mat->m33*mat->m44 + mat->m12*mat->m43*mat->m34 + mat->m13*mat->m32*mat->m44
				- mat->m13*mat->m42*mat->m34 - mat->m14*mat->m32*mat->m43 + mat->m14*mat->m42*mat->m33;
	result->m13 = mat->m12*mat->m23*mat->m44 - mat->m12*mat->m43*mat->m24 - mat->m13*mat->m22*mat->m44
				+ mat->m13*mat->m42*mat->m24 + mat->m14*mat->m22*mat->m43 - mat->m14*mat->m42*mat->m23;
	result->m14 =-mat->m12*mat->m23*mat->m34 + mat->m12*mat->m33*mat->m24 + mat->m13*mat->m22*mat->m34
				- mat->m13*mat->m32*mat->m24 - mat->m14*mat->m22*mat->m33 + mat->m14*mat->m32*mat->m23;
	
	det = mat->m11*result->m11 + mat->m21*result->m12 + mat->m31*result->m13 + mat->m41*result->m14;
	if (det == 0)
        return false;
	
	result->m21 =-mat->m21*mat->m33*mat->m44 + mat->m21*mat->m43*mat->m34 + mat->m23*mat->m31*mat->m44
				- mat->m23*mat->m41*mat->m34 - mat->m24*mat->m31*mat->m43 + mat->m24*mat->m41*mat->m33;
	result->m22 = mat->m11*mat->m33*mat->m44 - mat->m11*mat->m43*mat->m34 - mat->m13*mat->m31*mat->m44
				+ mat->m13*mat->m41*mat->m34 + mat->m14*mat->m31*mat->m43 - mat->m14*mat->m41*mat->m33;
	result->m23 =-mat->m11*mat->m23*mat->m44 + mat->m11*mat->m43*mat->m24 + mat->m13*mat->m21*mat->m44
				- mat->m13*mat->m41*mat->m24 - mat->m14*mat->m21*mat->m43 + mat->m14*mat->m41*mat->m23;
	result->m24 = mat->m11*mat->m23*mat->m34 - mat->m11*mat->m33*mat->m24 - mat->m13*mat->m21*mat->m34
				+ mat->m13*mat->m31*mat->m24 + mat->m14*mat->m21*mat->m33 - mat->m14*mat->m31*mat->m23;
	result->m31 = mat->m21*mat->m32*mat->m44 - mat->m21*mat->m42*mat->m34 - mat->m22*mat->m31*mat->m44
				+ mat->m22*mat->m41*mat->m34 + mat->m24*mat->m31*mat->m42 - mat->m24*mat->m41*mat->m32;
	result->m32 =-mat->m11*mat->m32*mat->m44 + mat->m11*mat->m42*mat->m34 + mat->m12*mat->m31*mat->m44
				- mat->m12*mat->m41*mat->m34 - mat->m14*mat->m31*mat->m42 + mat->m14*mat->m41*mat->m32;
	result->m33 = mat->m11*mat->m22*mat->m44 - mat->m11*mat->m42*mat->m24 - mat->m12*mat->m21*mat->m44
				+ mat->m12*mat->m41*mat->m24 + mat->m14*mat->m21*mat->m42 - mat->m14*mat->m41*mat->m22;
	result->m34 =-mat->m11*mat->m22*mat->m34 + mat->m11*mat->m32*mat->m24 + mat->m12*mat->m21*mat->m34
				- mat->m12*mat->m31*mat->m24 - mat->m14*mat->m21*mat->m32 + mat->m14*mat->m31*mat->m22;
	result->m41 =-mat->m21*mat->m32*mat->m43 + mat->m21*mat->m42*mat->m33 + mat->m22*mat->m31*mat->m43
				- mat->m22*mat->m41*mat->m33 - mat->m23*mat->m31*mat->m42 + mat->m23*mat->m41*mat->m32;
	result->m42 = mat->m11*mat->m32*mat->m43 - mat->m11*mat->m42*mat->m33 - mat->m12*mat->m31*mat->m43
				+ mat->m12*mat->m41*mat->m33 + mat->m13*mat->m31*mat->m42 - mat->m13*mat->m41*mat->m32;
	result->m43 =-mat->m11*mat->m22*mat->m43 + mat->m11*mat->m42*mat->m23 + mat->m12*mat->m21*mat->m43
				- mat->m12*mat->m41*mat->m23 - mat->m13*mat->m21*mat->m42 + mat->m13*mat->m41*mat->m22;
	result->m44 = mat->m11*mat->m22*mat->m33 - mat->m11*mat->m32*mat->m23 - mat->m12*mat->m21*mat->m33
				+ mat->m12*mat->m31*mat->m23 + mat->m13*mat->m21*mat->m32 - mat->m13*mat->m31*mat->m22;
	
	det = 1.0 / det;
	
	result->m11 *= det;
	result->m21 *= det;
	result->m31 *= det;
	result->m41 *= det;
	result->m12 *= det;
	result->m22 *= det;
	result->m32 *= det;
	result->m42 *= det;
	result->m13 *= det;
	result->m23 *= det;
	result->m33 *= det;
	result->m43 *= det;
	result->m14 *= det;
	result->m24 *= det;
	result->m34 *= det;
	result->m44 *= det;
	
	return true;
}

bool Matrix::InverseTranslation(const Matrix *const mat, Vector3 *const result)
{	
	result->x =	-mat->m12*mat->m23*mat->m34 + mat->m12*mat->m33*mat->m24 + mat->m13*mat->m22*mat->m34
				-mat->m13*mat->m32*mat->m24 - mat->m14*mat->m22*mat->m33 + mat->m14*mat->m32*mat->m23;
	result->y = mat->m11*mat->m23*mat->m34 - mat->m11*mat->m33*mat->m24 - mat->m13*mat->m21*mat->m34
				+mat->m13*mat->m31*mat->m24 + mat->m14*mat->m21*mat->m33 - mat->m14*mat->m31*mat->m23;
	result->z = -mat->m11*mat->m22*mat->m34 + mat->m11*mat->m32*mat->m24 + mat->m12*mat->m21*mat->m34
				-mat->m12*mat->m31*mat->m24 - mat->m14*mat->m21*mat->m32 + mat->m14*mat->m31*mat->m22;
	float det = mat->m11*mat->m22*mat->m33 - mat->m11*mat->m32*mat->m23 - mat->m12*mat->m21*mat->m33
				+mat->m12*mat->m31*mat->m23 + mat->m13*mat->m21*mat->m32 - mat->m13*mat->m31*mat->m22;
	
	det = mat->m41*result->x + mat->m42*result->y + mat->m43*result->z + mat->m44*det;
	if (det == 0)
        return false;
	
	det = 1.0 / det;
	
	result->x *= det;
	result->y *= det;
	result->z *= det;
	return true;
}

bool Matrix::Invert3x3(const Matrix *const mat, Matrix *const result)
{
	const float det = mat->m11*mat->m22*mat->m33 + mat->m21*mat->m32*mat->m13 + mat->m12*mat->m23*mat->m31 - mat->m31*mat->m22*mat->m13 - mat->m11*mat->m32*mat->m23 - mat->m21*mat->m12*mat->m33;
	if(det != 0.f)
	{
		const float id = 1.0f / det;
		result->m11 = (mat->m22*mat->m33-mat->m23*mat->m32)*id;	result->m12 = (mat->m13*mat->m32-mat->m12*mat->m33)*id;	result->m13 = (mat->m12*mat->m23-mat->m13*mat->m22)*id;	result->m14 = 0;
		result->m21 = (mat->m23*mat->m31-mat->m21*mat->m33)*id;	result->m22 = (mat->m11*mat->m33-mat->m13*mat->m31)*id;	result->m23 = (mat->m13*mat->m21-mat->m11*mat->m23)*id;	result->m24 = 0;
		result->m31 = (mat->m21*mat->m32-mat->m22*mat->m31)*id;	result->m32 = (mat->m12*mat->m31-mat->m11*mat->m32)*id;	result->m33 = (mat->m11*mat->m22-mat->m12*mat->m21)*id;	result->m34 = 0;
		result->m41 = 0;										result->m42 = 0;										result->m43 = 0;										result->m44 = 1;
		return true;
	}
	return false;
}

Matrix Matrix::CreateFromQuaternion(const Quaternion *const quaternion)
{
	const float Nq = quaternion->x*quaternion->x + quaternion->y*quaternion->y + quaternion->z*quaternion->z + quaternion->w*quaternion->w;
	const float s = Nq > 0.f ? (2.f/Nq) : 0.f;
	
	const float X = quaternion->x*s;  const float Y = quaternion->y*s;  const float Z = quaternion->z*s;
	const float wX = quaternion->w*X; const float wY = quaternion->w*Y; const float wZ = quaternion->w*Z;
	const float xX = quaternion->x*X; const float xY = quaternion->x*Y; const float xZ = quaternion->x*Z;
	const float yY = quaternion->y*Y; const float yZ = quaternion->y*Z; const float zZ = quaternion->z*Z;
	
	return Matrix
	(
		1.f-yY-zZ,  xY-wZ,		xZ+wY,		0.f,
		xY+wZ,		1.f-xX-zZ,	yZ-wX,		0.f,
		xZ-wY,      yZ+wX,		1.f-xX-yY,	0.f,
		0.f,		0.f,		0.f,		1.f
	);
}

void Matrix::CreateFromQuaternion(const Quaternion *const quaternion, Matrix *const result)
{
	const float Nq = quaternion->x*quaternion->x + quaternion->y*quaternion->y + quaternion->z*quaternion->z + quaternion->w*quaternion->w;
	const float s = Nq > 0.f ? (2.f/Nq) : 0.f;
	
	const float X = quaternion->x*s;  const float Y = quaternion->y*s;  const float Z = quaternion->z*s;
	const float wX = quaternion->w*X; const float wY = quaternion->w*Y; const float wZ = quaternion->w*Z;
	const float xX = quaternion->x*X; const float xY = quaternion->x*Y; const float xZ = quaternion->x*Z;
	const float yY = quaternion->y*Y; const float yZ = quaternion->y*Z; const float zZ = quaternion->z*Z;
	

	result->m11 = 1.f-yY-zZ;	result->m12 = xY-wZ;		result->m13 = xZ+wY;		result->m14 = 0.f;
	result->m21 = xY+wZ;		result->m22 = 1.f-xX-zZ;	result->m23 = yZ-wX;		result->m24 = 0.f;
	result->m31 = xZ-wY;		result->m32 = yZ+wX;		result->m33 = 1.f-xX-yY;	result->m34 = 0.f;
	result->m41 = 0.f;			result->m42 = 0.f;			result->m43 = 0.f;			result->m44 = 1.f;
}

Matrix	Matrix::CreateFromAxisAngle(const Vector3 *const axis, const float angle)
{
	const float c = cosf(angle);	const float s = sinf(angle);	const float C = 1.f - c;
	const float xs = axis->x*s;		const float ys = axis->y*s;		const float zs = axis->z*s;
	const float xC = axis->x*C;		const float yC = axis->y*C;		const float zC = axis->z*C;
	const float xyC = axis->x*yC;	const float yzC = axis->y*zC;	const float zxC = axis->z*xC;

	return Matrix(axis->x*xC+c,	xyC-zs,			zxC+ys,			0.f,
				  xyC+zs,		axis->y*yC+c,   yzC-xs,			0.f,
				  zxC-ys,		yzC+xs,			axis->z*zC+c,	0.f,
				  0.f,			0.f,			0.f,			1.f);
}

void	Matrix::CreateFromAxisAngle(const Vector3 *const axis, const float angle, Matrix *const result)
{
	const float c = cosf(angle);	const float s = sinf(angle);	const float C = 1.f - c;
	const float xs = axis->x*s;		const float ys = axis->y*s;		const float zs = axis->z*s;
	const float xC = axis->x*C;		const float yC = axis->y*C;		const float zC = axis->z*C;
	const float xyC = axis->x*yC;	const float yzC = axis->y*zC;	const float zxC = axis->z*xC;
	result->m11 = axis->x*xC+c;
	result->m12 = xyC-zs;
	result->m13 = zxC+ys;
	result->m21 = xyC+zs;
	result->m22 = axis->y*yC+c;
	result->m23 = yzC-xs;
	result->m31 = zxC-ys;
	result->m32 = yzC+xs;
	result->m33 = axis->z*zC+c;
	
	result->m41 =
	result->m42 =
	result->m43 =
	result->m14 =
	result->m24 =
	result->m34 = 0.f;
	result->m44 = 1.f;
}

Matrix Matrix::Multiply(const Matrix *const mat1, const Matrix *const mat2)
{
	return Matrix
	(
		mat1->m11*mat2->m11 + mat1->m12*mat2->m21 + mat1->m13*mat2->m31 + mat1->m14*mat2->m41,
		mat1->m11*mat2->m12 + mat1->m12*mat2->m22 + mat1->m13*mat2->m32 + mat1->m14*mat2->m42,
		mat1->m11*mat2->m13 + mat1->m12*mat2->m23 + mat1->m13*mat2->m33 + mat1->m14*mat2->m43,
		mat1->m11*mat2->m14 + mat1->m12*mat2->m24 + mat1->m13*mat2->m34 + mat1->m14*mat2->m44,
	 
		mat1->m21*mat2->m11 + mat1->m22*mat2->m21 + mat1->m23*mat2->m31 + mat1->m24*mat2->m41,
		mat1->m21*mat2->m12 + mat1->m22*mat2->m22 + mat1->m23*mat2->m32 + mat1->m24*mat2->m42,
		mat1->m21*mat2->m13 + mat1->m22*mat2->m23 + mat1->m23*mat2->m33 + mat1->m24*mat2->m43,
		mat1->m21*mat2->m14 + mat1->m22*mat2->m24 + mat1->m23*mat2->m34 + mat1->m24*mat2->m44,
	 
		mat1->m31*mat2->m11 + mat1->m32*mat2->m21 + mat1->m33*mat2->m31 + mat1->m34*mat2->m41,
		mat1->m31*mat2->m12 + mat1->m32*mat2->m22 + mat1->m33*mat2->m32 + mat1->m34*mat2->m42,
		mat1->m31*mat2->m13 + mat1->m32*mat2->m23 + mat1->m33*mat2->m33 + mat1->m34*mat2->m43,
		mat1->m31*mat2->m14 + mat1->m32*mat2->m24 + mat1->m33*mat2->m34 + mat1->m34*mat2->m44,
	 
		mat1->m41*mat2->m11 + mat1->m42*mat2->m21 + mat1->m43*mat2->m31 + mat1->m44*mat2->m41,
		mat1->m41*mat2->m12 + mat1->m42*mat2->m22 + mat1->m43*mat2->m32 + mat1->m44*mat2->m42,
		mat1->m41*mat2->m13 + mat1->m42*mat2->m23 + mat1->m43*mat2->m33 + mat1->m44*mat2->m43,
		mat1->m41*mat2->m14 + mat1->m42*mat2->m24 + mat1->m43*mat2->m34 + mat1->m44*mat2->m44
	);
}

void Matrix::Multiply(const Matrix *const mat1, const Matrix *const mat2, Matrix *const result)
{
	result->m11 = mat1->m11*mat2->m11 + mat1->m12*mat2->m21 + mat1->m13*mat2->m31 + mat1->m14*mat2->m41;
	result->m12 = mat1->m11*mat2->m12 + mat1->m12*mat2->m22 + mat1->m13*mat2->m32 + mat1->m14*mat2->m42;
	result->m13 = mat1->m11*mat2->m13 + mat1->m12*mat2->m23 + mat1->m13*mat2->m33 + mat1->m14*mat2->m43;
	result->m14 = mat1->m11*mat2->m14 + mat1->m12*mat2->m24 + mat1->m13*mat2->m34 + mat1->m14*mat2->m44;
	
	result->m21 = mat1->m21*mat2->m11 + mat1->m22*mat2->m21 + mat1->m23*mat2->m31 + mat1->m24*mat2->m41;
	result->m22 = mat1->m21*mat2->m12 + mat1->m22*mat2->m22 + mat1->m23*mat2->m32 + mat1->m24*mat2->m42;
	result->m23 = mat1->m21*mat2->m13 + mat1->m22*mat2->m23 + mat1->m23*mat2->m33 + mat1->m24*mat2->m43;
	result->m24 = mat1->m21*mat2->m14 + mat1->m22*mat2->m24 + mat1->m23*mat2->m34 + mat1->m24*mat2->m44;
	
	result->m31 = mat1->m31*mat2->m11 + mat1->m32*mat2->m21 + mat1->m33*mat2->m31 + mat1->m34*mat2->m41;
	result->m32 = mat1->m31*mat2->m12 + mat1->m32*mat2->m22 + mat1->m33*mat2->m32 + mat1->m34*mat2->m42;
	result->m33 = mat1->m31*mat2->m13 + mat1->m32*mat2->m23 + mat1->m33*mat2->m33 + mat1->m34*mat2->m43;
	result->m34 = mat1->m31*mat2->m14 + mat1->m32*mat2->m24 + mat1->m33*mat2->m34 + mat1->m34*mat2->m44;
	
	result->m41 = mat1->m41*mat2->m11 + mat1->m42*mat2->m21 + mat1->m43*mat2->m31 + mat1->m44*mat2->m41;
	result->m42 = mat1->m41*mat2->m12 + mat1->m42*mat2->m22 + mat1->m43*mat2->m32 + mat1->m44*mat2->m42;
	result->m43 = mat1->m41*mat2->m13 + mat1->m42*mat2->m23 + mat1->m43*mat2->m33 + mat1->m44*mat2->m43;
	result->m44 = mat1->m41*mat2->m14 + mat1->m42*mat2->m24 + mat1->m43*mat2->m34 + mat1->m44*mat2->m44;
}

Vector3  Matrix::Multiply(const Matrix *const mat1, const Vector3 *const v)
{
	return Vector3(
		mat1->m11 * v->x + mat1->m21 * v->y + mat1->m31 * v->z,
		mat1->m12 * v->x + mat1->m22 * v->y + mat1->m32 * v->z,
		mat1->m13 * v->x + mat1->m23 * v->y + mat1->m33 * v->z
		);
}

void     Matrix::Multiply(const Matrix *const mat1, const Vector3 *const v, Vector3 *const result)
{
    result->x = mat1->m11 * v->x + mat1->m21 * v->y + mat1->m31 * v->z;
    result->y = mat1->m12 * v->x + mat1->m22 * v->y + mat1->m32 * v->z;
    result->z = mat1->m13 * v->x + mat1->m23 * v->y + mat1->m33 * v->z;
}

Matrix	Matrix::Transpose(const Matrix *const mat)
{
	return Matrix
	(
		mat->m11, mat->m21, mat->m31, mat->m41,
		mat->m12, mat->m22, mat->m32, mat->m42,
		mat->m13, mat->m23, mat->m33, mat->m43,
		mat->m14, mat->m24, mat->m34, mat->m44
	);
}

void	Matrix::Transpose(const Matrix *const mat, Matrix *const result)
{
	result->m11 = mat->m11; result->m12 = mat->m21; result->m13 = mat->m31; result->m14 = mat->m41;
	result->m21 = mat->m12; result->m22 = mat->m22; result->m23 = mat->m32; result->m24 = mat->m42;
	result->m31 = mat->m13; result->m32 = mat->m23; result->m33 = mat->m33; result->m34 = mat->m43;
	result->m41 = mat->m14; result->m42 = mat->m24; result->m43 = mat->m34; result->m44 = mat->m44;
}

Matrix	Matrix::CreateTranslation(const float xPosition, const float yPosition, const float zPosition)
{
	return Matrix
	(
		1.f, 0.f, 0.f, xPosition,
		0.f, 1.f, 0.f, yPosition,
		0.f, 0.f, 1.f, zPosition,
		0.f, 0.f, 0.f, 1.f
	);
}

void	Matrix::CreateTranslation(const float xPosition, const float yPosition, const float zPosition, Matrix *const result)
{
	result->m11 = 1.f; result->m12 = 0.f; result->m13 = 0.f; result->m14 = xPosition;
	result->m21 = 0.f; result->m22 = 1.f; result->m23 = 0.f; result->m24 = yPosition;
	result->m31 = 0.f; result->m32 = 0.f; result->m33 = 1.f; result->m34 = zPosition;
	result->m41 = 0.f; result->m42 = 0.f; result->m43 = 0.f; result->m44 = 1.f;
}

Matrix	Matrix::CreateTranslation(const Vector3 *const position)
{
	return Matrix
	(
		1.f, 0.f, 0.f, position->x,
		0.f, 1.f, 0.f, position->y,
		0.f, 0.f, 1.f, position->z,
		0.f, 0.f, 0.f, 1.f
	);
}

void	Matrix::CreateTranslation(const Vector3 *const position, Matrix *const result)
{
	result->m11 = 1.f; result->m12 = 0.f; result->m13 = 0.f; result->m14 = position->x;
	result->m21 = 0.f; result->m22 = 1.f; result->m23 = 0.f; result->m24 = position->y;
	result->m31 = 0.f; result->m32 = 0.f; result->m33 = 1.f; result->m34 = position->z;
	result->m41 = 0.f; result->m42 = 0.f; result->m43 = 0.f; result->m44 = 1.f;
}

Matrix	Matrix::CreateScale(const float scale)
{
	return Matrix
	(
		scale,   0.f, 0.f, 0.f,
		0.f, scale,   0.f, 0.f,
		0.f, 0.f, scale,   0.f,
		0.f, 0.f, 0.f, 1.f
	);
}

void	Matrix::CreateScale(const float scale, Matrix *const result)
{
	result->m11 = scale; result->m12 = 0.f; result->m13 = 0.f; result->m14 = 0.f;
	result->m21 = 0.f; result->m22 = scale; result->m23 = 0.f; result->m24 = 0.f;
	result->m31 = 0.f; result->m32 = 0.f; result->m33 = scale; result->m34 = 0.f;
	result->m41 = 0.f; result->m42 = 0.f; result->m43 = 0.f; result->m44 = 1.f;
}

Matrix	Matrix::CreateScale(const float xScale, const float yScale, const float zScale)
{
	return Matrix
	(
		xScale,   0.f, 0.f, 0.f,
		0.f, yScale,   0.f, 0.f,
		0.f, 0.f, zScale,   0.f,
		0.f, 0.f, 0.f, 1.f
	);
}

void	Matrix::CreateScale(const float xScale, const float yScale, const float zScale, Matrix *const result)
{
	result->m11 = xScale; result->m12 = 0.f; result->m13 = 0.f; result->m14 = 0.f;
	result->m21 = 0.f; result->m22 = yScale; result->m23 = 0.f; result->m24 = 0.f;
	result->m31 = 0.f; result->m32 = 0.f; result->m33 = zScale; result->m34 = 0.f;
	result->m41 = 0.f; result->m42 = 0.f; result->m43 = 0.f; result->m44 = 1.f;
}

Matrix	Matrix::CreateScale(const Vector3 *const scales)
{
	return Matrix
	(
		scales->x,   0.f, 0.f, 0.f,
		0.f, scales->y,   0.f, 0.f,
		0.f, 0.f, scales->z,   0.f,
		0.f, 0.f, 0.f, 1.f
	);
}

void	Matrix::CreateScale(const Vector3 *const scales, Matrix *const result)
{
	result->m11 = scales->x; result->m12 = 0.f; result->m13 = 0.f; result->m14 = 0.f;
	result->m21 = 0.f; result->m22 = scales->y; result->m23 = 0.f; result->m24 = 0.f;
	result->m31 = 0.f; result->m32 = 0.f; result->m33 = scales->z; result->m34 = 0.f;
	result->m41 = 0.f; result->m42 = 0.f; result->m43 = 0.f; result->m44 = 1.f;
}

Matrix	Matrix::CreateRotationX(const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	return Matrix
	(
		1.f,	0.f,	0.f,	0.f,
		0.f,	c,		-s,		0.f,
		0.f,	s,		c,		0.f,
		0.f,	0.f,	0.f,	1.f
	);
}

void	Matrix::CreateRotationX(const float angle, Matrix *const result)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	result->m11 = 1.f; result->m12 = 0.f;	result->m13 = 0.f;	result->m14 = 0.f;
	result->m21 = 0.f; result->m22 = c;		result->m23 = -s;	result->m24 = 0.f;
	result->m31 = 0.f; result->m32 = s;		result->m33 = c;	result->m34 = 0.f;
	result->m41 = 0.f; result->m42 = 0.f;	result->m43 = 0.f;	result->m44 = 1.f;
}

Matrix	Matrix::CreateRotationY(const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	return Matrix
	(
		c,		0.f,	s,		0.f,
		0.f,	1.f,	0.f,	0.f,
		-s,		0.f,	c,		0.f,
		0.f,	0.f,	0.f,	1.f
	);
}

void	Matrix::CreateRotationY(const float angle, Matrix *const result)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	result->m11 = c;	result->m12 = 0.f; result->m13 = s;		result->m14 = 0.f;
	result->m21 = 0.f;	result->m22 = 1.f; result->m23 = 0.f;	result->m24 = 0.f;
	result->m31 = -s;	result->m32 = 0.f; result->m33 = c;		result->m34 = 0.f;
	result->m41 = 0.f;	result->m42 = 0.f; result->m43 = 0.f;	result->m44 = 1.f;
}

Matrix	Matrix::CreateRotationZ(const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	return Matrix
	(
		c,		-s,		0.f,	0.f,
		s,		c,		0.f,	0.f,
		0.f,	0.f,	1.f,	0.f,
		0.f,	0.f,	0.f,	1.f
	);
}

void	Matrix::CreateRotationZ(const float angle, Matrix *const result)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	result->m11 = c;	result->m12 = -s;	result->m13 = 0.f;	result->m14 = 0.f;
	result->m21 = s;	result->m22 = c;	result->m23 = 0.f;	result->m24 = 0.f;
	result->m31 = 0.f;	result->m32 = 0.f;	result->m33 = 1.f;	result->m34 = 0.f;
	result->m41 = 0.f;	result->m42 = 0.f;	result->m43 = 0.f;	result->m44 = 1.f;
}

Matrix   Matrix::CreateBias(const float scale, const float offset)
{
    return Matrix
    (
        scale,    0.0f,   0.0f,   offset,
        0.0f,     scale,  0.0f,   offset,
        0.0f,     0.0f,   scale,  offset,
        0.0f,     0.0f,   0.0f,   1.0f
    );
}

void     Matrix::CreateBias(const float scale, const float offset, Matrix *const result)
{
    result->m11 = scale;	result->m12 = 0.f;      result->m13 = 0.f;      result->m14 = offset;
	result->m21 = 0.f;      result->m22 = scale;	result->m23 = 0.f;      result->m24 = offset;
	result->m31 = 0.f;      result->m32 = 0.f;      result->m33 = scale;	result->m34 = offset;
	result->m41 = 0.f;      result->m42 = 0.f;      result->m43 = 0.f;      result->m44 = 1.f;
}

Matrix   Matrix::CreateBias(const Vector3 *const scale, const Vector3 *const offset)
{
    return Matrix
    (
        scale->x,   0.0f,       0.0f,       offset->x,
        0.0f,       scale->y,   0.0f,       offset->y,
        0.0f,       0.0f,       scale->z,   offset->z,
        0.0f,       0.0f,       0.0f,       1.0f
    );
}

void     Matrix::CreateBias(const Vector3 *const scale, const Vector3 *const offset, Matrix *const result)
{
    result->m11 = scale->x;	result->m12 = 0.f;      result->m13 = 0.f;      result->m14 = offset->x;
	result->m21 = 0.f;      result->m22 = scale->y;	result->m23 = 0.f;      result->m24 = offset->y;
	result->m31 = 0.f;      result->m32 = 0.f;      result->m33 = scale->z;	result->m34 = offset->z;
	result->m41 = 0.f;      result->m42 = 0.f;      result->m43 = 0.f;      result->m44 = 1.f;
}

Matrix	Matrix::CreateWorld(const Vector3 *const position, const Vector3 *const forward, const Vector3 *const up)
{
	Matrix ret;
	
	Vector3::Cross(forward, up, &ret.Right());
	ret.m41 = 0.f;
	ret.Up() = *up;
	ret.m42 = 0.f;
	ret.m13 = -forward->x;
	ret.m23 = -forward->y;
	ret.m33 = -forward->z;
	ret.m43 = 0.f;
	ret.Translation() = *position;
	ret.m44 = 1.f;
	
	return ret;
}

void	Matrix::CreateWorld(const Vector3 *const position, const Vector3 *const forward, const Vector3 *const up, Matrix *const result)
{
	Vector3::Cross(forward, up, &result->Right());
	result->m41 = 0.f;
	result->Up() = *up;
	result->m42 = 0.f;
	result->m13 = -forward->x;
	result->m23 = -forward->y;
	result->m33 = -forward->z;
	result->m43 = 0.f;
	result->Translation() = *position;
	result->m44 = 1.f;
}

Matrix	Matrix::CreateWorldN(const Vector3 *const position, const Vector3 *const forward, const Vector3 *const up)
{
	Matrix ret;

	Vector3 vfw = *forward;
	Vector3 vup = *up;

	vfw.Normalize();
	vup.Normalize();

	Vector3::Cross(&vfw, &vup, &ret.Right());
	Vector3::Cross(&ret.Right(), &vfw, &ret.Up());

	ret.m41 = 0.f;
	ret.m42 = 0.f;
	ret.m13 = -vfw.x;
	ret.m23 = -vfw.y;
	ret.m33 = -vfw.z;
	ret.m43 = 0.f;
	ret.Translation() = *position;
	ret.m44 = 1.f;

	return ret;
}


Matrix	Matrix::CreateLookAt(const Vector3 *const eye, const Vector3 *const center, const Vector3 *const _up)
{
	Vector3 right, up, back;
	
	Vector3::Subtract(eye, center, &back);
	back.Normalize();
	Vector3::Cross(_up, &back, &right);
    right.Normalize();
    Vector3::Cross(&back, &right, &up);
	
	return Matrix
	(
		right.x,	right.y,	right.z,	-Vector3::Dot(&right, eye),
		up.x,		up.y,		up.z,		-Vector3::Dot(&up, eye),
		back.x,     back.y,     back.z,     -Vector3::Dot(&back, eye),
		0.f,		0.f,		0.f,		1.f
	);
}

void	Matrix::CreateLookAt(const Vector3 *const eye, const Vector3 *const center, const Vector3 *const _up, Matrix *const result)
{
	Vector3 right, up, back;
	
	Vector3::Subtract(eye, center, &back);
	back.Normalize();
	Vector3::Cross(_up, &back, &right);
    right.Normalize();
    Vector3::Cross(&back, &right, &up);
	
	result->m11 = right.x;		result->m12 = right.y;		result->m13 = right.z;		result->m14 = -Vector3::Dot(&right, eye);
	result->m21 = up.x;         result->m22 = up.y;         result->m23 = up.z;         result->m24 = -Vector3::Dot(&up, eye);
	result->m31 = back.x;       result->m32 = back.y;       result->m33 = back.z;       result->m34 = -Vector3::Dot(&back, eye);
	result->m41 = 0.f;			result->m42 = 0.f;			result->m43 = 0.f;			result->m44 = 1.f;
}

Matrix Matrix::CreateFromQuaternionTranslation(const Quaternion*const quaternion, const Vector3*const pos)
{
	const float Nq = quaternion->x*quaternion->x + quaternion->y*quaternion->y + quaternion->z*quaternion->z + quaternion->w*quaternion->w;
	const float s = Nq > 0.f ? (2.f/Nq) : 0.f;
	
	const float X = quaternion->x*s;  const float Y = quaternion->y*s;  const float Z = quaternion->z*s;
	const float wX = quaternion->w*X; const float wY = quaternion->w*Y; const float wZ = quaternion->w*Z;
	const float xX = quaternion->x*X; const float xY = quaternion->x*Y; const float xZ = quaternion->x*Z;
	const float yY = quaternion->y*Y; const float yZ = quaternion->y*Z; const float zZ = quaternion->z*Z;
	
	return Matrix
	(
		 1.f-yY-zZ,  xY-wZ,		xZ+wY,		pos->x,
		 xY+wZ,		1.f-xX-zZ,	yZ-wX,		pos->y,
		 xZ-wY,      yZ+wX,		1.f-xX-yY,	pos->z,
		 0.f,		0.f,		0.f,		1.f
	);
}

Matrix Matrix::CreateFromQuaternionTranslationScale(const Quaternion*const quaternion, const Vector3*const pos, const Vector3 *const scale)
{
	Matrix ret;
	Matrix::CreateFromQuaternion(quaternion, &ret);
	ret.Scale(scale);
	ret.Translation() = *pos;
	return ret;
}

void Matrix::CreateFromQuaternionTranslation(const Quaternion*const quaternion, const Vector3*const pos, Matrix *const result)
{
	const float Nq = quaternion->x*quaternion->x + quaternion->y*quaternion->y + quaternion->z*quaternion->z + quaternion->w*quaternion->w;
	const float s = Nq > 0.f ? (2.f/Nq) : 0.f;
	
	const float X = quaternion->x*s;  const float Y = quaternion->y*s;  const float Z = quaternion->z*s;
	const float wX = quaternion->w*X; const float wY = quaternion->w*Y; const float wZ = quaternion->w*Z;
	const float xX = quaternion->x*X; const float xY = quaternion->x*Y; const float xZ = quaternion->x*Z;
	const float yY = quaternion->y*Y; const float yZ = quaternion->y*Z; const float zZ = quaternion->z*Z;
	
	result->m11 = 1.f-yY-zZ;	result->m12 = xY-wZ;		result->m13 = xZ+wY;		result->m14 = pos->x;
	result->m21 = xY+wZ;		result->m22 = 1.f-xX-zZ;	result->m23 = yZ-wX;		result->m24 = pos->y;
	result->m31 = xZ-wY;		result->m32 = yZ+wX;		result->m33 = 1.f-xX-yY;	result->m34 = pos->z;
	result->m41 = 0.f;			result->m42 = 0.f;			result->m43 = 0.f;			result->m44 = 1.f;
}

void Matrix::CreateFromQuaternionTranslationScale(const Quaternion*const quaternion, const Vector3*const pos, const Vector3 *const scale, Matrix *const result)
{
	Matrix::CreateFromQuaternion(quaternion, result);
	result->Scale(scale);
	result->Translation() = *pos;
}

Matrix	Matrix::CreatePerspective(const float fieldOfView, const float aspectRatio, const float nearPlaneDistance, const float farPlaneDistance)
{
	const float halfFov = fieldOfView * 0.5f;
	const float cotangent = cosf(halfFov)/sinf(halfFov);
	const float deltaZ = nearPlaneDistance - farPlaneDistance;
	
	return Matrix
	(
		cotangent/aspectRatio,	0.f,		0.f,											0.f,
		0.f,					cotangent,	0.f,											0.f,
		0.f,					0.f,		(nearPlaneDistance+farPlaneDistance)/deltaZ,	2.f*farPlaneDistance*nearPlaneDistance/deltaZ,
		0.f,					0.f,		-1.f,											0.f
	);
}

void	Matrix::CreatePerspective(const float fieldOfView, const float aspectRatio, const float nearPlaneDistance, const float farPlaneDistance, Matrix *const result)
{
	const float halfFov = fieldOfView * 0.5f;
	const float cotangent = cosf(halfFov)/sinf(halfFov);
	const float deltaZ = nearPlaneDistance - farPlaneDistance;
	
	result->m11 = cotangent/aspectRatio;	result->m12 = 0.f;			result->m13 = 0.f;											result->m14 = 0.f;
	result->m21 = 0.f;						result->m22 = cotangent;	result->m23 = 0.f;											result->m24 = 0.f;
	result->m31 = 0.f;						result->m32 = 0.f;			result->m33 = (nearPlaneDistance+farPlaneDistance)/deltaZ;	result->m34 = 2.f*farPlaneDistance*nearPlaneDistance/deltaZ;
	result->m41 = 0.f;						result->m42 = 0.f;			result->m43 = -1.f;											result->m44 = 0.f;
}

Matrix	Matrix::CreatePerspectiveInfinite(const float fieldOfView, const float aspectRatio)
{
	const float halfFov = fieldOfView * 0.5f;
	const float cotangent = cosf(halfFov)/sinf(halfFov);
	
	return Matrix
	(
		cotangent/aspectRatio,	0.f,		0.f,	0.f,
		0.f,					cotangent,	0.f,	0.f,
		0.f,					0.f,		-1.f,	-2.f,
		0.f,					0.f,		-1.f,	0.f
	);
}

void	Matrix::CreatePerspectiveInfinite(const float fieldOfView, const float aspectRatio, Matrix *const result)
{
	const float halfFov = fieldOfView * 0.5f;
	const float cotangent = cosf(halfFov)/sinf(halfFov);
	
	result->m11 = cotangent/aspectRatio;	result->m12 = 0.f;			result->m13 = 0.f;	result->m14 = 0.f;
	result->m21 = 0.f;						result->m22 = cotangent;	result->m23 = 0.f;	result->m24 = 0.f;
	result->m31 = 0.f;						result->m32 = 0.f;			result->m33 = -1.f;	result->m34 = -2.f;
	result->m41 = 0.f;						result->m42 = 0.f;			result->m43 = -1.f;	result->m44 = 0.f;
}

Matrix	Matrix::CreatePerspectiveObliqueClipPlane(const Matrix *const projection, const Matrix *const view, const Plane *const clipPlaneWorldSpace)
{
	Matrix tmp;
	Matrix suffix;
	Plane clipPlane;
	
	Matrix::Multiply(projection, view, &suffix);
	Matrix::Invert(&suffix, &tmp);
	tmp.Transpose(); //tmp = projection_view_invT
	
	Plane::Transform(clipPlaneWorldSpace, &tmp, &clipPlane);
	
	const float norm = Math::Abs(clipPlane.normal.z);
	clipPlane.normal.x /= norm;
	clipPlane.normal.y /= norm; 
	clipPlane.normal.z /= norm;
	clipPlane.d /= norm;
	clipPlane.d -= 1.f;
	
	if(clipPlane.normal.z < 0.f)
	{
		clipPlane.normal.x	= -clipPlane.normal.x;
		clipPlane.normal.y	= -clipPlane.normal.y;
		clipPlane.normal.z	= -clipPlane.normal.z;
		clipPlane.d			= -clipPlane.d;
	}
	
	suffix = Matrix::Identity;
	suffix.m31 = clipPlane.normal.x;
	suffix.m32 = clipPlane.normal.y;
	suffix.m33 = clipPlane.normal.z;
	suffix.m34 = clipPlane.d;
	
	Matrix::Multiply(&suffix, projection, &tmp);
	return tmp;
}

void	Matrix::CreatePerspectiveObliqueClipPlane(const Matrix *const projection, const Matrix *const view, const Plane *const clipPlaneWorldSpace, Matrix *const result)
{
	Matrix suffix;
	Plane clipPlane;
	
	Matrix::Multiply(projection, view, &suffix);
	Matrix::Invert(&suffix, result);
	result->Transpose(); //tmp = projection_view_invT
	
	Plane::Transform(clipPlaneWorldSpace, result, &clipPlane);
	
	const float norm = Math::Abs(clipPlane.normal.z);
	clipPlane.normal.x /= norm;
	clipPlane.normal.y /= norm; 
	clipPlane.normal.z /= norm;
	clipPlane.d /= norm;
	clipPlane.d -= 1.f;
	
	if(clipPlane.normal.z < 0.f)
	{
		clipPlane.normal.x	= -clipPlane.normal.x;
		clipPlane.normal.y	= -clipPlane.normal.y;
		clipPlane.normal.z	= -clipPlane.normal.z;
		clipPlane.d			= -clipPlane.d;
	}
	
	suffix = Matrix::Identity;
	suffix.m31 = clipPlane.normal.x;
	suffix.m32 = clipPlane.normal.y;
	suffix.m33 = clipPlane.normal.z;
	suffix.m34 = clipPlane.d;
	
	Matrix::Multiply(&suffix, projection, result);
}

Matrix	Matrix::CreateOrthographic(const float left, const float right, const float bottom, const float top, const float zNearPlane, const float zFarPlane)
{
	const float rml = right - left;
	const float tmb = top - bottom;
	const float fmn = zFarPlane - zNearPlane;
	
	return Matrix
	(
		2.f/rml,	0.f,		0.f,		-(right+left)/rml,
		0.f,		2.f/tmb,	0.f,		-(top+bottom)/tmb,
		0.f,		0.f,		-2.f/fmn,	-(zFarPlane+zNearPlane)/fmn,
		0.f,		0.f,		0.f,		1.f
	);
}

void	Matrix::CreateOrthographic(const float left, const float right, const float bottom, const float top, const float zNearPlane, const float zFarPlane, Matrix *const result)
{
	const float rml = right - left;
	const float tmb = top - bottom;
	const float fmn = zFarPlane - zNearPlane;
	
	result->m11 = 2.f/rml;	result->m12 = 0.f;		result->m13 = 0.f;		result->m14 = -(right+left)/rml;
	result->m21 = 0.f;		result->m22 = 2.f/tmb;	result->m23 = 0.f;		result->m24 = -(top+bottom)/tmb;
	result->m31 = 0.f;		result->m32 = 0.f;		result->m33 = -2.f/fmn;	result->m34 = -(zFarPlane+zNearPlane)/fmn;
	result->m41 = 0.f;		result->m42 = 0.f;		result->m43 = 0.f;		result->m44 = 1.f;
}

Matrix	Matrix::CreateFromDirection(const Vector3 *const direction)
{
	const float cp = Math::Sqrt(1.f - (direction->y*direction->y));					//cos pitch
	const float minus_rnorm = -Math::Rsqrt((direction->x*direction->x) + (direction->z*direction->z));
	const float sy = minus_rnorm * direction->x;					//sin yaw
	const float cy = minus_rnorm * direction->z;					//cos yaw

	return Matrix
	(
		cy,		direction->y*sy,		cp*sy,			0.f,
		0.f,	cp,						-direction->y,	0.f,
		-sy,	direction->y*cy,		cp*cy,			0.f,
		0.f,	0.f,					0.f,			1.f
	);
}

void	Matrix::CreateFromDirection(const Vector3 *const direction, Matrix *const result)
{
	const float cp = Math::Sqrt(1.f - (direction->y*direction->y));					//cos pitch
	const float minus_rnorm = -Math::Rsqrt((direction->x*direction->x) + (direction->z*direction->z));
	const float sy = minus_rnorm * direction->x;					//sin yaw
	const float cy = minus_rnorm * direction->z;					//cos yaw
	

	result->m11 = cy;	result->m12 = direction->y*sy;		result->m13 = cp*sy;			result->m14 = 0.f;
	result->m21 = 0.f;	result->m22 = cp;					result->m23 = -direction->y;	result->m24 = 0.f;
	result->m31 = -sy;	result->m32 = direction->y*cy;		result->m33 = cp*cy;			result->m34 = 0.f;
	result->m41 = 0.f;	result->m42 = 0.f;					result->m43 = 0.f;				result->m44 = 1.f;
}

Matrix	Matrix::CreateFromYawPitchRoll(const float yaw, const float pitch, const float roll)
{
	const float sy = sinf(yaw);
	const float cy = cosf(yaw);
	const float sp = sinf(pitch);
	const float cp = cosf(pitch);
	const float sr = sinf(roll);
	const float cr = cosf(roll);
	const float srsp = sr*sp;
	const float crsy = cr*sy;
	const float crcy = cr*cy;
	
	return Matrix
	(
		crcy+srsp*sy,	-sr*cy+crsy*sp,	cp*sy,	0.f,
		sr*cp,			cr*cp,			-sp,	0.f,
		-crsy+srsp*cy,	sr*sy+crcy*sp,	cp*cy,	0.f,
		0.f,			0.f,			0.f,	1.f
	);
}

void	Matrix::CreateFromYawPitchRoll(const float yaw, const float pitch, const float roll, Matrix *const result)
{
	const float sy = sinf(yaw);
	const float cy = cosf(yaw);
	const float sp = sinf(pitch);
	const float cp = cosf(pitch);
	const float sr = sinf(roll);
	const float cr = cosf(roll);
	const float srsp = sr*sp;
	const float crsy = cr*sy;
	const float crcy = cr*cy;
	
	result->m11 = crcy+srsp*sy; result->m12 = -sr*cy+crsy*sp;	result->m13 = cp*sy;	result->m14 = 0.f;
	result->m21 = sr*cp;		result->m22 = cr*cp;			result->m23 = -sp;		result->m24 = 0.f;
	result->m31 = -crsy+srsp*cy;result->m32 = sr*sy+crcy*sp;	result->m33 = cp*cy;	result->m34 = 0.f;
	result->m41 = 0.f;			result->m42 = 0.f;				result->m43 = 0.f;		result->m44 = 1.f;
}

Matrix	Matrix::CreatePlaneProjection(const Plane *const plane, const Vector3 *const v3, const float v3w)
{
	const float dot = Vector3::Dot(&plane->normal, v3) + plane->d * v3w;
	return Matrix
	(
		dot - v3->x * plane->normal.x,	 - v3->x * plane->normal.y,	  - v3->x * plane->normal.z,	 - v3->x * plane->d,
			- v3->y * plane->normal.x,dot- v3->y * plane->normal.y,	  - v3->y * plane->normal.z,	 - v3->y * plane->d,
			- v3->z * plane->normal.x,	 - v3->z * plane->normal.y,dot- v3->z * plane->normal.z,	 - v3->z * plane->d,
			- v3w	* plane->normal.x,	 - v3w	 * plane->normal.y,	  - v3w	  * plane->normal.z, dot - v3w * plane->d
	);
}

void	Matrix::CreatePlaneProjection(const Plane *const plane, const Vector3 *const v3, const float v3w, Matrix *const result)
{
	const float dot = Vector3::Dot(&plane->normal, v3) + plane->d * v3w;
	result->m11 = dot - v3->x * plane->normal.x;	result->m12 =	  - v3->x * plane->normal.y;	result->m13 =	  - v3->x * plane->normal.z;	result->m14 =	  - v3->x * plane->d;
	result->m21 =	  - v3->y * plane->normal.x;	result->m22 = dot - v3->y * plane->normal.y;	result->m23 =	  - v3->y * plane->normal.z;	result->m24 =	  - v3->y * plane->d;
	result->m31 =	  - v3->z * plane->normal.x;	result->m32 =	  - v3->z * plane->normal.y;	result->m33 = dot - v3->z * plane->normal.z;	result->m34 =	  - v3->z * plane->d;
	result->m41 =	  - v3w	  * plane->normal.x;	result->m42 =	  - v3w	  * plane->normal.y;	result->m43 =	  - v3w	  * plane->normal.z;	result->m44 = dot - v3w	  * plane->d;
}

Matrix	Matrix::Translate(const Matrix *const mat, const float xPosition, const float yPosition, const float zPosition)
{
	return Matrix
	(
		mat->m11, mat->m12, mat->m13, mat->m11*xPosition + mat->m12*yPosition + mat->m13*zPosition + mat->m14,
		mat->m21, mat->m22, mat->m23, mat->m21*xPosition + mat->m22*yPosition + mat->m23*zPosition + mat->m24,
		mat->m31, mat->m32, mat->m33, mat->m31*xPosition + mat->m32*yPosition + mat->m33*zPosition + mat->m34,
		mat->m41, mat->m42, mat->m43, mat->m41*xPosition + mat->m42*yPosition + mat->m43*zPosition + mat->m44
	);
}
void	Matrix::Translate(const Matrix *const mat, const float xPosition, const float yPosition, const float zPosition, Matrix *const result)
{
	result->m11 = mat->m11; result->m12 = mat->m12; result->m13 = mat->m13; result->m14 = mat->m11*xPosition + mat->m12*yPosition + mat->m13*zPosition + mat->m14;
	result->m21 = mat->m21; result->m22 = mat->m22; result->m23 = mat->m23; result->m24 = mat->m21*xPosition + mat->m22*yPosition + mat->m23*zPosition + mat->m24;
	result->m31 = mat->m31; result->m32 = mat->m32; result->m33 = mat->m33; result->m34 = mat->m31*xPosition + mat->m32*yPosition + mat->m33*zPosition + mat->m34;
	result->m41 = mat->m41; result->m42 = mat->m42; result->m43 = mat->m43; result->m44 = mat->m41*xPosition + mat->m42*yPosition + mat->m43*zPosition + mat->m44;
}

Matrix	Matrix::Translate(const Matrix *const mat, const Vector3 *const position)
{
	return Matrix
	(
		mat->m11, mat->m12, mat->m13, mat->m11*position->x + mat->m12*position->y + mat->m13*position->z + mat->m14,
		mat->m21, mat->m22, mat->m23, mat->m21*position->x + mat->m22*position->y + mat->m23*position->z + mat->m24,
		mat->m31, mat->m32, mat->m33, mat->m31*position->x + mat->m32*position->y + mat->m33*position->z + mat->m34,
		mat->m41, mat->m42, mat->m43, mat->m41*position->x + mat->m42*position->y + mat->m43*position->z + mat->m44
	);
}

void	Matrix::Translate(const Matrix *const mat, const Vector3 *const position, Matrix *const result)
{
	result->m11 = mat->m11; result->m12 = mat->m12; result->m13 = mat->m13; result->m14 = mat->m11*position->x + mat->m12*position->y + mat->m13*position->z + mat->m14;
	result->m21 = mat->m21; result->m22 = mat->m22; result->m23 = mat->m23; result->m24 = mat->m21*position->x + mat->m22*position->y + mat->m23*position->z + mat->m24;
	result->m31 = mat->m31; result->m32 = mat->m32; result->m33 = mat->m33; result->m34 = mat->m31*position->x + mat->m32*position->y + mat->m33*position->z + mat->m34;
	result->m41 = mat->m41; result->m42 = mat->m42; result->m43 = mat->m43; result->m44 = mat->m41*position->x + mat->m42*position->y + mat->m43*position->z + mat->m44;
}

Matrix	Matrix::RotateX(const Matrix *const mat, const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	return Matrix
	(
		mat->m11, mat->m12*c + mat->m13*s, -mat->m12*s + mat->m13*c, mat->m14,
		mat->m21, mat->m22*c + mat->m23*s, -mat->m22*s + mat->m23*c, mat->m24,
		mat->m31, mat->m32*c + mat->m33*s, -mat->m32*s + mat->m33*c, mat->m34,
		mat->m41, mat->m42*c + mat->m43*s, -mat->m42*s + mat->m43*c, mat->m44
	);
}

void	Matrix::RotateX(const Matrix *const mat, const float angle, Matrix *const result)
{
	const float s = sinf(angle);
	const float c = cosf(angle);

	result->m11 = mat->m11; result->m12 = mat->m12*c + mat->m13*s; result->m13 = -mat->m12*s + mat->m13*c; result->m14 = mat->m14;
	result->m21 = mat->m21; result->m22 = mat->m22*c + mat->m23*s; result->m23 = -mat->m22*s + mat->m23*c; result->m24 = mat->m24;
	result->m31 = mat->m31; result->m32 = mat->m32*c + mat->m33*s; result->m33 = -mat->m32*s + mat->m33*c; result->m34 = mat->m34;
	result->m41 = mat->m41; result->m42 = mat->m42*c + mat->m43*s; result->m43 = -mat->m42*s + mat->m43*c; result->m44 = mat->m44;
}

Matrix	Matrix::RotateY(const Matrix *const mat, const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	return Matrix
	(
		mat->m11*c - mat->m13*s, mat->m12, mat->m11*s + mat->m13*c, mat->m14,
		mat->m21*c - mat->m23*s, mat->m22, mat->m21*s + mat->m23*c, mat->m24,
		mat->m31*c - mat->m33*s, mat->m32, mat->m31*s + mat->m33*c, mat->m34,
		mat->m41*c - mat->m43*s, mat->m42, mat->m41*s + mat->m43*c, mat->m44
	);
}

void	Matrix::RotateY(const Matrix *const mat, const float angle, Matrix *const result)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	result->m11 = mat->m11*c - mat->m13*s; result->m12 = mat->m12; result->m13 = mat->m11*s + mat->m13*c; result->m14 = mat->m14;
	result->m21 = mat->m21*c - mat->m23*s; result->m22 = mat->m22; result->m23 = mat->m21*s + mat->m23*c; result->m24 = mat->m24;
	result->m31 = mat->m31*c - mat->m33*s; result->m32 = mat->m32; result->m33 = mat->m31*s + mat->m33*c; result->m34 = mat->m34;
	result->m41 = mat->m41*c - mat->m43*s; result->m42 = mat->m42; result->m43 = mat->m41*s + mat->m43*c; result->m44 = mat->m44;
}

Matrix	Matrix::RotateZ(const Matrix *const mat, const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	return Matrix
	(
		mat->m11*c + mat->m12*s, -mat->m11*s + mat->m12*c, mat->m13, mat->m14,
		mat->m21*c + mat->m22*s, -mat->m21*s + mat->m22*c, mat->m23, mat->m24,
		mat->m31*c + mat->m32*s, -mat->m31*s + mat->m32*c, mat->m33, mat->m34,
		mat->m41*c + mat->m42*s, -mat->m41*s + mat->m42*c, mat->m43, mat->m44
	);
}

void	Matrix::RotateZ(const Matrix *const mat, const float angle, Matrix *const result)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	
	result->m11 = mat->m11*c + mat->m12*s; result->m12 = -mat->m11*s + mat->m12*c; result->m13 = mat->m13; result->m14 = mat->m14;
	result->m21 = mat->m21*c + mat->m22*s; result->m22 = -mat->m21*s + mat->m22*c; result->m23 = mat->m23; result->m24 = mat->m24;
	result->m31 = mat->m31*c + mat->m32*s; result->m32 = -mat->m31*s + mat->m32*c; result->m33 = mat->m33; result->m34 = mat->m34;
	result->m41 = mat->m41*c + mat->m42*s; result->m42 = -mat->m41*s + mat->m42*c; result->m43 = mat->m43; result->m44 = mat->m44;
}

Matrix	Matrix::Scale(const Matrix *const mat, const float scale)
{
	return Matrix
	(
		mat->m11*scale, mat->m12*scale, mat->m13*scale, mat->m14,
		mat->m21*scale, mat->m22*scale, mat->m23*scale, mat->m24,
		mat->m31*scale, mat->m32*scale, mat->m33*scale, mat->m34,
		mat->m41*scale, mat->m42*scale, mat->m43*scale, mat->m44
	);
}

void	Matrix::Scale(const Matrix *const mat, const float scale, Matrix *const result)
{
	result->m11 = mat->m11*scale; result->m12 = mat->m12*scale; result->m13 = mat->m13*scale; result->m14 = mat->m14;
	result->m21 = mat->m21*scale; result->m22 = mat->m22*scale; result->m23 = mat->m23*scale; result->m24 = mat->m24;
	result->m31 = mat->m31*scale; result->m32 = mat->m32*scale; result->m33 = mat->m33*scale; result->m34 = mat->m34;
	result->m41 = mat->m41*scale; result->m42 = mat->m42*scale; result->m43 = mat->m43*scale; result->m44 = mat->m44;
}

Matrix	Matrix::Scale(const Matrix *const mat, const float xScale, const float yScale, const float zScale)
{
	return Matrix
	(
		mat->m11*xScale, mat->m12*yScale, mat->m13*zScale, mat->m14,
		mat->m21*xScale, mat->m22*yScale, mat->m23*zScale, mat->m24,
		mat->m31*xScale, mat->m32*yScale, mat->m33*zScale, mat->m34,
		mat->m41*xScale, mat->m42*yScale, mat->m43*zScale, mat->m44
	);
}

void	Matrix::Scale(const Matrix *const mat, const float xScale, const float yScale, const float zScale, Matrix *const result)
{
	result->m11 = mat->m11*xScale; result->m12 = mat->m12*yScale; result->m13 = mat->m13*zScale; result->m14 = mat->m14;
	result->m21 = mat->m21*xScale; result->m22 = mat->m22*yScale; result->m23 = mat->m23*zScale; result->m24 = mat->m24;
	result->m31 = mat->m31*xScale; result->m32 = mat->m32*yScale; result->m33 = mat->m33*zScale; result->m34 = mat->m34;
	result->m41 = mat->m41*xScale; result->m42 = mat->m42*yScale; result->m43 = mat->m43*zScale; result->m44 = mat->m44;
}

Matrix	Matrix::Scale(const Matrix *const mat, const Vector3 *const scales)
{
	return Matrix
	(
		mat->m11*scales->x, mat->m12*scales->y, mat->m13*scales->z, mat->m14,
		mat->m21*scales->x, mat->m22*scales->y, mat->m23*scales->z, mat->m24,
		mat->m31*scales->x, mat->m32*scales->y, mat->m33*scales->z, mat->m34,
		mat->m41*scales->x, mat->m42*scales->y, mat->m43*scales->z, mat->m44
	);
}

void	Matrix::Scale(const Matrix *const mat, const Vector3 *const scales, Matrix *const result)
{
	result->m11 = mat->m11*scales->x; result->m12 = mat->m12*scales->y; result->m13 = mat->m13*scales->z; result->m14 = mat->m14;
	result->m21 = mat->m21*scales->x; result->m22 = mat->m22*scales->y; result->m23 = mat->m23*scales->z; result->m24 = mat->m24;
	result->m31 = mat->m31*scales->x; result->m32 = mat->m32*scales->y; result->m33 = mat->m33*scales->z; result->m34 = mat->m34;
	result->m41 = mat->m41*scales->x; result->m42 = mat->m42*scales->y; result->m43 = mat->m43*scales->z; result->m44 = mat->m44;
}

Matrix & Matrix::Translate(const float xPosition, const float yPosition, const float zPosition)
{
	m14 += m11*xPosition + m12*yPosition + m13*zPosition;
	m24 += m21*xPosition + m22*yPosition + m23*zPosition;
	m34 += m31*xPosition + m32*yPosition + m33*zPosition;
	m44 += m41*xPosition + m42*yPosition + m43*zPosition;
	return (*this);
}

Matrix & Matrix::Translate(const Vector3 *const position)
{
	m14 += m11*position->x + m12*position->y + m13*position->z;
	m24 += m21*position->x + m22*position->y + m23*position->z;
	m34 += m31*position->x + m32*position->y + m33*position->z;
	m44 += m41*position->x + m42*position->y + m43*position->z;
	return (*this);
}

Matrix & Matrix::Scale(const float xScale, const float yScale, const float zScale)
{
	m11*=xScale; m12*=yScale; m13*=zScale;
	m21*=xScale; m22*=yScale; m23*=zScale;
	m31*=xScale; m32*=yScale; m33*=zScale;
	m41*=xScale; m42*=yScale; m43*=zScale;
	return (*this);
}

Matrix & Matrix::Scale(const float scale)
{
	m11*=scale; m12*=scale; m13*=scale;
	m21*=scale; m22*=scale; m23*=scale;
	m31*=scale; m32*=scale; m33*=scale;
	m41*=scale; m42*=scale; m43*=scale;
	return (*this);
}

Matrix & Matrix::Scale(const Vector3 *const scales)
{
	m11*=scales->x; m12*=scales->y; m13*=scales->z;
	m21*=scales->x; m22*=scales->y; m23*=scales->z;
	m31*=scales->x; m32*=scales->y; m33*=scales->z;
	m41*=scales->x; m42*=scales->y; m43*=scales->z;
	return (*this);
}

Matrix & Matrix::RotateX(const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	float p1, p2;
	
	p1 = m12; p2 = m13; m12 = p1*c + p2*s; m13 = -p1*s + p2*c;
	p1 = m22; p2 = m23; m22 = p1*c + p2*s; m23 = -p1*s + p2*c;
	p1 = m32; p2 = m33; m32 = p1*c + p2*s; m33 = -p1*s + p2*c;
	p1 = m42; p2 = m43; m42 = p1*c + p2*s; m43 = -p1*s + p2*c;
	return (*this);
}

Matrix & Matrix::RotateY(const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	float p1, p2;
	
	p1 = m11; p2 = m13; m11 = p1*c - p2*s; m13 = p1*s + p2*c;
	p1 = m21; p2 = m23; m21 = p1*c - p2*s; m23 = p1*s + p2*c;
	p1 = m31; p2 = m33; m31 = p1*c - p2*s; m33 = p1*s + p2*c;
	p1 = m41; p2 = m43; m41 = p1*c - p2*s; m43 = p1*s + p2*c;
	return (*this);
}

Matrix & Matrix::RotateZ(const float angle)
{
	const float s = sinf(angle);
	const float c = cosf(angle);
	float p1, p2;
	
	p1 = m11; p2 = m12; m11 = p1*c + p2*s; m12 = -p1*s + p2*c;
	p1 = m21; p2 = m22; m21 = p1*c + p2*s; m22 = -p1*s + p2*c;
	p1 = m31; p2 = m32; m31 = p1*c + p2*s; m32 = -p1*s + p2*c;
	p1 = m41; p2 = m42; m41 = p1*c + p2*s; m42 = -p1*s + p2*c;
	return (*this);
}

Matrix & Matrix::Rotate(const float axisX, const float axisY, const float axisZ, const float angle)
{
	const float c = cosf(angle);	const float s = sinf(angle);	const float C = 1.f - c;
	const float xs = axisX*s;		const float ys = axisY*s;		const float zs = axisZ*s;
	const float xC = axisX*C;		const float yC = axisY*C;		const float zC = axisZ*C;
	const float xyC = axisX*yC;	const float yzC = axisY*zC;	const float zxC = axisZ*xC;
	const float rm11 = axisX*xC+c;
	const float rm12 = xyC-zs;
	const float rm13 = zxC+ys;
	const float rm21 = xyC+zs;
	const float rm22 = axisY*yC+c;
	const float rm23 = yzC-xs;
	const float rm31 = zxC-ys;
	const float rm32 = yzC+xs;
	const float rm33 = axisZ*zC+c;
	float p1, p2, p3;
	
	p1 = m11; p2 = m12; p3 = m13;
	m11 = p1*rm11 + p2*rm21 + p3*rm31;
	m12 = p1*rm12 + p2*rm22 + p3*rm32;
	m13 = p1*rm13 + p2*rm23 + p3*rm33;
	
	p1 = m21; p2 = m22; p3 = m23;
	m21 = p1*rm11 + p2*rm21 + p3*rm31;
	m22 = p1*rm12 + p2*rm22 + p3*rm32;
	m23 = p1*rm13 + p2*rm23 + p3*rm33;
	
	p1 = m31; p2 = m32; p3 = m33;
	m31 = p1*rm11 + p2*rm21 + p3*rm31;
	m32 = p1*rm12 + p2*rm22 + p3*rm32;
	m33 = p1*rm13 + p2*rm23 + p3*rm33;
	
	p1 = m41; p2 = m42; p3 = m43;
	m41 = p1*rm11 + p2*rm21 + p3*rm31;
	m42 = p1*rm12 + p2*rm22 + p3*rm32;
	m43 = p1*rm13 + p2*rm23 + p3*rm33;
	
	return (*this);
}

Matrix & Matrix::Rotate(const Vector3 *const axis, const float angle)
{
	const float c = cosf(angle);	const float s = sinf(angle);	const float C = 1.f - c;
	const float xs = axis->x*s;		const float ys = axis->y*s;		const float zs = axis->z*s;
	const float xC = axis->x*C;		const float yC = axis->y*C;		const float zC = axis->z*C;
	const float xyC = axis->x*yC;	const float yzC = axis->y*zC;	const float zxC = axis->z*xC;
	const float rm11 = axis->x*xC+c;
	const float rm12 = xyC-zs;
	const float rm13 = zxC+ys;
	const float rm21 = xyC+zs;
	const float rm22 = axis->y*yC+c;
	const float rm23 = yzC-xs;
	const float rm31 = zxC-ys;
	const float rm32 = yzC+xs;
	const float rm33 = axis->z*zC+c;
	float p1, p2, p3;

	p1 = m11; p2 = m12; p3 = m13;
	m11 = p1*rm11 + p2*rm21 + p3*rm31;
	m12 = p1*rm12 + p2*rm22 + p3*rm32;
	m13 = p1*rm13 + p2*rm23 + p3*rm33;
	
	p1 = m21; p2 = m22; p3 = m23;
	m21 = p1*rm11 + p2*rm21 + p3*rm31;
	m22 = p1*rm12 + p2*rm22 + p3*rm32;
	m23 = p1*rm13 + p2*rm23 + p3*rm33;
	
	p1 = m31; p2 = m32; p3 = m33;
	m31 = p1*rm11 + p2*rm21 + p3*rm31;
	m32 = p1*rm12 + p2*rm22 + p3*rm32;
	m33 = p1*rm13 + p2*rm23 + p3*rm33;
	
	p1 = m41; p2 = m42; p3 = m43;
	m41 = p1*rm11 + p2*rm21 + p3*rm31;
	m42 = p1*rm12 + p2*rm22 + p3*rm32;
	m43 = p1*rm13 + p2*rm23 + p3*rm33;
	
	return (*this);
}

Matrix & Matrix::Transpose()
{
	float p;
	p = m12; m12 = m21; m21 = p;
	p = m13; m13 = m31; m31 = p;
	p = m14; m14 = m41; m41 = p;
	
	p = m23; m23 = m32; m32 = p;
	p = m24; m24 = m42; m42 = p;
	
	p = m34; m34 = m43; m43 = p;
	return (*this);
}

Matrix & Matrix::Invert()
{
	Matrix inv;
	if(Matrix::Invert(this, &inv))
		(*this) = inv;
	return (*this);
}

Matrix & Matrix::Invert3x3()
{
	Matrix inv;
	if(Matrix::Invert3x3(this, &inv))
		(*this) = inv;
	return (*this);
}
