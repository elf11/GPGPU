#include "Matrix.h"
#include "Vector2.h"
#include "BoundingFrustum.h"
#include "BoundingBox.h"
#include "BoundingSphere.h"
#include <math.h>

float BoundingFrustum::Project2DZDist(const Vector3 *const p, Vector2 *const pout) const
{
	pout->x = ProjectionViewMat.m11 * p->x + ProjectionViewMat.m12 * p->y + ProjectionViewMat.m13 * p->z + ProjectionViewMat.m14;
	pout->y =-ProjectionViewMat.m21 * p->x - ProjectionViewMat.m22 * p->y - ProjectionViewMat.m23 * p->z - ProjectionViewMat.m24;
    const float w = ProjectionViewMat.m41 * p->x + ProjectionViewMat.m42 * p->y + ProjectionViewMat.m43 * p->z + ProjectionViewMat.m44;
    
	if (w != 0.f)
		Vector2::Divide(pout, w, pout);
    
	return ViewMat.m31 * p->x + ViewMat.m32 * p->y + ViewMat.m33 * p->z + ViewMat.m34;// + m_zNear;
}

float BoundingFrustum::Project2DZDist(const Vector3 *const p, Vector3 *const pout) const
{
	
	pout->x = ProjectionViewMat.m11 * p->x + ProjectionViewMat.m12 * p->y + ProjectionViewMat.m13 * p->z + ProjectionViewMat.m14;
	pout->y =-ProjectionViewMat.m21 * p->x - ProjectionViewMat.m22 * p->y - ProjectionViewMat.m23 * p->z - ProjectionViewMat.m24;
    pout->z =-ProjectionViewMat.m31 * p->x - ProjectionViewMat.m32 * p->y - ProjectionViewMat.m33 * p->z - ProjectionViewMat.m34;
    const float w = ProjectionViewMat.m41 * p->x + ProjectionViewMat.m42 * p->y + ProjectionViewMat.m43 * p->z + ProjectionViewMat.m44;

	if (w != 0.f)
		Vector3::Divide(pout, w, pout);

	return ViewMat.m31 * p->x + ViewMat.m32 * p->y + ViewMat.m33 * p->z + ViewMat.m34;// + m_zNear;
}

void BoundingFrustum::Update(const Matrix *const ProjectionMat, const Matrix *const ViewMat)
{
	this->ProjectionMat = *ProjectionMat;
	this->ViewMat = *ViewMat;
	Matrix::Multiply(&this->ProjectionMat, &this->ViewMat, &this->ProjectionViewMat);

	_right.normal.x	= ProjectionViewMat.m41 - ProjectionViewMat.m11;
    _right.normal.y	= ProjectionViewMat.m42 - ProjectionViewMat.m12;
    _right.normal.z	= ProjectionViewMat.m43 - ProjectionViewMat.m13;
    _right.d			= ProjectionViewMat.m44 - ProjectionViewMat.m14;
	_right.Normalize();
	
    _left.normal.x	= ProjectionViewMat.m41 + ProjectionViewMat.m11;
    _left.normal.y	= ProjectionViewMat.m42 + ProjectionViewMat.m12;
	_left.normal.z	= ProjectionViewMat.m43 + ProjectionViewMat.m13;
    _left.d			= ProjectionViewMat.m44 + ProjectionViewMat.m14;
	_left.Normalize();
    
	_bottom.normal.x = ProjectionViewMat.m41 + ProjectionViewMat.m21;
    _bottom.normal.y = ProjectionViewMat.m42 + ProjectionViewMat.m22;
    _bottom.normal.z = ProjectionViewMat.m43 + ProjectionViewMat.m23;
    _bottom.d		= ProjectionViewMat.m44 + ProjectionViewMat.m24;
	_bottom.Normalize();
	
    _top.normal.x	= ProjectionViewMat.m41 - ProjectionViewMat.m21;
    _top.normal.y	= ProjectionViewMat.m42 - ProjectionViewMat.m22;
    _top.normal.z	= ProjectionViewMat.m43 - ProjectionViewMat.m23;
    _top.d			= ProjectionViewMat.m44 - ProjectionViewMat.m24;
	_top.Normalize();
	
    _far.normal.x	= ProjectionViewMat.m41 - ProjectionViewMat.m31;
    _far.normal.y	= ProjectionViewMat.m42 - ProjectionViewMat.m32;
    _far.normal.z	= ProjectionViewMat.m43 - ProjectionViewMat.m33;
    _far.d			= ProjectionViewMat.m44 - ProjectionViewMat.m34;
	_far.Normalize();
	
    _near.normal.x	= ProjectionViewMat.m41 + ProjectionViewMat.m31;
    _near.normal.y	= ProjectionViewMat.m42 + ProjectionViewMat.m32;
    _near.normal.z	= ProjectionViewMat.m43 + ProjectionViewMat.m33;
    _near.d			= ProjectionViewMat.m44 + ProjectionViewMat.m34;
	_near.Normalize();
}

bool BoundingFrustum::Contains(const Vector3 *const point) const
{
	return	_right.DotCoordinate(point)	>= 0.f &&
			_left.DotCoordinate(point)	>= 0.f &&
			_bottom.DotCoordinate(point) >= 0.f &&
			_top.DotCoordinate(point)	>= 0.f &&
			_far.DotCoordinate(point)	>= 0.f &&
			_near.DotCoordinate(point)	>= 0.f;
			
}

bool BoundingFrustum::Contains(const Vector3 *const point, const float radius) const
{
	return	_right.DotCoordinate(point)	>= -radius &&
			_left.DotCoordinate(point)	>= -radius &&
			_bottom.DotCoordinate(point) >= -radius &&
			_top.DotCoordinate(point)	>= -radius &&
			_far.DotCoordinate(point)	>= -radius &&
			_near.DotCoordinate(point)	>= -radius;
}

bool BoundingFrustum::Contains(const BoundingSphere *const sphere) const
{
	return	_right.DotCoordinate(&sphere->center)	>= -sphere->radius &&
			_left.DotCoordinate(&sphere->center)		>= -sphere->radius &&
			_bottom.DotCoordinate(&sphere->center)	>= -sphere->radius &&
			_top.DotCoordinate(&sphere->center)		>= -sphere->radius &&
			_far.DotCoordinate(&sphere->center)		>= -sphere->radius &&
			_near.DotCoordinate(&sphere->center)		>= -sphere->radius;
}

bool BoundingFrustum::Contains(const BoundingBox *	const box) const
{	
	return	(_right.DotCoordinate(&box->center)	+ fabsf(_right.normal.x)*box->offset.x	+ fabsf(_right.normal.y)*box->offset.y	+ fabsf(_right.normal.z)*box->offset.z)	>= 0.f &&
			(_left.DotCoordinate(&box->center)	+ fabsf(_left.normal.x)*box->offset.x	+ fabsf(_left.normal.y)*box->offset.y	+ fabsf(_left.normal.z)*box->offset.z)	>= 0.f &&
			(_bottom.DotCoordinate(&box->center)	+ fabsf(_bottom.normal.x)*box->offset.x	+ fabsf(_bottom.normal.y)*box->offset.y	+ fabsf(_bottom.normal.z)*box->offset.z) >= 0.f &&
			(_top.DotCoordinate(&box->center)	+ fabsf(_top.normal.x)*box->offset.x		+ fabsf(_top.normal.y)*box->offset.y		+ fabsf(_top.normal.z)*box->offset.z)	>= 0.f &&
			(_far.DotCoordinate(&box->center)	+ fabsf(_far.normal.x)*box->offset.x		+ fabsf(_far.normal.y)*box->offset.y		+ fabsf(_far.normal.z)*box->offset.z)	>= 0.f &&
			(_near.DotCoordinate(&box->center)	+ fabsf(_near.normal.x)*box->offset.x	+ fabsf(_near.normal.y)*box->offset.y	+ fabsf(_near.normal.z)*box->offset.z)	>= 0.f;
}

bool BoundingFrustum::Contains(const BoundingBox *	const box, const Matrix *const boxTransform) const
{
	return true;
}

