#pragma once

#include "Plane.h"

class BoundingBox;
class BoundingSphere;
class Matrix;

class BoundingFrustum
{
private:
	Plane	_right;
	Plane	_left;
	Plane	_bottom;
	Plane	_top;
	Plane	_far;
	Plane	_near;
	
	Matrix	ProjectionMat;
	Matrix	ViewMat;
	Matrix	ProjectionViewMat;
	
public:
	
	inline BoundingFrustum()	{ Update(&Matrix::Identity, &Matrix::Identity); }

	~BoundingFrustum()	{}
	
	inline const float * Ptr() const { return &_left.normal.x; }
	
	inline const Plane & Right()			const { return _right; }
	inline const Plane & Left()				const { return _left; }
	inline const Plane & Bottom()			const { return _bottom; }
	inline const Plane & Top()				const { return _top; }
	inline const Plane & Far()				const { return _far; }
	inline const Plane & Near()				const { return _near; }
	
	inline const Matrix & Projection()		const { return ProjectionMat; }
	inline const Matrix & View()			const { return ViewMat; }
	inline const Matrix & ProjectionView()	const { return ProjectionViewMat; }
	
	void Update(const Matrix *const ProjectionMat, const Matrix *const ViewMat);
	
	bool  Contains(const Vector3 *const point) const;
	bool  Contains(const Vector3 *const point, const float radius) const;
	bool  Contains(const BoundingSphere *const sphere) const;
	bool  Contains(const BoundingBox *	const box) const;
	bool  Contains(const BoundingBox *	const box, const Matrix *const boxTransform) const;
	
	/// <summary>
	/// Returns the projection of a point in screen space and the world=z distance in camera space.
	/// </summary>
	/// <param name="p">The point in world space</param>
	/// <param name="pout">pout.XY are the screen projection and pout.Z is the distance on z in camera space</param>
	float Project2DZDist(const Vector3 *const point, Vector2 *const pout) const;
    float Project2DZDist(const Vector3 *const point, Vector3 *const pout) const;
};

