#pragma once

#include "Vector3.h"

class BoundingBox
{
public:
	Vector3 center;
	Vector3 offset;
	
	inline BoundingBox() {}
	inline BoundingBox(const Vector3 *const Center, const Vector3 *const Offset) : center(*Center), offset(*Offset) {}
	~BoundingBox()	{}
	
	inline float * Ptr() { return &center.x; }
	inline const float * Ptr() const { return &center.x; }
    
    bool Contains(const Vector3 *const pos) const;
    bool Contains(const Vector3 *const pos, const float radius) const;
};

class BoundingBoxMM
{
public:
	Vector3 min, max;

public:
	void Reset();
	void Expand(const Vector3& pos, float radius);

	void ConvertTo(BoundingBox* b) const;
	void ConvertFrom(const BoundingBox& b);
};
