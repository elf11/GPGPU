#include "BoundingBox.h"
#include <math.h>
#include "RMath.h"


bool BoundingBox::Contains(const Vector3 *const pos) const
{
    const Vector3 adiff(fabsf(pos->x - center.x),
                        fabsf(pos->y - center.y),
                        fabsf(pos->z - center.z));
    
    return  adiff.x < offset.x &&
            adiff.y < offset.y &&
            adiff.z < offset.z;
}

bool BoundingBox::Contains(const Vector3 *const pos, const float radius) const
{
    const Vector3 adiff(fabsf(pos->x - center.x),
                        fabsf(pos->y - center.y),
                        fabsf(pos->z - center.z));
    
    return  adiff.x < (offset.x + radius) ||
            adiff.y < (offset.y + radius) ||
            adiff.z < (offset.z + radius);
}

void BoundingBoxMM::Reset()
{
	min.Set(Math::MaxFloat, Math::MaxFloat, Math::MaxFloat);
	max.Set(-Math::MaxFloat, -Math::MaxFloat, -Math::MaxFloat);
}

void BoundingBoxMM::Expand(const Vector3& pos, float radius)
{
	if (min.x > pos.x-radius)
		min.x = pos.x-radius;
	if (min.y > pos.y-radius)
		min.y = pos.y-radius;
	if (min.z > pos.z-radius)
		min.z = pos.z-radius;

	if (max.x < pos.x+radius)
		max.x = pos.x+radius;
	if (max.y < pos.y+radius)
		max.y = pos.y+radius;
	if (max.z < pos.z+radius)
		max.z = pos.z+radius;
}

void BoundingBoxMM::ConvertTo(BoundingBox* b) const
{
	b->center = (max+min) * 0.5f;
	b->offset = (max-min) * 0.5f;
}

void BoundingBoxMM::ConvertFrom(const BoundingBox& b)
{
	min = b.center - b.offset;
	max = b.center + b.offset;
}
