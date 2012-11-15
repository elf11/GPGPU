/*
#pragma once


#include <vector>

using namespace std;
class Vector3;
class VertexPosColor;

#define MAX_PATH_POINTS 150
#define NUM_PATH_VTX ((MAX_PATH_POINTS))
#define NUM_PATH_IDX (MAX_PATH_POINTS *2 +1)
class CRSpline
{
public:

    // Constructors and destructor
    CRSpline();
    CRSpline(const CRSpline&);
    ~CRSpline();
	VertexPosColor* pathVtx;
	unsigned short pathIdx[NUM_PATH_IDX];
    // Operations
    void AddSplinePoint(Vector3 v);
	Vector3 GetInterpolatedSplinePoint(float t);   // t = 0...1; 0=vp[0] ... 1=vp[max]
	int GetNumPoints();
	Vector3& GetNthPoint(int n);
	float spline_length;
	float CalculateSplineLength();
	void MakeSplineUniform();
	void Draw();
	void GeneratePathDrawInfo(float yoffset = 100);
    // Static method for computing the Catmull-Rom parametric equation
    // given a time (t) and a vector quadruple (p1,p2,p3,p4).
    static Vector3 Eq(float t, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& p4);

private:
    vector<Vector3> vp;
    float delta_t;
};

*/