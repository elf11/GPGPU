#include "CRSpline.h"
#include "RMath.h"
#include "stdinc.h"
#include "VertexDeclarations.h"
//#include "Graphics.h"
//#include "Game.h"

CRSpline::CRSpline()
{
	vp.clear();
	delta_t = 0;
	spline_length = 0.f;
	pathVtx = NULL;
}

CRSpline::CRSpline(const CRSpline& s)
{
    for (int i = 0; i < (int)s.vp.size(); i++)
        vp.push_back(s.vp[i]);
    delta_t = s.delta_t;
	spline_length = s.spline_length;
	pathVtx = NULL;
}

CRSpline::~CRSpline()
{
	SAFE_DEL_ARRAY(pathVtx);

}

// Solve the Catmull-Rom parametric equation for a given time(t) and vector quadruple (p1,p2,p3,p4)
Vector3 CRSpline::Eq(float t, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& p4)
{
    float t2 = t * t;
    float t3 = t2 * t;

    float b1 = .5 * (  -t3 + 2*t2 - t);
    float b2 = .5 * ( 3*t3 - 5*t2 + 2);
    float b3 = .5 * (-3*t3 + 4*t2 + t);
    float b4 = .5 * (   t3 -   t2    );

    return (p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4); 
}

void CRSpline::AddSplinePoint(Vector3 v)
{
    vp.push_back(v);
    delta_t = (float)1 / (float)vp.size();
}

Vector3 CRSpline::GetInterpolatedSplinePoint(float t)
{
    // Find out in which interval we are on the spline
    int p = (int)(t / delta_t);
    // Compute local control point indices
#define BOUNDS(pp) { if (pp < 0) pp = 0; else if (pp >= (int)vp.size()-1) pp = vp.size() - 1; }
    int p0 = p - 1;     BOUNDS(p0);
    int p1 = p;         BOUNDS(p1);
    int p2 = p + 1;     BOUNDS(p2);
    int p3 = p + 2;     BOUNDS(p3);

    // Relative (local) time 
	float lt = (t - delta_t*(float)p) / delta_t;
	// Interpolate
    return CRSpline::Eq(lt, vp[p0], vp[p1], vp[p2], vp[p3]);
}

int CRSpline::GetNumPoints()
{
	return vp.size();
}

Vector3& CRSpline::GetNthPoint(int n)
{
	if (n>=vp.size())
		return vp[vp.size()-1];
	return vp[n];
}

float CRSpline::CalculateSplineLength()
{
	spline_length = 0.f;
	int precision = 100;
	for(int i = 0;i<(precision);i++)
	{
		float current = (float)i/((float)precision);
		float next = (float)(i+1)/((float)precision);
        Vector3 tmp1 = GetInterpolatedSplinePoint(current);
        Vector3 tmp2 = GetInterpolatedSplinePoint(next);
		spline_length+= Vector3::Distance(&tmp1,&tmp2);
	}
	return spline_length;
}

void CRSpline::MakeSplineUniform()
{
	CalculateSplineLength();
	
	int numpoints = 50;
	float arclength = 0.f;
	float epsilon = 0.1;
	float step = 0.001;
	float time = 0;
	float segment = 0.f;
	float segment_const = spline_length/((float)numpoints);
	vector<Vector3> tmp;
	tmp.push_back(GetInterpolatedSplinePoint(0));
	while((arclength < (spline_length)))
	{
		float small_segment = 0.f;
		float current = time;
		float next = time + step;
		time+=step;
        Vector3 tmp1 = GetInterpolatedSplinePoint(current);
        Vector3 tmp2 = GetInterpolatedSplinePoint(next);
		small_segment = Vector3::Distance(&tmp1, &tmp2);
		segment +=small_segment;
		arclength +=small_segment;
		if ((segment > segment_const) && (arclength < (spline_length)))
		{
			tmp.push_back(GetInterpolatedSplinePoint(next));
			segment -= segment_const;
		}
		
	}
	tmp.push_back(vp[vp.size()-1]);
	//tmp.push_back(vp[vp.size()-1]);
	//tmp.push_back(vp[vp.size()-1]);
	vp.clear();
	delta_t = 0;
	spline_length = 0.f;
	for(int i = 0;i<tmp.size();i++)
	{
		AddSplinePoint(tmp[i]);
	}
	CalculateSplineLength();

	/*
	arclength = 0.9 * spline_length;
	DEBUG_OUT("spline len %f",spline_length);
	while(arclength<spline_length)
	{
		DEBUG_OUT("spline(%f): %f %f %f",arclength,GetInterpolatedSplinePoint(arclength/spline_length).x,GetInterpolatedSplinePoint(arclength/spline_length).y,GetInterpolatedSplinePoint(arclength/spline_length).z);
		arclength+=30.f;
	}
	*/
}

void CRSpline::GeneratePathDrawInfo(float yoffset)
{
#ifndef _DEDICATED_SERVER
	int const nSegments = MAX_PATH_POINTS;
	pathVtx = new VertexPosColor[MAX_PATH_POINTS];
	for(int i = 0; i < NUM_PATH_IDX -1; i++)
	{
		pathIdx[i] = i;
		pathIdx[i+1] = i+1;

	}
	static float PATH_WIDTH = 30.f;
	for(int i = 0; i < nSegments; i++)
	{
		Vector3 v1= GetInterpolatedSplinePoint((float)i/(float)nSegments);
		v1.y += yoffset;


		pathVtx[i].pos = v1;
		pathVtx[i].color = Color::Lerp(Color::Red,Color::Blue,i/((float)MAX_PATH_POINTS));
		
	}
#endif
}
