#ifndef TVECTOR_H
#define TVECTOR_H

#include <iostream>
#include <math.h>
#include "mathex.h"

using namespace std;

class TRay;

class TVector {

	public:

		enum TStatus { INVALID, DEFAULT, UNIT };

		double _x, _y, _z;

	private:

		

		TStatus _Status;

		TVector(double x, double y, double z, TStatus s) : _x(x), _y(y), _z(z), _Status(s) {}

		ostream &write(ostream &out) const;

		istream &read(istream &in);

	public:


		TVector() : _x(0.0), _y(0.0), _z(0.0), _Status(INVALID) {}

		TVector(double x, double y, double z) : _x(x), _y(y), _z(z), _Status(DEFAULT) {}


		TVector(const TRay &line1, const TRay &line2);


		double X() const { return _x; }

		double Y() const { return _y; }

		double Z() const { return _z; }

		int isUnit() const { return _Status==UNIT; }

		int isDefault() const { return _Status==DEFAULT; }

		int isValid() const { return _Status!=INVALID; }


		TVector &unit();

		static TVector &unit(const TVector &v, TVector &result) { result = v; return result.unit(); }

		static TVector unit(const TVector &v) { return TVector(v).unit(); }


		TVector &Default();

		static TVector Default(const TVector &v, TVector &result) { result = v; return result.Default(); }

		static TVector Default(const TVector &v) { return TVector(v).Default(); }


		double mag() const { return (isValid() ? (isUnit() ? 1.0 : sqrt(sqr(X()) + sqr(Y()) + sqr(Z()))) : 0.0); }

		double magSqr() const { return (isValid() ? (isUnit() ? 1.0 : sqr(X()) + sqr(Y()) + sqr(Z())) : 0.0); }


		double dot(const TVector &v) const { return ((isValid() && v.isValid()) ? (X()*v.X() + Y()*v.Y() + Z()*v.Z()) : 0.0); }

		static double dot(const TVector &v1, const TVector &v2) { return v1.dot(v2); }


		double dist(const TVector &v) const { return (*this-v).mag(); }

		double distSqr(const TVector &v) const { return (*this-v).magSqr(); }


		static TVector &add(const TVector &v1, const TVector &v2, TVector &result);

		static TVector &subtract(const TVector &v1, const TVector &v2, TVector &result);

		static TVector &cross(const TVector &v1, const TVector &v2, TVector &result);

		static TVector &invert(const TVector &v1, TVector &result);

		static TVector &multiply(const TVector &v1, const double &scale, TVector &result);


		TVector operator-() const { return invert(*this, TVector()); }

		TVector &operator+=(const TVector &v) { return add(*this, v, *this); }

		TVector &operator-=(const TVector &v) { return subtract(*this, v, *this); }

		TVector &operator*=(const TVector &v) { TVector tv(*this); return cross(tv, v, *this); }

		TVector &operator*=(const double &scale) { return multiply(*this, scale, *this); }

		TVector operator+(const TVector &v) const { TVector tv; return add(*this, v, tv); }

		TVector operator-(const TVector &v) const { TVector tv; return subtract(*this, v, tv); }

		TVector operator*(const TVector &v) const { TVector tv; return cross(*this, v, tv); }

		TVector operator*(const double &scale) const { TVector tv; return multiply(*this, scale, tv); }

		friend ostream &operator<<(ostream &out, const TVector &o) { return o.write(out); }

		friend istream &operator>>(istream &in, TVector &o) { return o.read(in); }

};

#endif