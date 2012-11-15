#ifndef TRAY_H
#define TRAY_H

#include <iostream>
#include "mathex.h"
#include "tvector.h"

class TRay
{
	private:
		TVector _P;
		TVector _V;

		ostream &write(ostream &out) const;

		istream &read(istream &in);

	public:
		TRay() {}
		TRay(const TVector &point1, const TVector &point2);
		bool adjacentPoints(const TRay &ray, TVector &point1, TVector &point2) const;
		static TRay &invert(const TRay &r, TRay &result) { result._P = r._P; TVector::invert(r._V, result._V); return result; }
		TRay operator-() const { return invert(*this, TRay()); }
		TVector P() const { return _P; }
		TVector V() const { return _V; }
		int isValid() const { return V().isUnit() && P().isValid(); }
		double dist(const TRay &ray) const;
		double dist(const TVector &point) const;
		friend ostream &operator<<(ostream &out, const TRay &o) { return o.write(out); }
		friend istream &operator>>(istream &in, TRay &o) { return o.read(in); }
};

#endif