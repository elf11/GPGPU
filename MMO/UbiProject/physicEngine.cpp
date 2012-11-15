#include "physicEngine.h"


Vector3 MechanicsEngine::getCollisitonDirections(Vector3 &dir1, float m1, Vector3 &dir2, float m2)
{
  Vector3 rez;
  float m = m1+m2;
  float sem = 2.0f;
  rez = ((dir1*m1 + dir2*m2)/m) * sem - dir1;
  return rez;
}