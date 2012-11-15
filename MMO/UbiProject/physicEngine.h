#ifndef _MECHANICS_LIB_
#define _MECHANICS_LIB_
#include <math.h>
#include "CompCore.h"

/**
* \class MechanicsEngine - Aici sunt metodele care fac calcule de 
* fizica
*/
class MechanicsEngine{
public:
  /**
  * \brief returneaza directia unui obiect cu viteza \param dir1, masa \param  m1, cand se ciocneste 
  * cu obiectul de directie \param dir2 si masa \param m2
  */
  static Vector3 getCollisitonDirections(Vector3 &dir1, float m1, Vector3 &dir2, float m2);
};

#endif