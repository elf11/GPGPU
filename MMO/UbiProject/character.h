#ifndef _CHARACTER_LIB_
#define _CHARACTER_LIB_
#include "CompCore.h"

/**
* \class Bone - neimplementata 
* TODO:
* O sa reprezinte un engine pentru oase si joint-uri,
* sau mai bine zis o librarie
*/
class Bone{
public:
  object section;
  int middle, in, out;
  Vector3 localIn, localOut;
  Vector3 globalIn, globalOut;
  Bone **childs;

  Bone();
  ~Bone();
};



#endif