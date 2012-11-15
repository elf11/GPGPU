#include <iostream>
#include <vector>
#include "CompCore.h"

/**
* \struct TargaHeader - constine elementele unui format TGA
*/
struct TargaHeader{
  unsigned char idLength;
  unsigned char colorMapType;
  unsigned char imageTypeCode;
  unsigned char colorMapSpec[5];
  unsigned short xOrigin;
  unsigned short yOrigin;
  unsigned short width;
  unsigned short height;
  unsigned char bpp;
  unsigned char imageDesc;
};


/**
* \class TGAImage - clasa care citeste o imagine TGA
* si o incarca in memoria GPU-ului
*/
class TGAImage{
  TargaHeader tga_header;
public:
  unsigned char *data;
  unsigned int width, height;
  explicit TGAImage(char* filename);
  ~TGAImage();
  virtual bool loadFile(char* filename);

  /**
  * \brief Modifica octetii pentru a fi incarcati in memoria textura
  */
  void BGRtoRGB();
};