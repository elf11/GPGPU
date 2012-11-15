#include "TGAImage.h"

TGAImage::TGAImage(char* filename)
{
  loadFile(filename);
  BGRtoRGB();
}

/**
 * \attention Do not delete \param data
 */
TGAImage::~TGAImage()
{
}

bool TGAImage::loadFile(char* filename)
{
  FILE *f = fopen(filename, "rb");
  if(!f){
    std::cout<<"Error reading texture"<<std::endl;
    return false;
  }
  fread(&tga_header, 1, sizeof(TargaHeader), f);
  height = tga_header.height;
  width = tga_header.width;
  data = (unsigned char*)malloc(4*width*height*sizeof(unsigned char));
  fread(data, 1, width*height*4*sizeof(unsigned char), f);
  fclose(f);
  return true;
}

void TGAImage::BGRtoRGB()
{
  for(unsigned int i = 0; i < width*height; i++){
    std::swap(data[4*i+2], data[4*i]);
  }
}