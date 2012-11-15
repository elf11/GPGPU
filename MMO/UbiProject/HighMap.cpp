#include "HighMap.h"

HighMap::HighMap(std::vector<Vector3> *vertices, char* detailsFile)
{
  vertices;
  this->vertices = vertices;
  FILE *f = fopen(detailsFile, "r");
  if(f == NULL){
    std::cout<<"Error reading highmap configuration file"<<std::endl;
    return;
  }
  char line[60];
  fgets(line, 60, f);
  sscanf(line, "lines=%d", &lines);
  fgets(line, 60, f);
  sscanf(line, "columns=%d", &columns);
  fgets(line, 60, f);
  sscanf(line, "lineStep=%f", &lineStep);
  fgets(line, 60, f);
  sscanf(line, "columnStep=%f", &columnStep);
  fclose(f);
}

/**
 * \attention Caution! vertices dont have to be deleted here
 */
HighMap::~HighMap()
{
}

float HighMap::getYPoint(float x, float z)
{
  Point2D point;
  float p;
  point.x = x;
  point.y = z;
  this->getMaxLongitude();
  point.y = floor((float)(point.y + this->getMaxLongitude()/2)/this->getLatitudeStep());
  point.x = floor((float)(point.x + this->getMaxLatitude()/2)/this->getLongitudeStep());
  point.y; point.x;
  p = this->getHeight(point.y, point.x);
  return p;
}
