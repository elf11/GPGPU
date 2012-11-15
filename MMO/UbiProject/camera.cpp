#include "camera.h"
#include "staticObj.h"
#include <iostream>

extern HighMap *logicMap;
Camera::Camera()
{
  translation.x = translation.y = translation.z = 0.0f;
  rotation.x = rotation.y = rotation.z = 0.0f;
  scale.x = scale.y = scale.z = 0.0f;
}

Camera::~Camera()
{
}

void Camera::setTranslation(Vector3 &point)
{
  translation.x = point.x;
  translation.x = point.y;
  translation.x = point.z;
}

void Camera::setRotation(Vector3 &newAxis)
{
  rotation.x = newAxis.x;
  rotation.x = newAxis.y;
  rotation.x = newAxis.z;
}

void Camera::setScale(Vector3 &newSize)
{
  scale.x = newSize.x;
  scale.x = newSize.y;
  scale.x = newSize.z;
}

void Camera::moveForward(float step)
{
  undo_translation = translation;
  translation.x -= step * sin(rotation.y * PI/180);
  translation.z += step * cos(rotation.y * PI/180);
}

void Camera::moveBackward(float step)
{
  undo_translation = translation;
  translation.x += step * sin(rotation.y * PI/180);
  translation.z -= step * cos(rotation.y * PI/180);
}

void Camera::moveLeft(float step)
{
  undo_translation = translation;
  translation.x += step * sin((rotation.y + 90)*PI/180);
  translation.z -= step * cos((rotation.y + 90)*PI/180);
}

void Camera::moveRight(float step)
{
  undo_translation = translation;
  translation.x -= step * sin((rotation.y + 90)*PI/180);
  translation.z += step * cos((rotation.y + 90)*PI/180);
}

void Camera::rotateLeft(float step)
{
  if(rotation.y < 0){
    rotation.y += 360.0f;
    rotation.y -= step;
  }
  else{
    rotation.y -= step;
  }
}

void Camera::rotateRight(float step)
{
  if(rotation.y > 0){
    rotation.y -= 360.0f;
    rotation.y += step;
  }
  else{
    rotation.y += step;
  }
}

void Camera::reset()
{
  translation.x = translation.y = translation.z = 0.0f;
  rotation.x = rotation.y = rotation.z = 0.0f;
  scale.x = scale.y = scale.z = 0.0f;
}

void Camera::getPosition()
{
  Point2D point;
  point.x = -translation.x;
  point.y = -translation.z;
  point.y = floor((float)(point.y + logicMap->getMaxLongitude()/2)/logicMap->getLatitudeStep());
  point.x = floor((float)(point.x + logicMap->getMaxLatitude()/2)/logicMap->getLongitudeStep());
  translation.y = logicMap->getHeight(point.y, point.x);
}
