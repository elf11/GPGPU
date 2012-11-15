#ifndef _LIGHTS_LIB_
#define _LIGHTS_LIB_
#define WIN32_LEAN_AND_MEAN
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<GL/glew.h>
#include<GL/GL.H>
#include<GL/GLU.H>
#include<GL/GLAUX.H>
#include<GL/glut.h>

/**
* \namespace SceneSettings - stabileste luminile
* \brief Deocamdata exista doar lumini statice
*/
namespace SceneSettings{

  /**
  * \brief Initializeaza luminile
  */
  void initlights();
}
#endif