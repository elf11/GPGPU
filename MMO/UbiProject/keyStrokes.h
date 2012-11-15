#ifndef _KEY_STROKE_LIB_
#define _KEY_STROKE_LIB_
#include "camera.h"
#include "iostream"
#include "MenuGUI.h"


/**
* \class KeyStrokes - Reprezinta clasa care trateaza input-ul de la tastatura
*/
class KeyStrokes{
public:
  static void keyPressed (unsigned char key, int x, int y);
  static void keyUp(unsigned char key, int x, int y);
  static void checkKey();
  static void Escape_key();
  static void Enter_key();
  static void w_key();
  static void s_key();
};
#endif