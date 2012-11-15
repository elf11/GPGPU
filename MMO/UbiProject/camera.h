#ifndef _CAMERA_LIB_
#define _CAMERA_LIB_
#include<math.h>
#include<GL/glew.h>
#include<GL/GL.H>
#include<GL/GLU.H>
#include<GL/GLAUX.H>
#include<GL/glut.h>
#include "CompCore.h"
#include "HighMap.h"


/**
* class Camera - Clasa care reprezinta camera controlata de utilizator
* Foloseste metoda rotate-translate nu glLookAt, in contextul curent este
* discutabila schimbarea implementarii, dar este foarte probabil ca ar duce
* la un rezultat bun
*/
class Camera{
public:
  Vector3 translation, rotation, scale;

  /**
  * \param undo_translation Pentru cazul in care personajul se loveste
  * de ceva, nu trebuie sa mai avanseze si se intoarce la pozitia dinaintea
  * coliziunii
  */
  Vector3 undo_translation;

  /**
  * \brief Initializarea camerei se face la pozitia (0, 0, 0) cu rotatie
  * (0, 0, 0) si scale(1, 1, 1)
  */
  Camera();

  ~Camera();

  /**
  * \brief Schimba pozitia la pozitia <b>newPoint</b>
  * \param newPoint
  */
  void setTranslation(Vector3 &point);

  /**
  * \brief Schimba vectorul in jurul careia se desfasoara rotatia
  * la vectorul newAxis, ar fi o idee buna sa primeasca ca parametru
  * un quaternion
  */
  void setRotation(Vector3 &newAxis);

  /**
  * \brief Seteaza scalarea la dimensiunile newSize
  * \attetntion Netestata!
  */
  void setScale(Vector3 &newSize);

  /**
  * \brief Muta camera inainte oricare ar fi orientarea acesteia.
  * \param step Distanta pe care se face avansarea
  */
  void moveForward(float step);

  /**
  * \brief Muta camera inapoi oricare ar fi orientarea acesteia.
  */
  void moveBackward(float step);

  /**
  * \brief Muta camera spre stanga oricare ar fi orientarea acesteia.
  */
  void moveLeft(float step);

  /**
  * \brief Muta camera spre dreapta oricare ar fi orientarea acesteia.
  */
  void moveRight(float step);

  /**
  * \brief Roteste camera spre stanga
  */
  void rotateLeft(float step);

  /**
  * \brief Roteste camera spre dreapta
  */
  void rotateRight(float step);

  /**
  * \brief Efectueaza mutarea dupa metoda rotate-translate
  */
  inline void makeMove() const
  {
    glRotatef(rotation.y, 0, 1, 0);
    glTranslatef(translation.x, translation.y, translation.z);
  }

  /**
  * \brief Efecturaza rotatia in jurul axei oy
  */
  inline void makeRotation() const {glRotatef(rotation.y, 0, 1, 0);}

  /**
  * \brief Efectueaza translatia camerei
  */
  inline void makeTranslation() const 
  {glTranslatef(translation.x, -translation.y, translation.z);}

  /**
  * \brief Reseteaza pozitia camerei la cea initiala
  */
  void reset();

  /**
  * \brief Seteaza inaltimea la care se gaseste camera conform highMap-ului
  * Nu functioneaza foarte bine TODO - analizat bug-uri
  */
  virtual void getPosition();
};


#endif