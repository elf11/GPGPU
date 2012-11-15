#ifndef _HIGH_MAP_LIB_
#define _HIGH_MAP_LIB_
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<GL/glew.h>
#include<GL/GL.H>
#include<GL/GLU.H>
#include<GL/GLAUX.H>
#include<GL/glut.h>
#include<vector>
#include "CompCore.h"
#include <iostream>

/**
* \class HighMap - clasa pentru highmap
* Implementarea este optima, dar nu este implementata foarte bine
* in sensul ca are bug-uri, un studiu al acestora si apoi rezolvarea lor este necesara
* De asemenea trebuie modificat modelul dupa care este creat high-map-ul din motive pe care
* le expplic in cele ce urmeaza:
* High-map-ul reprezinta de fapt vertecsii modelului terenului, acestia sunt special
* aliniati la distante egale pe ox, oz, distanta specificata in fisier-ul de configurare
* al high-map-ului, de asemenea numarul coloanelor si al liniilor sunt specificate
* Cum mesh-ul este compus numai din triunghiuri - este bine sa fie asa pentru ca daca ar
* fi dreptunghiuri ar putea incalca usor regulile cum ca fetele trebuie sa fie complanare-
* diagonalele dupa implementarea curenta nu sunt aliniate in acelasi sens.
* Prin aceasta metoda inaltimea corespunzatoare este calculata in O(1).
* 
*/
class HighMap{
  std::vector<Vector3> *vertices;
  int lines, columns;
  float lineStep, columnStep;
public:
  HighMap(std::vector<Vector3> *vertices, char* detailsFile);
  ~HighMap();

  /**
  * \return Returneaza inaltimea corespunzatoare lementului de pe linia \param x
  * si coloana \param y
  */
  inline float getHeight(int x, int z) const{
    return (*vertices)[(lines - x)*columns + z].y;
  }

  /**
  * \return Returneaza inaltimea dupa puncte pentru care nu sunt disponibile
  * valori in matrice pentru ca nu sunt numere intregi, aici trebuie facuta interpolare liniara
  */
  float getYPoint(float x, float z);

  /**
  * \brief Returneaza numarul de coloane
  * \attention Numarul de linii nu este numarul de coloane din matrice
  * ci practic latimea hartii
  */
  inline int getMaxLatitude() const {return (columns-1) * columnStep;}

  /**
  * \brief Returneaza numarul de linii
  * \attention Numarul de linii nu este numarul de linii din matrice
  * ci practic adancimea hartii
  */
  inline int getMaxLongitude() const {return (lines-1) * lineStep;}

  /**
  * \brief Returneaza distanta dintr 2 coloane consecutive
  */
  inline float getLongitudeStep() const {return columnStep;}

  /**
  * \brief Returneaza distanta intre 2 linii consecutive
  */
  inline float getLatitudeStep() const {return lineStep;}

  /**
  * \brief Returneaza numarul de linii din matrice
  */
  inline int getLinesCount()const {return lines;}

  /**
  * \brief Returneaza numarul de coloane din matrice
  */
  inline int getColumnsCount() const {return columns;}
};

#endif