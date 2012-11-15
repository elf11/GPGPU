/***************************************************************************
main.cpp
-------------------
begin                : 2012-20-06
email                : yonutix@yahoo.com
***************************************************************************/
/***************************************************************************
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
***************************************************************************/

#include "Client.h"

//C standard
#include<stdio.h>
#include<string.h>
#include <time.h>
//C++ standard
#include<vector>
#include<iostream>
//Math
#include<math.h>
#include <time.h>
//Graphics
#include<GL/glew.h>
#include<GL/GL.H>
#include<GL/GLU.H>
#include<GL/GLAUX.H>
#include<GL/glut.h>
//Sound
//#include <al.h>
//#include <alc.h>
#include "lights.h"
#include "staticObj.h"
#include "keyStrokes.h"
#include "camera.h"
#include "HighMap.h"
#include "MenuGUI.h"

#define NUM_BUFFERS 1
#define NUM_SOURCES 1
#define NUM_ENVIRONMENTS 1

StaticObj *objects;
bool keys[256];
Camera *camera;
HighMap *logicMap;
HANDLE thread[2];
MenuGUI *menu;
MainCharacter * mainCharacter;
//tool chain
int objNr_edit;
Vector3 translate_edit;
Client *client;


void myReshape(GLsizei w , GLsizei h)
{
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(20, (GLfloat)w/(GLfloat)h ,1 , 1200);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glClearColor(0, 0, 0, 1);
}


void sends()
{
  
  mainCharacter->position = camera->translation;
  Vector3 undo_trans;
  undo_trans.x = -camera->undo_translation.x;
  undo_trans.y = mainCharacter->position.y ;
  undo_trans.z = -camera->undo_translation.z-100;
  undo_trans.y = logicMap->getYPoint(-undo_trans.x, -undo_trans.z-100) - 15;
  mainCharacter->position.y 
    = logicMap->getYPoint(-mainCharacter->position.x, -mainCharacter->position.z-100) - 15;
  Vector3 req_poz;
  req_poz.x = -mainCharacter->position.x;
  req_poz.y = mainCharacter->position.y;
  req_poz.z = -mainCharacter->position.z-100;
  int length;
  float *buffer = mainCharacter->wrapBoundingBox(req_poz, length, client->id);
  float *restore_buff = mainCharacter->wrapBoundingBox(undo_trans, length, client->id);
  mainCharacter->render(mainCharacter->frames, buffer, restore_buff);
  client->sendBuffer(buffer, 8);
  char b_c[512];
  if(recv(client->sockfd, b_c, 512, NULL)){
  }
  else{
    std::cout<<"Error on recive"<<std::endl;
  }
  delete[] buffer;
  delete[] restore_buff;
}

void display()
{
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_SMOOTH);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushMatrix();


  if(menu->isVisible()){
    menu->display();

  }
  else{
    KeyStrokes::checkKey();      
    glTranslatef(0, 0, -100);
    camera->makeRotation();
    glTranslatef(0, 0, 100);
    StaticObj::drawSky();
    camera->makeTranslation();
    sends();
    objects->drawAllObjects();
  }
  glPopMatrix();
  glFlush();
  glutSwapBuffers();
}

void idleFunc()
{
  time_t start = clock();
  display();
  Sleep(33.333333);
  time_t finish = clock();
  float t = (float)(finish- start)/CLOCKS_PER_SEC;
  char title[50];
  strcpy(title, "");
  sprintf(title, "Collision test %f %f FPS", t, 1.0f/t);
  glutSetWindowTitle(title);
}

void exitGame()
{
  delete objects;
  delete camera;
  delete logicMap;
  exit(0);
}


void playSound(void*)
{
  _endthread();
}

void mouseControl( int button , int state , int x , int y)
{
  if( button == GLUT_LEFT_BUTTON ) 
    if( state == GLUT_DOWN ){
    }
    if(state == GLUT_UP){
      GLint m_viewport[4];
      glGetIntegerv( GL_VIEWPORT, m_viewport );
    }
}

void highMapdependentSettings()
{
  for(int i = 0; i < objects->components.size(); i++){
    if(strcmp(objects->components[i].name, "grass") == 0){
      for(int j = 0; j < objects->components[i].duplicates.size(); j++){
        objects->components[i].duplicates[j].y 
          = logicMap->getYPoint(objects->components[i].duplicates[j].x,
          objects->components[i].duplicates[j].z) -28.0f;
      }
    }
  }
}


int main(int argc, char **argv)
{
  client = new Client("10.22.2.119", 42425);
  srand ( time(NULL) );
  glutInit(&argc, argv); 
  glutInitDisplayMode (GLUT_DOUBLE| GLUT_RGBA | GLUT_DEPTH); 
  glutInitWindowSize (1280, 720); 
  glutInitWindowPosition (0, 0);
  glutCreateWindow ("Lights Game - Freya");
  glutReshapeFunc(myReshape);
  glewInit();
  SceneSettings::initlights();
  objNr_edit = 14;
  camera = new Camera();
  objects = new StaticObj("static_obj.txt");
  logicMap = 
    new HighMap(&objects->components[1].vertices, "HighMap.txt");
  highMapdependentSettings();
  menu = new MenuGUI();
  mainCharacter = 
    new MainCharacter("static_objects/freya_rotoscopie/f", camera);
    Vector3 in_poz;
    in_poz.x = -mainCharacter->position.x;
    in_poz.y = mainCharacter->position.y;
    in_poz.z = -mainCharacter->position.z-100;
  float* buffer 
    = client->getInitial(mainCharacter->frames[0].bounding_box[0], in_poz);
  client->sendBuffer(buffer, 8);
  delete[] buffer;
  char b_c[512];
  if(recv(client->sockfd, b_c, 512, NULL)){
    printf("rec: %s\n", b_c);
    client->setId(0);
  }
  else{
    std::cout<<"Error on recive"<<std::endl;
  }
  
  glClearColor(0,0,0,0);
  glutKeyboardFunc(KeyStrokes::keyPressed);
  glutKeyboardUpFunc(KeyStrokes::keyUp);
  glutMouseFunc(mouseControl);
  glutDisplayFunc(display);
  glutIdleFunc(idleFunc);
  glutMainLoop();
  return 0;
}

/**
* <img src="logo.gif" align="left"/>
* \authors Cosmin Mihai
* \version 1.0\n\n
* \date 2012
* \copyright GNU Public License.
* \details 
* - email: yonutix@yahoo.com\n
* - website: <a href ="http://www.freya.zzl.org">Freya</a>
* \mainpage
* <h1>Fisierele de configurare:</h1>
* <b>Fisierul de configurare pentru obiectele statice:</b>\n
* Proiectul trebuie sa contina in radacina un fisier "static_obj.txt" cu urmatorul format:\n
* name=[nume obiect]\n
* filename=[adresa obiect in format binar(vezi formatul binar al obiectelor)]\n
* texture=[adresa imaginii care va fi folosita ca textura in format TGA 8B uncompressed]\n
* pozX=[pozitia pe axa ox]\n
* pozY=[pozitia pe axa oy]\n
* pozZ=[pozitia pe axa oz]\n
* angle=[unghiul rotatiei]\n
* rotX=[componenta pe ox a vectorului rotatie]\n
* rotY=[componenta pe oy a vectorului rotatie]\n
* rotZ=[componenta pe oz a vectorului rotatie]\n
* scaleX=[componenta pe ox a scalarii]\n
* scaleY=[componenta pe oy a scalarii]\n
* scaleZ=[componenta pe oz a scalarii]\n
* //scaleX, scaleY, scaleZ nu sunt folosite in verisunea 1.0\n
* boundingbox=[Adresa bounding-box-ului in format bb - vezi format bb]\n
* <b>Fisierul de configurare pentru highmap</b>\n
* Se numeste "highMap.txt" si contine:\n
* lines=[numarul de vertecsi de-a lungul unei coloane]\n
* columns=[numarul de vertecsi de-a lungul unei linii]\n
* lineStep=[distanta dintre 2 vertecsi de pe linii consecutive]\n
* columnStep=[distanta dintre 2 coloane de vertecsi consecutive]\n
* <h1>Formate fisiere:</h1>
* <b>.bin</b>\n
* Fisier binar:\n
* |Numar vertecsi sizeof(int)|\ v(veretex.x vertex.y vertex.z) sizeof(float)* nr vertecsi * 3 \\\n
* |Numar normale sizeof(int)|\ v(normala.x normala.y normala.z) sizeof(float)* nr normale * 3 \\\n
* |Numar coordonate textura sizeof(int)|\ v(map.x map.y) sizeof(float)* nr normale * 2 \\\n
* |Numar fete sizeof(int)|\ numar fete * sizeof(Face)\\\n
* <b>.bb</b>\n
* Reprezinta un set de bounding-box-uri\n
* In versiunea 1.0 exista 2 tipuri de bounding-box-uri:\n
* - 0 Paralelipiped care este reprezentat de 6 float-uri(minPoint, maxPoint)\n
* - 1 Sfera care este reprezentata de 4 float-uri pozitia, raza\n
* - Pe prima linie se afla numarul de bounding-box-uri\n
* pe urmaroaterele se afla informatiile despre fiecare bounding-box sfera sau paralelipiped\n
*/