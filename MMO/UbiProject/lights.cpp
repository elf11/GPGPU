#include "lights.h"

void SceneSettings::initlights()
{
	GLfloat ambient[] = { 5 , 5 , 5 , 5 };
	GLfloat diffuse[] = { 3 , 3 , 3 , 3 };
	GLfloat position[] = { 0 , 0 , 1 , 2 };

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0,GL_AMBIENT,ambient);
	glLightfv(GL_LIGHT0,GL_AMBIENT,diffuse);
	glLightfv(GL_LIGHT0,GL_POSITION,position);
}