
#include "stdafx.h"
#include "BaseObject.h"

BaseObject::BaseObject(void)
{
	DefaultSettings();
}

BaseObject::BaseObject(ObjectType oType)
{
	DefaultSettings();
	Type = oType;
}

BaseObject::BaseObject(Vector3 * oTranslation)
{
	DefaultSettings();
	SetPosition(oTranslation);
}

BaseObject::BaseObject(Vector3 *oTranslation, Vector3 *oRotation)
{
	DefaultSettings();
	SetPosition(oTranslation);
	SetRotation(oRotation);
}

void BaseObject::DefaultSettings(void)
{
	translation = Vector3(0.0f, 0.0f, 0.0f);
	rotation = Vector3(0.0f, 0.0f, 0.0f);

	Type = otUnknown;
}

void BaseObject::Draw ()
{

	glPushMatrix();

	glTranslatef( translation.x , translation.y , translation.z );

	glRotatef( rotation.x , 1.0 , 0.0 , 0.0 );
	glRotatef( rotation.y , 0.0 , 1.0 , 0.0 );
	glRotatef( rotation.z , 0.0 , 0.0 , 1.0 );

	GLfloat spec[]={1.0, 1.0 ,1.0 ,1.0}; 
	float df=100.0;

	glColor3f(color.x,color.y,color.z);
	glMaterialfv(GL_FRONT,GL_SPECULAR,spec);
	glMaterialfv(GL_FRONT,GL_SHININESS,&df);


	glPopMatrix();
}

void BaseObject::SetColor(Vector3 *oColor)
{
	color = * oColor;
}

void BaseObject::SetPosition(Vector3 *oTranslation)
{
	translation = * oTranslation;
}

void BaseObject::SetRotation(Vector3 * oRotation)
{
	rotation = * oRotation;
}

