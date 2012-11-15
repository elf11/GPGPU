#ifndef BASEOBJECT_H
#define BASEOBJECT_H

#include "stdafx.h"
#include <glut.h>
#include "Vector3.h"


enum ObjectType
{
	otUnknown,
	otCube,	
	otSphere,
	otCilinder
};


class BaseObject
{
// VARIABILE STATICE ::
//-------------------------------------------------
//public:
//	static Vector3 SelectedColor;
//	static Vector3 ColorIncrement;

// VARIABILE
//-------------------------------------------------
// publice
public :
	// incep cu litera mare
	ObjectType Type;	// tipul obiectuliu
	//bool Wireframe;		// daca va fi desenat wireframe
	//bool Visible;		// daca va fi sau nu desenat
	//bool Lighted;		// daca este sau nu luminat

// private
//protected:
	// incep cu litera mica
	Vector3 translation;	// pozitie
	Vector3 rotation;		// rotatie
//	Vector3 scale;			// scalare
	Vector3 color;			// culoare

// FUNCTII ::
//-------------------------------------------------
public:
// constructor
	// fara parametri
	BaseObject(void);		
	// doar cu tip
	BaseObject(ObjectType);
	// doar cu pozitie
	BaseObject(Vector3 *);			
	// cu pozitie, rotatie, scalare
	//BaseObject(Vector3 *,Vector3 *,Vector3 *);
	BaseObject(Vector3 *,Vector3 *);


// functie de desenare
	void virtual Draw(void);

// setters 
	// pentru toate variabilele care nu sunt publice
	//void Select(void);
	//void Deselect(void);
	void SetPosition(Vector3 *);
	void SetRotation(Vector3 *);
	void SetScale(Vector3 *);
	void SetColor(Vector3 *);

// getters
	// pentru toate variabilele care nu sunt publice, si pot fi modificate din exterior
//	Vector3 GetRotation(void);
//	Vector3 GetScale(void);
//	Vector3 GetPosition(void);

	// seteaza valorile default 
	void DefaultSettings(void);
	
};

#endif

/*
#ifndef BASEOBJECT__H
#define BASEOBJECT__H

#include "RMath.h"
#include "Renderer.h"

class BaseObject
{
private :
public :
	Vector3 pos;
	Quaternion orient;

	BaseObject(void);

	virtual void Update(void);
	virtual void Render(void);
};

#endif


*/