// Chestie.cpp : Defines the entry point for the console application.


#include <windows.h>		// Header File For Windows
#include <math.h>			// Header File For Math Library Routines
#include <stdio.h>			// Header File For Standard I/O Routines
#include <stdlib.h>			// Header File For Standard Library
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library
#include "tvector.h"
#include "tmatrix.h"
#include "tray.h"
#include <mmsystem.h>
#include "image.h"
//#include "Vector3.h"

#include <GL/gl.h>
#include <glut.h>

//CUDA 
#include <assert.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>

#define CUSTOM 1
#define MODE1 2
#define MODE2 3
#define MODE3 4
#define P_SCALE 500.0f
#define RIGHT P_SCALE
#define LEFT -P_SCALE
#define UP (2*P_SCALE)
#define DOWN 0
#define FRONT P_SCALE
#define BACK -P_SCALE
#define OFFSET 0.01

int const NR_BALLS = 100, NR_CUBES = 250, NR_CIL = 500; 

//int const NR_BALLS = 250, NR_CUBES = 500, NR_CIL = 1000;
//int const NR_BALLS = 500, NR_CUBES = 1000, NR_CIL = 2500;

void runCUDACube(void);
void runCUDASphere(void);


GLfloat spec[]={1.0, 1.0 ,1.0 ,1.0};      //sets specular highlight of balls
GLfloat posl[]={0,400,0,1};               //position of ligth source
GLfloat amb[]={0.2f, 0.2f, 0.2f ,1.0f};   //global ambient
GLfloat amb2[]={0.3f, 0.3f, 0.3f ,1.0f};  //ambient of lightsource

GLuint texture[2], dlist;                 //stores texture objects and display list

static GLint TO = 0;
static GLint FRAMES = 0;

TVector dir(0,0,-10);                     //initial direction of camera
TVector pos(0,-50,1000);                  //initial position of camera
float camera_rotation=0;                  //holds rotation around the Y axis


TVector veloc(0.5,-0.1,0.5);              //initial velocity of balls
TVector accel(0,-0.05,0);                 //acceleration ie. gravity of balls

double Time=0.6;                          //timestep of simulation
int hook_toball1=0, sounds=1;             //hook camera on ball, and sound on/off
                                          //Plane structure
struct Plane{
	        TVector _Position;
			TVector _Normal;
};
                                          //Cylinder structure
struct Cylinder{                          
	   TVector _Position;
       TVector _Axis;
       double _Radius;
};

Plane pl1,pl2,pl3,pl4,pl5;                //the 5 planes of the room

TVector cube_pos[NR_CUBES];
TVector cube_speed[NR_CUBES];

TVector sphere_poz[NR_BALLS];
TVector sphere_speed[NR_BALLS];

TVector OldPos[NR_BALLS];
TVector OldPosCube[NR_CUBES];
TVector OldPosCil[NR_CIL];

TVector cone_poz[NR_CIL];
TVector cone_speed[NR_CIL];

//Perform Intersection tests with primitives
int TestIntersionPlane(const Plane& plane,const TVector& position,const TVector& direction, double& lamda, TVector& pNormal);
int TestIntersionCylinder(const Cylinder& cylinder,const TVector& position,const TVector& direction, double& lamda, TVector& pNormal,TVector& newposition);
void LoadGLTextures();                    //Loads Texture Objects
void InitVars();
void idle();

HDC				hDC=NULL;			// Private GDI Device Context
HGLRC			hRC=NULL;			// Permanent Rendering Context
HWND			hWnd=NULL;			// Holds Our Window Handle
HINSTANCE		hInstance;			// Holds The Instance Of The Application

DEVMODE			DMsaved;			// Saves the previous screen settings (NEW)

bool			keys[256];			// Array Used For The Keyboard Routine
bool			active=TRUE;		// Window Active Flag Set To TRUE By Default
bool			fullscreen=TRUE;	// Fullscreen Flag Set To Fullscreen Mode By Default


int ProcessKeys();
LRESULT	CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);	// Declaration For WndProc

/************************************************************************************/

static void
printstring(void *font, char *string)
{
   int len, i;

   len = (int) strlen(string);
   for (i = 0; i < len; i++)
      glutBitmapCharacter(font, string[i]);
}

//CUDA CALLS

int block_size_cones = 16;
dim3 dimBlock(1,block_size_cones,1);
//dim3 dimGridCones( 1, (NR_CIL+dimBlock.y-1)/dimBlock.y,1);
dim3 dimGridCubes( 1, (NR_CUBES+dimBlock.y-1)/dimBlock.y,1);
dim3 dimGridSpheres( 1, (NR_BALLS+dimBlock.y-1)/dimBlock.y,1);

//GLfloat *cone_poz_d, *cone_speed_d,*cone_size_d;
GLfloat *cube_poz_d, *cube_speed_d,*cube_size_d;
GLfloat *sphere_poz_d, *sphere_speed_d,*sphere_size_d;

//FUNCTII EXTERNE C LAUNCH_CIL, LAUNCH_CUBE, LAUNCH_SPHERE

extern "C"
	cudaError_t launch_Sphere(float* sphere_poz_d,
									  float* sphere_speed_d, 
									  float sphere_size_d,
									  int NR_SPHERES,
									  dim3 DIM_GRID, 
									  dim3 DIM_BLOCK);

extern "C"
	cudaError_t launch_Cube(float* cube_poz_d,
									  float* cube_speed_d, 
									  float cube_size_d,
									  int NR_CUBES,
									  dim3 DIM_GRID, 
									  dim3 DIM_BLOCK);
/************************************************************************************/
// (no changes)

GLvoid ReSizeGLScene(GLsizei width, GLsizei height)		// Resize And Initialize The GL Window
{
	if (height==0)										// Prevent A Divide By Zero By
	{
		height=1;										// Making Height Equal One
	}

	glViewport(0,0,width,height);						// Reset The Current Viewport

	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(50.0f,(GLfloat)width/(GLfloat)height,10.f,1700.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix
}


int InitGL(GLvoid)										// All Setup For OpenGL Goes Here
{
   	float df=100.0;

	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations

	glClearColor(0,0,0,0);
  	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	glShadeModel(GL_SMOOTH);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);

	
	glMaterialfv(GL_FRONT,GL_SPECULAR,spec);
	glMaterialfv(GL_FRONT,GL_SHININESS,&df);

	glEnable(GL_LIGHTING);
	glLightfv(GL_LIGHT0,GL_POSITION,posl);
	glLightfv(GL_LIGHT0,GL_AMBIENT,amb2);
	glEnable(GL_LIGHT0);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,amb);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);
   
	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
   	
	glEnable(GL_TEXTURE_2D);
    LoadGLTextures();

	return TRUE;										// Initialization Went OK
}



GLvoid glDrawCube()
{
	float latura = 5.0f;
	glBegin(GL_QUADS);
	//top
		glColor3f(0.0f,1.0f,0.0f);          // Set The Color To Green
		glNormal3f(0.0,1.0,0.0);
		glVertex3f( latura, latura,-latura);          // Top Right Of The Quad (Top)
		glNormal3f(0.0,1.0,0.0);
		glVertex3f(-latura, latura,-latura);          // Top Left Of The Quad (Top)
		glNormal3f(0.0,1.0,0.0);
		glVertex3f(-latura, latura, latura);          // Bottom Left Of The Quad (Top)
		glNormal3f(0.0,1.0,0.0);
		glVertex3f( latura, latura, latura);          // Bottom Right Of The Quad (Top)

		glColor3f(1.0f,0.5f,0.0f);          // Set The Color To Orange
		glNormal3f(0.0,-1.0,0.0);
		glVertex3f( latura,-latura, latura);          // Top Right Of The Quad (Bottom)
		glNormal3f(0.0,-1.0,0.0);
		glVertex3f(-latura,-latura, latura);          // Top Left Of The Quad (Bottom)
		glNormal3f(0.0,-1.0,0.0);
		glVertex3f(-latura,-latura,-latura);          // Bottom Left Of The Quad (Bottom)
		glNormal3f(0.0,-1.0,0.0);
		glVertex3f( latura,-latura,-latura);          // Bottom Right Of The Quad (Bottom)
	
		glColor3f(1.0f,0.0f,0.0f);          // Set The Color To Red
		glNormal3f(0.0,0.0,1.0);
		glVertex3f( latura, latura, latura);          // Top Right Of The Quad (Front)
		glNormal3f(0.0,0.0,1.0);
		glVertex3f(-latura, latura, latura);          // Top Left Of The Quad (Front)
		glNormal3f(0.0,0.0,1.0);
		glVertex3f(-latura,-latura, latura);          // Bottom Left Of The Quad (Front)
		glNormal3f(0.0,0.0,1.0);
		glVertex3f( latura,-latura, latura);          // Bottom Right Of The Quad (Front)
	
		glColor3f(1.0f,1.0f,0.0f);          // Set The Color To Yellow
		glNormal3f(0.0,0.0,-1.0);
		glVertex3f( latura,-latura,-latura);          // Bottom Left Of The Quad (Back)
		glNormal3f(0.0,0.0,-1.0);
		glVertex3f(-latura,-latura,-latura);          // Bottom Right Of The Quad (Back)
		glNormal3f(0.0,0.0,-1.0);
		glVertex3f(-latura, latura,-latura);          // Top Right Of The Quad (Back)
		glNormal3f(0.0,0.0,-1.0);
		glVertex3f( latura, latura,-latura);          // Top Left Of The Quad (Back)

		glColor3f(0.0f,0.0f,1.0f);          // Set The Color To Blue
		glNormal3f(-1.0,0.0,0.0);
		glVertex3f(-latura, latura, latura);          // Top Right Of The Quad (Left)
		glNormal3f(-1.0,0.0,0.0);
		glVertex3f(-latura, latura,-latura);          // Top Left Of The Quad (Left)
		glNormal3f(-1.0,0.0,0.0);
		glVertex3f(-latura,-latura,-latura);          // Bottom Left Of The Quad (Left)
		glNormal3f(-1.0,0.0,0.0);
		glVertex3f(-latura,-latura, latura);          // Bottom Right Of The Quad (Left)

		glColor3f(1.0f,0.0f,1.0f);          // Set The Color To Violet
		glNormal3f(1.0,0.0,0.0);   
		glVertex3f( latura, latura,-latura);          // Top Right Of The Quad (Right)
		glNormal3f(1.0,0.0,0.0);
		glVertex3f( latura, latura, latura);          // Top Left Of The Quad (Right)
	    glNormal3f(1.0,0.0,0.0);
		glVertex3f( latura,-latura, latura);          // Bottom Left Of The Quad (Right)
	    glNormal3f(1.0,0.0,0.0);
		glVertex3f( latura,-latura,-latura);          // Bottom Right Of The Quad (Right)

		glEnd();
}

int DrawGLScene(GLvoid)	            // Here's Where We Do All The Drawing
{								
	int i;
	static char frbuf[80] = "";
	
	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    
    //set camera in hookmode 
	if (hook_toball1)
	{
		TVector unit_followvector=sphere_speed[0];
		unit_followvector.unit();
 		gluLookAt(sphere_speed[0].X()+250,sphere_speed[0].Y()+250 ,sphere_speed[0].Z(), sphere_speed[0].X()+sphere_speed[0].X() ,sphere_speed[0].Y()+sphere_speed[0].Y() ,sphere_speed[0].Z()+sphere_speed[0].Z() ,0,1,0);  
    
    }
	else
	    gluLookAt(pos._x,pos._y,pos._z, pos._x + dir.X(),pos._y+dir.Y(),pos._z+dir.Z(), 0,1.0,0.0);

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glRotatef(camera_rotation,0,1,0);
	
	for (i=0;i<NR_BALLS;i++)
	{
		switch(i % 7){
        case 1: glColor3f(1.0f,1.0f,1.0f);
			       break;
		case 2: glColor3f(1.0f,1.0f,0.0f);
			       break;
		case 3: glColor3f(0.0f,1.0f,1.0f);
			       break;
		case 4: glColor3f(0.0f,1.0f,0.0f);
			       break;
		case 5: glColor3f(0.0f,0.0f,1.0f);
			       break;
		case 6: glColor3f(0.65f,0.2f,0.3f);
			       break;
		case 7: glColor3f(1.0f,0.0f,1.0f);
			       break;
		case 8: glColor3f(0.0f,0.7f,0.4f);
			       break;
		default: glColor3f(1.0f,0,0);
		}
		glPushMatrix();
		glTranslated(sphere_poz[i]._x,sphere_poz[i]._y,sphere_poz[i]._z);
		GLUquadricObj *obj = gluNewQuadric();
		gluQuadricDrawStyle(obj,GLU_FILL);
		gluQuadricNormals(obj,GLU_SMOOTH);
		gluQuadricOrientation(obj,GLU_OUTSIDE);
		gluSphere(obj, 5.0f, 20, 20);
		glPopMatrix();
		gluDeleteQuadric(obj);
	}

	/*
	for (i=0;i<NR_CIL;i++)
	{
		switch(i % 7){
        case 1: glColor3f(1.0f,1.0f,1.0f);
			       break;
		case 2: glColor3f(1.0f,1.0f,0.0f);
			       break;
		case 3: glColor3f(0.0f,1.0f,1.0f);
			       break;
		case 4: glColor3f(0.0f,1.0f,0.0f);
			       break;
		case 5: glColor3f(0.0f,0.0f,1.0f);
			       break;
		case 6: glColor3f(0.65f,0.2f,0.3f);
			       break;
		case 7: glColor3f(1.0f,0.0f,1.0f);
			       break;
		case 8: glColor3f(0.0f,0.7f,0.4f);
			       break;
		default: glColor3f(1.0f,0,0);
		}
		glPushMatrix();
		glTranslated(cone_poz[i]._x,cone_poz[i]._y,cone_poz[i]._z);
		GLUquadricObj *obj = gluNewQuadric();
		gluQuadricDrawStyle(obj,GLU_FILL);
		gluQuadricNormals(obj,GLU_SMOOTH);
		gluQuadricOrientation(obj,GLU_OUTSIDE);
		gluCylinder(obj,5.0, 5.0, 10.0, 20.0, 20.0);
		glPopMatrix();
		gluDeleteQuadric(obj);
	}

	*/
	float df = 100.0;

	for (int i = 0; i < NR_CUBES; i += 1)
	{
		glPushMatrix();
		glTranslated(cube_pos[i]._x,cube_pos[i]._y,cube_pos[i]._z);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glDrawCube();
		glPopMatrix();
	}
	
	glEnable(GL_TEXTURE_2D);

	

	//glMaterialfv(GL_FRONT,GL_SPECULAR,spec);
	//glMaterialfv(GL_FRONT,GL_SHININESS,&df);
	
	//render walls(planes) with texture
	glBindTexture(GL_TEXTURE_2D, texture[1]); 
	glColor3f(1, 1, 1);
	glBegin(GL_QUADS);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(520,520,520);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(520,-520,520);
    glTexCoord2f(0.0f, 1.0f); glVertex3f(-520,-520,520);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-520,520,520);
        
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-520,520,-520);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(-520,-520,-520);
    glTexCoord2f(0.0f, 1.0f); glVertex3f(520,-520,-520);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(520,520,-520);
    
	glTexCoord2f(1.0f, 0.0f); glVertex3f(520,520,-520);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(520,-520,-520);
    glTexCoord2f(0.0f, 1.0f); glVertex3f(520,-520,520);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(520,520,520);
	
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-520,520,520);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(-520,-520,520);
    glTexCoord2f(0.0f, 1.0f); glVertex3f(-520,-520,-520);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-520,520,-520);
	glEnd();

	//render floor (plane) with colours
	glBindTexture(GL_TEXTURE_2D, texture[0]); 
    glBegin(GL_QUADS);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-520,-520,520);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(520,-520,520);
    glTexCoord2f(0.0f, 1.0f); glVertex3f(520,-520,-520);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-520,-520,-520);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	
	glRasterPos2i(10, 10);
	printstring(GLUT_BITMAP_HELVETICA_18, frbuf);
	glRasterPos2i(10, 10);
	
	FRAMES++;
	{
		GLint t = glutGet(GLUT_ELAPSED_TIME);
		if (t - TO >= 2000)
		{
			GLfloat seconds = (t - TO) / 1000.0;
			GLfloat fps = FRAMES / seconds;
			sprintf(frbuf, "Frame rate: %f", fps);
			TO = t;
			FRAMES = 0;
		}
	}
	return TRUE;										// Keep Going
}


GLvoid KillGLWindow(GLvoid)								// Properly Kill The Window
{
	if (fullscreen)										// Are We In Fullscreen Mode?
	{
		if (!ChangeDisplaySettings(NULL,CDS_TEST)) { // if the shortcut doesn't work
			ChangeDisplaySettings(NULL,CDS_RESET);		// Do it anyway (to get the values out of the registry)
			ChangeDisplaySettings(&DMsaved,CDS_RESET);	// change it to the saved settings
		} else {
			ChangeDisplaySettings(NULL,CDS_RESET);
		}
			
		ShowCursor(TRUE);								// Show Mouse Pointer
	}

	if (hRC)											// Do We Have A Rendering Context?
	{
		if (!wglMakeCurrent(NULL,NULL))					// Are We Able To Release The DC And RC Contexts?
		{
			MessageBoxA(NULL,"Release Of DC And RC Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		}

		if (!wglDeleteContext(hRC))						// Are We Able To Delete The RC?
		{
			MessageBoxA(NULL,"Release Rendering Context Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		}
		hRC=NULL;										// Set RC To NULL
	}

	if (hDC && !ReleaseDC(hWnd,hDC))					// Are We Able To Release The DC
	{
		MessageBoxA(NULL,"Release Device Context Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		hDC=NULL;										// Set DC To NULL
	}

	if (hWnd && !DestroyWindow(hWnd))					// Are We Able To Destroy The Window?
	{
		MessageBoxA(NULL,"Could Not Release hWnd.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		hWnd=NULL;										// Set hWnd To NULL
	}

	if (!UnregisterClassA("OpenGL",hInstance))			// Are We Able To Unregister Class
	{
		MessageBoxA(NULL,"Could Not Unregister Class.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		hInstance=NULL;									// Set hInstance To NULL
	}
}



/************************************************************************************/



/*	This Code Creates Our OpenGL Window.  Parameters Are:					*
 *	title			- Title To Appear At The Top Of The Window				*
 *	width			- Width Of The GL Window Or Fullscreen Mode				*
 *	height			- Height Of The GL Window Or Fullscreen Mode			*
 *	bits			- Number Of Bits To Use For Color (8/16/24/32)			*
 *	fullscreenflag	- Use Fullscreen Mode (TRUE) Or Windowed Mode (FALSE)	*/
 
BOOL CreateGLWindow(char* title, int width, int height, int bits, bool fullscreenflag)
{
	GLuint		PixelFormat;			// Holds The Results After Searching For A Match
	WNDCLASS	wc;						// Windows Class Structure
	DWORD		dwExStyle;				// Window Extended Style
	DWORD		dwStyle;				// Window Style
	RECT		WindowRect;				// Grabs Rectangle Upper Left / Lower Right Values
	WindowRect.left=(long)0;			// Set Left Value To 0
	WindowRect.right=(long)width;		// Set Right Value To Requested Width
	WindowRect.top=(long)0;				// Set Top Value To 0
	WindowRect.bottom=(long)height;		// Set Bottom Value To Requested Height

	fullscreen=fullscreenflag;			// Set The Global Fullscreen Flag

	hInstance			= GetModuleHandle(NULL);				// Grab An Instance For Our Window
	wc.style			= CS_HREDRAW | CS_VREDRAW | CS_OWNDC;	// Redraw On Size, And Own DC For Window.
	wc.lpfnWndProc		= (WNDPROC) WndProc;					// WndProc Handles Messages
	wc.cbClsExtra		= 0;									// No Extra Window Data
	wc.cbWndExtra		= 0;									// No Extra Window Data
	wc.hInstance		= hInstance;							// Set The Instance
	wc.hIcon			= LoadIcon(NULL, IDI_WINLOGO);			// Load The Default Icon
	wc.hCursor			= LoadCursor(NULL, IDC_ARROW);			// Load The Arrow Pointer
	wc.hbrBackground	= NULL;									// No Background Required For GL
	wc.lpszMenuName		= NULL;									// We Don't Want A Menu
	wc.lpszClassName	= "OpenGL";								// Set The Class Name
	
	EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &DMsaved); // save the current display state (NEW)

	if (fullscreen)												// Attempt Fullscreen Mode?
	{
		DEVMODE dmScreenSettings;								// Device Mode
		memset(&dmScreenSettings,0,sizeof(dmScreenSettings));	// Makes Sure Memory's Cleared
		dmScreenSettings.dmSize=sizeof(dmScreenSettings);		// Size Of The Devmode Structure
		dmScreenSettings.dmPelsWidth	= width;				// Selected Screen Width
		dmScreenSettings.dmPelsHeight	= height;				// Selected Screen Height
		dmScreenSettings.dmBitsPerPel	= bits;					// Selected Bits Per Pixel
		dmScreenSettings.dmFields=DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

		// Try To Set Selected Mode And Get Results.  NOTE: CDS_FULLSCREEN Gets Rid Of Start Bar.
		if (ChangeDisplaySettings(&dmScreenSettings,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL)
		{
			// If The Mode Fails, Offer Two Options.  Quit Or Use Windowed Mode.
			if (MessageBoxA(NULL,"The Requested Fullscreen Mode Is Not Supported By\nYour Video Card. Use Windowed Mode Instead?","Magic GL",MB_YESNO|MB_ICONEXCLAMATION)==IDYES)
			{
				fullscreen=FALSE;		// Windowed Mode Selected.  Fullscreen = FALSE
			}
			else
			{
				// Pop Up A Message Box Letting User Know The Program Is Closing.
				MessageBoxA(NULL,"Program Will Now Close.","ERROR",MB_OK|MB_ICONSTOP);
				return FALSE;									// Return FALSE
			}
		}
	}

	if (!RegisterClass(&wc))									// Attempt To Register The Window Class
	{
		MessageBoxA(NULL,"Failed To Register The Window Class.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;											// Return FALSE
	}

	if (fullscreen)												// Are We Still In Fullscreen Mode?
	{
		dwExStyle=WS_EX_APPWINDOW;								// Window Extended Style
		dwStyle=WS_POPUP;										// Windows Style
		ShowCursor(FALSE);										// Hide Mouse Pointer
	}
	else
	{
		dwExStyle=WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;			// Window Extended Style
		dwStyle=WS_OVERLAPPEDWINDOW;							// Windows Style
	}

	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);		// Adjust Window To True Requested Size

	// Create The Window
	if (!(hWnd=CreateWindowEx(	dwExStyle,							// Extended Style For The Window
								"OpenGL",							// Class Name
								"MagicRoom",								// Window Title
								dwStyle |							// Defined Window Style
								WS_CLIPSIBLINGS |					// Required Window Style
								WS_CLIPCHILDREN,					// Required Window Style
								0, 0,								// Window Position
								WindowRect.right-WindowRect.left,	// Calculate Window Width
								WindowRect.bottom-WindowRect.top,	// Calculate Window Height
								NULL,								// No Parent Window
								NULL,								// No Menu
								hInstance,							// Instance
								NULL)))								// Dont Pass Anything To WM_CREATE
	{
		KillGLWindow();								// Reset The Display
		MessageBoxA(NULL,"Window Creation Error.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	static	PIXELFORMATDESCRIPTOR pfd=				// pfd Tells Windows How We Want Things To Be
	{
		sizeof(PIXELFORMATDESCRIPTOR),				// Size Of This Pixel Format Descriptor
		1,											// Version Number
		PFD_DRAW_TO_WINDOW |						// Format Must Support Window
		PFD_SUPPORT_OPENGL |						// Format Must Support OpenGL
		PFD_DOUBLEBUFFER,							// Must Support Double Buffering
		PFD_TYPE_RGBA,								// Request An RGBA Format
		bits,										// Select Our Color Depth
		0, 0, 0, 0, 0, 0,							// Color Bits Ignored
		0,											// No Alpha Buffer
		0,											// Shift Bit Ignored
		0,											// No Accumulation Buffer
		0, 0, 0, 0,									// Accumulation Bits Ignored
		16,											// 16Bit Z-Buffer (Depth Buffer)  
		0,											// No Stencil Buffer
		0,											// No Auxiliary Buffer
		PFD_MAIN_PLANE,								// Main Drawing Layer
		0,											// Reserved
		0, 0, 0										// Layer Masks Ignored
	};
	
	if (!(hDC=GetDC(hWnd)))							// Did We Get A Device Context?
	{
		KillGLWindow();								// Reset The Display
		MessageBoxA(NULL,"Can't Create A GL Device Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	if (!(PixelFormat=ChoosePixelFormat(hDC,&pfd)))	// Did Windows Find A Matching Pixel Format?
	{
		KillGLWindow();								// Reset The Display
		MessageBoxA(NULL,"Can't Find A Suitable PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	if(!SetPixelFormat(hDC,PixelFormat,&pfd))		// Are We Able To Set The Pixel Format?
	{
		KillGLWindow();								// Reset The Display
		MessageBoxA(NULL,"Can't Set The PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	if (!(hRC=wglCreateContext(hDC)))				// Are We Able To Get A Rendering Context?
	{
		KillGLWindow();								// Reset The Display
		MessageBoxA(NULL,"Can't Create A GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	if(!wglMakeCurrent(hDC,hRC))					// Try To Activate The Rendering Context
	{
		KillGLWindow();								// Reset The Display
		MessageBoxA(NULL,"Can't Activate The GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	ShowWindow(hWnd,SW_SHOW);						// Show The Window
	SetForegroundWindow(hWnd);						// Slightly Higher Priority
	SetFocus(hWnd);									// Sets Keyboard Focus To The Window
	ReSizeGLScene(width, height);					// Set Up Our Perspective GL Screen

    if (!InitGL())									// Initialize Our Newly Created GL Window
	{
		KillGLWindow();								// Reset The Display
		MessageBoxA(NULL,"Initialization Failed.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return FALSE;								// Return FALSE
	}

	return TRUE;									// Success
}



/************************************************************************************/


LRESULT CALLBACK WndProc(	HWND	hWnd,			// Handle For This Window
							UINT	uMsg,			// Message For This Window
							WPARAM	wParam,			// Additional Message Information
							LPARAM	lParam)			// Additional Message Information
{
	switch (uMsg)									// Check For Windows Messages
	{
		case WM_ACTIVATE:							// Watch For Window Activate Message
		{
			if (!HIWORD(wParam))					// Check Minimization State
			{
				active=TRUE;						// Program Is Active
			}
			else
			{
				active=FALSE;						// Program Is No Longer Active
			}

			return 0;								// Return To The Message Loop
		}

		case WM_SYSCOMMAND:							// Intercept System Commands
		{
			switch (wParam)							// Check System Calls
			{
				case SC_SCREENSAVE:					// Screensaver Trying To Start?
				case SC_MONITORPOWER:				// Monitor Trying To Enter Powersave?
				return 0;							// Prevent From Happening
			}
			break;									// Exit
		}

		case WM_CLOSE:								// Did We Receive A Close Message?
		{
			PostQuitMessage(0);						// Send A Quit Message
			return 0;								// Jump Back
		}

		case WM_KEYDOWN:							// Is A Key Being Held Down?
		{
			keys[wParam] = TRUE;					// If So, Mark It As TRUE
			return 0;								// Jump Back
		}

		case WM_KEYUP:								// Has A Key Been Released?
		{
			keys[wParam] = FALSE;					// If So, Mark It As FALSE
			return 0;								// Jump Back
		}

		case WM_SIZE:								// Resize The OpenGL Window
		{
			ReSizeGLScene(LOWORD(lParam),HIWORD(lParam));  // LoWord=Width, HiWord=Height
			return 0;								// Jump Back
		}
	}

	// Pass All Unhandled Messages To DefWindowProc
	return DefWindowProc(hWnd,uMsg,wParam,lParam);
}



/************************************************************************************/



int WINAPI WinMain(	HINSTANCE	hInstance,			// Instance
					HINSTANCE	hPrevInstance,		// Previous Instance
					LPSTR		lpCmdLine,			// Command Line Parameters
					int			nCmdShow)			// Window Show State
{
	MSG		msg;									// Windows Message Structure
	BOOL	done=FALSE;								// Bool Variable To Exit Loop

	// Ask The User Which Screen Mode They Prefer
	if (MessageBoxA(NULL,"Would You Like To Run In Fullscreen Mode?", "Start FullScreen?",MB_YESNO|MB_ICONQUESTION)==IDNO)
	{
		fullscreen=FALSE;							// Windowed Mode
	}

	InitVars();                                     // Initialize Variables

	// Create Our OpenGL Window
	if (!CreateGLWindow("Magic Room",740,480,16,fullscreen))
	{
		return 0;									// Quit If Window Was Not Created
	}

	while(!done)									// Loop That Runs While done=FALSE
	{
		if (PeekMessage(&msg,NULL,0,0,PM_REMOVE))	// Is There A Message Waiting?
		{
			if (msg.message==WM_QUIT)				// Have We Received A Quit Message?
			{
				done=TRUE;							// If So done=TRUE
			}
			else									// If Not, Deal With Window Messages
			{
				TranslateMessage(&msg);				// Translate The Message
				DispatchMessage(&msg);				// Dispatch The Message
			}
		}
		else										// If There Are No Messages
		    if (active)
			{
				// Draw The Scene.  Watch For ESC Key And Quit Messages From DrawGLScene()
				if (keys[VK_ESCAPE])	// Active?  Was There A Quit Received?
				{
					done=TRUE;							// ESC or DrawGLScene Signalled A Quit
				}
				else									// Not Time To Quit, Update Screen
				{
					idle();                             // Advance Simulation
					DrawGLScene();                      // Draw Scene
					SwapBuffers(hDC);					// Swap Buffers (Double Buffering)
				}
	
				if (!ProcessKeys()) return 0;
			}
	}

	// Shutdown
	KillGLWindow();									// Kill The Window	
    glDeleteTextures(2,texture);
	return (msg.wParam);							// Exit The Program
}


//Collision Test!
bool Sphere2SphereTest (TVector s1Pos, TVector s2Pos, double s1Radius)
{	
	TVector *obj  = new TVector();
	
	TVector::subtract(s1Pos, s2Pos, *obj);
	
	float dist = obj->_x * obj->_x + obj->_y * obj->_y + obj->_z * obj->_z;
	float minDist = s1Radius;

	return dist <= minDist * minDist;
}

bool Cube2CubeTest(TVector c1Pos, TVector c2Pos, double c1Latura, double c2Latura)
{
	if(abs(c1Pos._x - c2Pos._x) > (c1Latura+c2Latura)/2) return 0;
	
	if(abs(c1Pos._y-c2Pos._y) > (c1Latura+c2Latura)/2) return 0;
	
	if(abs(c1Pos._z-c2Pos._z) > (c1Latura+c2Latura)/2) return 0;

	return 1;
}

bool Cilinder2CilinderTest(TVector c1Pos, TVector c2Pos, double c1Height, double c2Height, double c1Radius, double c2Radius){
	if(c1Pos._y > c2Pos._y){
		GLfloat height_dif = c1Pos._y - c2Pos._y;
		if( height_dif >c1Height)
			return 0;
		else{
			GLfloat new_radius = (c1Radius * height_dif)/c1Height;
			float dist = 
				(c1Pos._x - c2Pos._x) * 
				(c1Pos._x - c2Pos._x) + 
				(c1Pos._z - c2Pos._z) *
				(c1Pos._z - c2Pos._z);
			float minDist = new_radius + c2Radius;
			
			return dist <= minDist * minDist;
		}
	}else{
		GLfloat height_dif = c2Pos._y - c1Pos._y;
		if( height_dif > c2Height)
			return 0;
		else{
			GLfloat new_radius = (c2Radius * height_dif)/c2Height;
			float dist = 
				(c1Pos._x - c2Pos._x) * 
				(c1Pos._x - c2Pos._x) + 
				(c1Pos._z - c2Pos._z) *
				(c1Pos._z - c2Pos._z);
			float minDist = new_radius + c1Radius;
			
			return dist <= minDist * minDist;
		}
	}
	return 1;
}


/*************************************************************************************/
/*************************************************************************************/
/***                  Find if any of the current balls                            ****/
/***             intersect with each other in the current timestep                ****/
/***Returns the index of the 2 itersecting balls, the point and time of intersection */
/*************************************************************************************/
/*************************************************************************************/
int FindBallCol(TVector& point, double& TimePoint, double Time2, int& BallNr1, int& BallNr2)
{
	TVector RelativeV;
	TRay rays;
	double MyTime=0.0, Add=Time2/150.0, Timedummy=10000, Timedummy2=-1;
	TVector posi;
	
	//Test all balls against eachother in 150 small steps
	for (int i=0;i<NR_BALLS-1;i++)
	{
	 for (int j=i+1;j<NR_BALLS;j++)
	 {	
		    RelativeV=sphere_speed[i]-sphere_speed[j];
			rays=TRay(OldPos[i],TVector::unit(RelativeV));
			MyTime=0.0;

			if ( (rays.dist(OldPos[j])) > 40) continue; 

			while (MyTime<Time2)
			{
			   MyTime+=Add;
			   posi=OldPos[i]+RelativeV*MyTime;
			   if (posi.dist(OldPos[j])<=40) {
										   point=posi;
										   if (Timedummy>(MyTime-Add)) Timedummy=MyTime-Add;
										   BallNr1=i;
										   BallNr2=j;
										   break;
										}
			
			}
	 }

	}

	if (Timedummy!=10000) { TimePoint=Timedummy;
	                        return 1;
	}

	return 0;
}

/*************************************************************************************/
/*************************************************************************************/
/***             Main loop of the simulation                                      ****/
/***      Moves, finds the collisions and responses of the objects in the         ****/
/***      current time step.                                                      ****/
/*************************************************************************************/
/*************************************************************************************/

void idle()
{
  double rt,rt2,rt4,lamda=10000,rtCube,rt2Cube,rt4Cube,rtCil,rt2Cil,rt4Cil;
  TVector norm,uveloc;
  TVector normal,point,time;
  double RestTime,BallTime;
  TVector Pos2;
  int BallNr=0,dummy=0,BallColNr1,BallColNr2;
  TVector Nc;
  float xspeed = 0.4, yspeed = 0.2, zspeed = 0.25;

  if (!hook_toball1)
  {
	  camera_rotation+=0.1f;
	  if (camera_rotation>360)
		  camera_rotation=0;
  }
   RestTime=Time;
	  lamda=1000;

	//Compute velocity for next timestep using Euler equations
	for (int j=0;j<NR_BALLS;j++)
	  sphere_speed[j] += accel*RestTime;

	double R2 = RestTime/2;
	for (int j=0;j<NR_CIL;j++)
	  cone_speed[j] += accel*R2;

	double R4 = RestTime/4;
	for (int j=0;j<NR_CUBES;j++)
	  cube_speed[j] += accel*R4;
	
	while (RestTime>ZERO)
	{
	   lamda=10000;   //initialize to very large value
	   
	   /*
	   for(int i =0 ; i< NR_CIL ; i++){
		   if(cone_poz[i]._x >= (RIGHT -OFFSET) || cone_poz[i]._x <= (LEFT + OFFSET))
			cone_speed[i]._x = -cone_speed[i]._x;
		if(cone_poz[i]._y >= (UP - OFFSET) || cone_poz[i]._y <= (DOWN + OFFSET))
			cone_speed[i]._y = -cone_speed[i]._y;
		if(cone_poz[i]._z >= (FRONT - OFFSET) || cone_poz[i]._z <= (BACK + OFFSET))
			cone_speed[i]._z = -cone_speed[i]._z;
		}
		for(int i =0 ; i< NR_CUBES ; i++){
			if(cube_pos[i]._x >= (RIGHT -OFFSET) || cube_pos[i]._x <= (LEFT + OFFSET))
				cube_speed[i]._x = -cube_speed[i]._x;
			if(cube_pos[i]._y >= (UP - OFFSET) || cube_pos[i]._y <= (DOWN + OFFSET))
				cube_speed[i]._y = -cube_speed[i]._y;
			if(cube_pos[i]._z >= (FRONT - OFFSET) || cube_pos[i]._z <= (BACK + OFFSET))
				cube_speed[i]._z = -cube_speed[i]._z;
		}
		for(int i =0 ; i< NR_BALLS ; i++){
			if(sphere_poz[i]._x >= (RIGHT -OFFSET) || sphere_poz[i]._x <= (LEFT + OFFSET))
				sphere_speed[i]._x = -sphere_speed[i]._x;
			if(sphere_poz[i]._y >= (UP - OFFSET) || sphere_poz[i]._y <= (DOWN + OFFSET))
				sphere_speed[i]._y = -sphere_speed[i]._y;
			if(sphere_poz[i]._z >= (FRONT - OFFSET) || sphere_poz[i]._z <= (BACK + OFFSET))
				sphere_speed[i]._z = -sphere_speed[i]._z;
		}
		//bounding sphere test
		for(int i = 0 ; i < NR_BALLS ; i++) {
			for(int j = i + 1 ; j < NR_BALLS ; j++) {
				if(Sphere2SphereTest(sphere_poz[i],sphere_poz[j], 5.0)) {
				
					sphere_speed[j]._x = -sphere_speed[j]._x;
					sphere_speed[j]._y = -sphere_speed[j]._y;
					sphere_speed[j]._z = -sphere_speed[j]._z;
				}
			}
		}
		//bounding cone test
		for(int i = 0 ; i < NR_CIL ; i++) {
			for(int j = i + 1 ; j < NR_CIL ; j++) {
				if(Cilinder2CilinderTest(cone_poz[i],cone_poz[j], 5.0, 5.0, 10.0, 10.0)) {
				
					cone_speed[j]._x = -cone_speed[j]._x;
					cone_speed[j]._y = -cone_speed[j]._y;
					cone_speed[j]._z = -cone_speed[j]._z;
				}
			}
		}
		//bounding cube test
		for(int i = 0 ; i < NR_CUBES ; i++) {
			for(int j = i + 1 ; j < NR_CUBES ; j++) {
				if(Cube2CubeTest(cube_pos[i],cube_pos[j], 5.0, 5.0)) {
				
					cube_speed[j]._x = -cube_speed[j]._x;
					cube_speed[j]._y = -cube_speed[j]._y;
					cube_speed[j]._z = -cube_speed[j]._z;
				}
			}
		}

		// miscarea actuala
	for(int i =0 ; i< NR_CIL ; i++) {
		cone_poz[i]._x += cone_speed[i]._x;
		cone_poz[i]._y += cone_speed[i]._y;
		cone_poz[i]._z += cone_speed[i]._z;
	}
	for(int i =0 ; i< NR_CUBES ; i++) {
		cube_pos[i]._x += cube_speed[i]._x;
		cube_pos[i]._y += cube_speed[i]._y;
		cube_pos[i]._z += cube_speed[i]._z;
	}
	for(int i =0 ; i< NR_BALLS ; i++) {
		sphere_poz[i]._x += sphere_speed[i]._x;
		sphere_poz[i]._y += sphere_speed[i]._y;
		sphere_poz[i]._z += sphere_speed[i]._z;
	}

	*/

	runCUDASphere();

	runCUDACube();

	RestTime -= 0.00001;
	}
	
}

/*************************************************************************************/
/*************************************************************************************/
/***        Init Variables                                                        ****/
/*************************************************************************************/
/*************************************************************************************/

void InitVars()
{
	 //create planes
	pl1._Position=TVector(0,-500,0);
	pl1._Normal=TVector(0,1,0);
	pl2._Position=TVector(500,0,0);
	pl2._Normal=TVector(-1,0,0);
	pl3._Position=TVector(-500,0,0);
	pl3._Normal=TVector(1,0,0);
	pl4._Position=TVector(0,0,500);
	pl4._Normal=TVector(0,0,-1);
	pl5._Position=TVector(0,0,-500);
	pl5._Normal=TVector(0,0,1);

	int x, y;
	int xSet, ySet, zSet;
	for (int i = 0; i < NR_BALLS; i += 1)
	{
		xSet=((int)rand() % 2500)/10.0*pow(-1.0,rand()%10);
		ySet=((int)rand() % 2500)/10.0*pow(-1.0,rand()%10) + 30.0;
		zSet=((int)rand() % 2500)/10.0*pow(-1.0,rand()%10);
	
		sphere_poz[i]=TVector(xSet, ySet, zSet);
	}

	
	for (int i = 0; i < NR_CIL; i += 1)
	{
		xSet=((int)rand() % 3000)/10.0*pow(-1.0,rand()%10);
		ySet=((int)rand() % 3000)/10.0*pow(-1.0,rand()%10) + 30.0;
		zSet=((int)rand() % 3000)/10.0*pow(-1.0,rand()%10);
	
		cone_poz[i]=TVector(xSet, ySet, zSet);
	}

	for (int i = 0; i < NR_CUBES; i += 1)
	{
		xSet=((int)rand() % 3500)/10.0*pow(-1.0,rand()%10);
		ySet=((int)rand() % 3500)/10.0*pow(-1.0,rand()%10) + 30.0;
		zSet=((int)rand() % 3500)/10.0*pow(-1.0,rand()%10);
	
		cube_pos[i]=TVector(xSet, ySet, zSet);
	}
	
		// INITIALIZARI CUDA

	cutilSafeCall(cudaMalloc((void **) &cube_poz_d,NR_CUBES*3*sizeof(GLfloat)));
	cutilSafeCall(cudaMalloc((void **) &cube_speed_d,NR_CUBES*3*sizeof(GLfloat)));
	//cutilSafeCall(cudaMalloc((void **) &cube_size_d,NR_CUBES*sizeof(GLfloat)));

	cutilSafeCall(cudaMalloc((void **) &sphere_poz_d,NR_BALLS*3*sizeof(GLfloat)));
	cutilSafeCall(cudaMalloc((void **) &sphere_speed_d,NR_BALLS*3*sizeof(GLfloat)));
	//cutilSafeCall(cudaMalloc((void **) &sphere_size_d,NR_SPHERES*sizeof(GLfloat)));
	//*/

	return;
}


/*---------------------------------------------------------------
--------------MORE CUDA STUFF------------------------------------
*/

void cleanup()
{

	cudaFree(cube_poz_d); 
	cudaFree(cube_speed_d);
	cudaFree(sphere_poz_d); 
	cudaFree(sphere_speed_d);

	cudaThreadExit();
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();

    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,  cudaGetErrorString( err) );
		getchar();
        exit(EXIT_FAILURE);
    }                         
}

bool initCUDA(void)
{
#if __DEVICE_EMULATION__
	return true;
#else
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "Nu exista nici un device.\n");
		return false;
	}

	printf("Exista %d device-uri.\n",count);
	
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "Nu exista nici un device care suporta CUDA.\n");
		return false;
	}
	cudaSetDevice(cutGetMaxGflopsDeviceId());
	
	printf("CUDA initializat cu succes\n");

	return true;
#endif
}

/*
void runCUDACone()
{
	// Copiaza matricile de prelucrat la device
	cutilSafeCall(cudaMemcpy(cone_poz_d, cone_poz,NR_CONES*3*sizeof(GLfloat),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	cutilSafeCall(cudaMemcpy(cone_speed_d, cone_speed,NR_CONES*3*sizeof(GLfloat),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	//cutilSafeCall(cudaMemcpy(cone_size_d, cone_size,NR_CONES*sizeof(GLfloat),cudaMemcpyHostToDevice));
	//checkCUDAError("cudaMemcpy");

	// Run Kernel
	cutilSafeCall(launch_Cone(cone_poz_d,cone_speed_d,cone_size_d,NR_CONES,dimGridCones,dimBlock));
	cutilSafeCall(cudaThreadSynchronize());
	checkCUDAError("invocare kernel");

	// Copiaza rezultatul prelucrat
	cutilSafeCall(cudaMemcpy(cone_poz,cone_poz_d,NR_CONES*3*sizeof(GLfloat),cudaMemcpyDeviceToHost));
	checkCUDAError("cudaMemcpy");
	
	cutilSafeCall(cudaMemcpy(cone_speed,cone_speed_d,NR_CONES*3*sizeof(GLfloat),cudaMemcpyDeviceToHost));
	checkCUDAError("cudaMemcpy");
	
}

*/
void runCUDACube()
{
	// Copiaza matricile de prelucrat la device
	cutilSafeCall(cudaMemcpy(cube_poz_d, cube_pos,NR_CUBES*3*sizeof(GLfloat),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	cutilSafeCall(cudaMemcpy(cube_speed_d, cube_speed,NR_CUBES*3*sizeof(GLfloat),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	//cutilSafeCall(cudaMemcpy(cube_size_d, cube_size,NR_CUBES*sizeof(GLfloat),cudaMemcpyHostToDevice));
	//checkCUDAError("cudaMemcpy");

	// Run Kernel
	cutilSafeCall(launch_Cube(cube_poz_d,cube_speed_d,5.0,NR_CUBES,dimGridCubes,dimBlock));
	cutilSafeCall(cudaThreadSynchronize());
	checkCUDAError("invocare kernel");

	// Copiaza rezultatul prelucrat
	cutilSafeCall(cudaMemcpy(cube_pos,cube_poz_d,NR_CUBES*3*sizeof(GLfloat),cudaMemcpyDeviceToHost));
	checkCUDAError("cudaMemcpy");
	
	cutilSafeCall(cudaMemcpy(cube_speed,cube_speed_d,NR_CUBES*3*sizeof(GLfloat),cudaMemcpyDeviceToHost));
	checkCUDAError("cudaMemcpy");
	
}
void runCUDASphere()
{
	// Copiaza matricile de prelucrat la device
	cutilSafeCall(cudaMemcpy(sphere_poz_d, sphere_poz,NR_BALLS*3*sizeof(GLfloat),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	cutilSafeCall(cudaMemcpy(sphere_speed_d, sphere_speed,NR_BALLS*3*sizeof(GLfloat),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	//cutilSafeCall(cudaMemcpy(sphere_size_d, sphere_size,NR_SPHERES*sizeof(GLfloat),cudaMemcpyHostToDevice));
	//checkCUDAError("cudaMemcpy");

	// Run Kernel
	cutilSafeCall(launch_Sphere(sphere_poz_d,sphere_speed_d,5.0,NR_BALLS,dimGridSpheres,dimBlock));
	cutilSafeCall(cudaThreadSynchronize());
	checkCUDAError("invocare kernel");

	// Copiaza rezultatul prelucrat
	cutilSafeCall(cudaMemcpy(sphere_poz,sphere_poz_d,NR_BALLS*3*sizeof(GLfloat),cudaMemcpyDeviceToHost));
	checkCUDAError("cudaMemcpy");
	
	cutilSafeCall(cudaMemcpy(sphere_speed,sphere_speed_d,NR_BALLS*3*sizeof(GLfloat),cudaMemcpyDeviceToHost));
	checkCUDAError("cudaMemcpy");
	
}

/*************************************************************************************/
/*************************************************************************************/
/***        Fast Intersection Function between ray/plane                          ****/
/*************************************************************************************/
/*************************************************************************************/
int TestIntersionPlane(const Plane& plane,const TVector& position,const TVector& direction, double& lamda, TVector& pNormal)
{

    double DotProduct=direction.dot(plane._Normal);
	double l2;

    //determine if ray paralle to plane
    if ((DotProduct<ZERO)&&(DotProduct>-ZERO)) 
		return 0;

    l2=(plane._Normal.dot(plane._Position-position))/DotProduct;

    if (l2<-ZERO) 
		return 0;

    pNormal=plane._Normal;
	lamda=l2;
    return 1;

}

/*************************************************************************************/
/*************************************************************************************/
/***        Load Bitmaps And Convert To Textures                                  ****/
/*************************************************************************************/
void LoadGLTextures() {	
     /* Load Texture*/
    Image *image1, *image4;
    
    /* allocate space for texture*/
    image1 = (Image *) malloc(sizeof(Image));
    if (image1 == NULL) {
	printf("Error allocating space for image");
	exit(0);
    }
	
	image4 = (Image *) malloc(sizeof(Image));
    if (image4 == NULL) {
	printf("Error allocating space for image");
	exit(0);
    }

    if (!ImageLoad("data/boden.bmp", image1)) {
	exit(1);
    } 
	/*
	if (!ImageLoad("data/marble.bmp", image3)) {
	exit(1);
    } 
	*/
	if (!ImageLoad("data/wand.bmp", image4)) {
	exit(1);
    }

    /* Create Texture	*****************************************/
    glGenTextures(2, &texture[0]);
    glBindTexture(GL_TEXTURE_2D, texture[0]);   /* 2d texture (x and y size)*/

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR); /* scale linearly when image bigger than texture*/
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); /* scale linearly when image smalled than texture*/
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_REPEAT);

    /* 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image, */
    /* border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.*/
    glTexImage2D(GL_TEXTURE_2D, 0, 3, image1->sizeX, image1->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, image1->data);


	/* Create Texture	*********************************************/
    glBindTexture(GL_TEXTURE_2D, texture[1]);   /* 2d texture (x and y size)*/

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR); /* scale linearly when image bigger than texture*/
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); /* scale linearly when image smalled than texture*/
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_REPEAT);

    /* 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image, */
    /* border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.*/
    glTexImage2D(GL_TEXTURE_2D, 0, 3, image4->sizeX, image4->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, image4->data);

	free(image1->data);
	free(image1);
	free(image4->data);
	free(image4);

};

int ProcessKeys()
{
	if (keys[VK_UP])    pos+=TVector(0,0,-10);
	if (keys[VK_UP])    pos+=TVector(0,0,10);
	if (keys[VK_LEFT])  camera_rotation+=10;
	if (keys[VK_RIGHT]) camera_rotation-=10;
	if (keys[VK_ADD])
	{
		Time+=0.1;
        keys[VK_ADD]=FALSE;
	}
	if (keys[VK_SUBTRACT])
	{
		Time-=0.1;
        keys[VK_SUBTRACT]=FALSE;
	}
	if (keys[VK_F3]) 
	{   
		sounds^=1;
        keys[VK_F3]=FALSE;
	}
	if (keys[VK_F2])
	{
	    hook_toball1^=1;
	    camera_rotation=0;
		keys[VK_F2]=FALSE;
	}
	if (keys[VK_F1])						// Is F1 Being Pressed?
	{
		keys[VK_F1]=FALSE;					// If So Make Key FALSE
		KillGLWindow();						// Kill Our Current Window
		fullscreen=!fullscreen;				// Toggle Fullscreen / Windowed Mode
		// Recreate Our OpenGL Window
		if (!CreateGLWindow("Magic Room",640,480,16,fullscreen))
		{
			return 0;						// Quit If Window Was Not Created
		}
	}

	return 1;
}