// includes, GL
#include <GL/glew.h>

// includes
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
//#include <cutil_gl_error.h>
#include <rendercheck_gl.h>

extern float animTime;

// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);

// Callbacks

int drawMode=GL_TRIANGLE_FAN; // the default draw mode

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

//! Display callback for GLUT
//! Keyboard events handler for GLUT
//! Display callback for GLUT
void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set view matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, translate_z);
  glRotatef(rotate_x, 1.0, 0.0, 0.0);
  glRotatef(rotate_y, 0.0, 1.0, 0.0);
  
  // run CUDA kernel to generate vertex positions
  runCuda();
  
  // render the data
  renderCuda(drawMode);
  
  glutSwapBuffers();
  glutPostRedisplay();
  
  animTime += 0.01;
}

extern float xStart,yStart,zOffset;
extern float gain, octaves, lacunarity;

//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
  switch(key) {
  case(27) : // exit
  case('q') :
    exit(0);
    break;
  case '+': // lower the ocean level
    zOffset += 0.01;
    zOffset = (zOffset > 1.0)? 1.0:zOffset; // guard input
    break;
  case '-': // raise the ocean level
    zOffset -= 0.01;
    zOffset = (zOffset < -1.0)? -1.0:zOffset; // guard input
    break;
  case 'k': // move withing Perlin function
    yStart -= 0.1;
    break;
  case 'j': 
    yStart += 0.1;
    break;
  case 'l': 
    xStart += 0.1;
    break;
  case 'h': 
    xStart -= 0.1;
    break;
  case 'd':
  case 'D':
    switch(drawMode) {
    case GL_POINTS: drawMode = GL_LINE_STRIP; break;
    case GL_LINE_STRIP: drawMode = GL_TRIANGLE_FAN; break;
    default: drawMode=GL_POINTS;
    } break;
  case 'I': // change gain
    gain += 0.25;
    break;
  case 'i': // change gain
    gain -= 0.25;
    gain = (gain < 0.25)?0.25:gain; // guard input
    break;
  case 'O': // change octaves
    octaves += 1.0f;
    octaves = (octaves > 8)?8:octaves; // guard input
    break;
  case 'o': // change octaves
    octaves -= 1.0f;
    octaves = (octaves<2)?2:octaves; // guard input
    break;
  case 'P': // change lacunarity
    lacunarity += 0.25;
    break;
  case 'p': // change lacunarity
    lacunarity -= 0.25;
    lacunarity = (lacunarity<0.2)?0.2:lacunarity; // guard input
    break;
  }
  glutPostRedisplay();
}

// Mouse event handlers for GLUT
void mouse(int button, int state, int x, int y)
{
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1<<button;
  } else if (state == GLUT_UP) {
    mouse_buttons = 0;
  }
  
  mouse_old_x = x;
  mouse_old_y = y;
  glutPostRedisplay();
}

void motion(int x, int y)
{
  float dx, dy;
  dx = x - mouse_old_x;
  dy = y - mouse_old_y;
  
  if (mouse_buttons & 1) {
    rotate_x += dy * 0.2;
    rotate_y += dx * 0.2;
  } else if (mouse_buttons & 4) {
    translate_z += dy * 0.01;
  }
  
  mouse_old_x = x;
  mouse_old_y = y;
}


