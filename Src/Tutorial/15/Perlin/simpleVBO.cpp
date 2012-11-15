/* 
   This wrapper demonstrates how to use the Cuda OpenGL bindings to
   dynamically modify data using a Cuda kernel and display it with opengl.
   
   The steps are:
   1. Create an empty vertex buffer object (VBO)
   2. Register the VBO with Cuda
   3. Map the VBO for writing from Cuda
   4. Run Cuda kernel to modify the vertex positions
   5. Unmap the VBO
   6. Render the results using OpenGL
   
   Host code
*/

// includes, GL
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>

// includes
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
//#include <cutil_gl_error.h>
#include <rendercheck_gl.h>

extern float animTime;

////////////////////////////////////////////////////////////////////////////////
// VBO specific code
#include <cuda_runtime.h>
#include <cutil_inline.h>

// constants
const unsigned int mesh_width = 128;
const unsigned int mesh_height = 128;
const unsigned int RestartIndex = 0xffffffff;

extern "C" 
void launch_kernel(float4* pos, uchar4* posColor,
		   unsigned int mesh_width, unsigned int mesh_height, float time);

// vbo variables
GLuint vbo;
GLuint colorVBO;

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, unsigned int typeSize)
{
  // create buffer object
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  
  // initialize buffer object
  unsigned int size = mesh_width * mesh_height * typeSize;
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  // register buffer object with CUDA
  cudaGLRegisterBufferObject(*vbo);
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo)
{
  glBindBuffer(1, *vbo);
  glDeleteBuffers(1, vbo);
  
  cudaGLUnregisterBufferObject(*vbo);
  
  *vbo = NULL;
}

void cleanupCuda()
{
  deleteVBO(&vbo);
  deleteVBO(&colorVBO);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    uchar4 *cptr;
    unsigned int *iptr;
    cudaGLMapBufferObject((void**)&dptr, vbo);
    cudaGLMapBufferObject((void**)&cptr, colorVBO);

    // execute the kernel
    launch_kernel(dptr, cptr, mesh_width, mesh_height, animTime);

    // unmap buffer object
    cudaGLUnmapBufferObject(vbo);
    cudaGLUnmapBufferObject(colorVBO);
}

void initCuda(int argc, char** argv)
{
  // First initialize OpenGL context, so we can properly set the GL
  // for CUDA.  NVIDIA notes this is necessary in order to achieve
  // optimal performance with OpenGL/CUDA interop.  use command-line
  // specified CUDA device, otherwise use device with highest Gflops/s
  if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
    cutilGLDeviceInit(argc, argv);
  } else {
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
  }
  
  createVBO(&vbo, sizeof(float4));
  createVBO(&colorVBO, sizeof(uchar4));
  // make certain the VBO gets cleaned up on program exit
  atexit(cleanupCuda);

  runCuda();

}

void renderCuda(int drawMode)
{
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  
  glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
  glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
  glEnableClientState(GL_COLOR_ARRAY);

  //glColor3f(1.0, 0.5, 0.0);
  switch(drawMode) {
  case GL_LINE_STRIP:
    for(int i=0 ; i < mesh_width*mesh_height; i+= mesh_width)
      glDrawArrays(GL_LINE_STRIP, i, mesh_width);
    break;
  case GL_TRIANGLE_FAN: {
// I left these in to show some alternative drawing methods
#define PRIMITIVE_RESTART
//#define SIMPLE_ONE_BY_ONE
//#define MULTI_DRAW

#ifdef PRIMITIVE_RESTART
    static GLuint* qIndices=NULL;
    int size = 5*(mesh_height-1)*(mesh_width-1);

    if(qIndices == NULL) { // allocate and assign trianglefan indicies TODO move to init
      qIndices = (GLuint *) malloc(size*sizeof(GLint));
      int index=0;
      for(int i=1; i < mesh_height; i++) {
	for(int j=1; j < mesh_width; j++) {
	  qIndices[index++] = (i)*mesh_width + j; 
	  qIndices[index++] = (i)*mesh_width + j-1; 
	  qIndices[index++] = (i-1)*mesh_width + j-1; 
	  qIndices[index++] = (i-1)*mesh_width + j; 
	  qIndices[index++] = RestartIndex;
	}
      }
    }
    glPrimitiveRestartIndexNV(RestartIndex);
    glEnableClientState(GL_PRIMITIVE_RESTART_NV);
    glDrawElements(GL_TRIANGLE_FAN, size, GL_UNSIGNED_INT, qIndices);
    glDisableClientState(GL_PRIMITIVE_RESTART_NV);
#endif
    
#ifdef SIMPLE_ONE_BY_ONE
    static GLuint* qIndices=NULL;
    int size = 4*(mesh_height-1)*(mesh_width-1);

    if(qIndices == NULL) { // allocate and assign trianglefan indicies TODO move to init
      qIndices = (GLuint *) malloc(size*sizeof(GLint));
      int index=0;
      for(int i=1; i < mesh_height; i++) {
	for(int j=1; j < mesh_width; j++) {
	  qIndices[index++] = (i)*mesh_width + j; 
	  qIndices[index++] = (i)*mesh_width + j-1; 
	  qIndices[index++] = (i-1)*mesh_width + j-1; 
	  qIndices[index++] = (i-1)*mesh_width + j; 
	}
      }
      fprintf(stderr,"size %d index %d\n",size,index);
    }
    for(int i=0; i < size; i +=4)
      glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_INT, &qIndices[i]);
    
#endif
    
#ifdef MULTI_DRAW
    static GLint* qIndices=NULL;
    static GLint* qCounts=NULL;
    static GLint** qIndex=NULL;
    int size = (mesh_height-1)*(mesh_width-1);

    if(qIndices == NULL) { // allocate and assign trianglefan indicies TODO move to init
      qIndices = (GLint *) malloc(4*size*sizeof(GLint));
      qCounts = (GLint *) malloc(size*sizeof(GLint));
      qIndex = (GLint **) malloc(size*sizeof(GLint*));

      int index=0;
      for(int i=1; i < mesh_height; i++)
	for(int j=1; j < mesh_width; j++) {
	  qIndices[index++] = ((i)*mesh_width + j); 
	  qIndices[index++] = ((i)*mesh_width + j-1); 
	  qIndices[index++] = ((i-1)*mesh_width + j-1); 
	  qIndices[index++] = ((i-1)*mesh_width + j); 
	}
      for(int i=0; i < size; i++) qCounts[i] = 4;
      for(int i=0; i < size; i++) qIndex[i] = &qIndices[i*4];
    }

    glMultiDrawElements(GL_TRIANGLE_FAN, qCounts,
			GL_UNSIGNED_INT, (const GLvoid**)qIndex, size);
#endif
  } break;
  default:
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    break;
  }

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}

