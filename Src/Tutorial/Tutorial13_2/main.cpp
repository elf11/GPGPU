#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <windows.h>

//texture<float, 1, cudaReadModeElementType> texRef;

// Parametrii pentru a seta dimensiunile gridului si ale blocurilor
int block_size = 256;
// Dimensiunea vectorului
int N = 2560;


dim3 dimBlock(block_size, 1);
dim3 dimGrid((N +block_size-1)/ dimBlock.x, 1);
void checkCUDAError(const char *msg);


// Vectorii locali
float *a_h,*b_h,*r_h;
// Rezultatul de control
float *control;
// Vectorii pentru device
float *a_d,*b_d,*r_d;

// Functia de lanasare a kernelului CUDA
extern "C"
cudaError_t launch_actiune_thread(float* a_d, float* b_d,float *c_d,int N,dim3 DIM_GRID, dim3 DIM_BLOCK);
extern "C"
cudaError_t release();
extern "C"
cudaError_t legare(size_t * offset,const void * devPtr,int N);

unsigned int timer = 0;


void init()
{
	
	// Aloca memorie - local
	a_h = (float *)malloc(N*sizeof(float));
	b_h = (float *)malloc(N*sizeof(float));
	r_h = (float *)malloc(N*sizeof(float));
	control = (float *)malloc(N*sizeof(float));

	// Aloca memorie - CUDA
	cutilSafeCall(cudaMalloc((void **) &a_d, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &b_d, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &r_d, N*sizeof(float)));

	// Initializeaza vectori
	for(int i=0;i<N;i++)
	{
		a_h[i] = (float)(i % 13)+1;
		b_h[i] = (float)(i % 3)+1;
	}
}


void cleanup()
{
	free(a_h);free(b_h);free(r_h);
	free(control);

	cutilCheckError(cutStopTimer(timer));
	cutilCheckError(cutDeleteTimer( timer));

	cudaFree(a_d);cudaFree(b_d);cudaFree(r_d);

	
	cutilSafeCall(release());
	checkCUDAError("release");
	cudaThreadExit();
      
}

// Verifica daca exista eroare CUDA
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

	// Create the CUTIL timer
    cutilCheckError( cutCreateTimer( &timer));

	return true;
#endif
}

// Lanseaza procesarea CUDA
void runCUDA()
{
	// Copiaza vectorii de prelucrat la device
	cutilSafeCall(cudaMemcpy(a_d, a_h,N*sizeof(float),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");
	cutilSafeCall(legare(0,a_d,N*sizeof(float)));

	cutilSafeCall(cudaMemcpy(b_d, b_h,N*sizeof(float),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");
	cutilSafeCall(legare(0,b_d,N*sizeof(float)));

	// Run Kernel
	cutilSafeCall(launch_actiune_thread(a_d,b_d,r_d,N,dimGrid,dimBlock));
	cutilSafeCall(cudaThreadSynchronize());
	checkCUDAError("invocare kernel");

	// Copiaza rezultatul prelucrat
	cutilSafeCall(cudaMemcpy(r_h,r_d,N*sizeof(float),cudaMemcpyDeviceToHost));
	checkCUDAError("cudaMemcpy");
	
	
}

// Se calculeaza sample-ul de control
void computeControl()
{
	for(int i=0;i<N;i++)
	{
		control[i] = -a_h[i];
		
	}
}

void verificaCalcule()
{
	for(int i=0;i<N;i++)
		if(control[i] != r_h[i])
		{
			printf("INCORECT\n");
			printf("CUDA : %f\n",r_h[i]);
			return;
		}
	printf("CORECT\n");

	return;
}

// Afisare text
void printHeader(char *s)
{
	int line_len = 79;
	line_len -= strlen(s);
	line_len /=2;

	for(int i=0;i<line_len;i++)
		printf("*");
	printf("%s",s);
	for(int i=0;i<line_len;i++)
		printf("*");
	printf("\n");
}

int main(int argc, char** argv)
{
	printHeader("Initializare");
	initCUDA();
	init();
		
	printHeader("Calcul CPU");
	cutilCheckError(cutStartTimer(timer));

	// Calculeaza sampleul de control - CPU
	printf("Asteptati: Se calculeaza controlul pe CPU ... ");
	computeControl();
	printf("DONE\n");
	float time = cutGetTimerValue(timer);
	printf("Timp de calcul pe CPU = %f milisecunde\n",time);
	
	cutilCheckError(cutResetTimer(timer));
	
	printHeader("Calcul CUDA");
	// Se calculeaza pe CUDA
	printf("Asteptati: Se calculeaza pe CUDA ... ");
	runCUDA();
	printf("DONE\n");
	time = cutGetTimerValue(timer);
	printf("Timp de calcul pe GPU = %f milisecunde\n",time);
	
	printHeader("Verificare calcule");
	// Se verifica daca s-a calculat corect pe CUDA
	printf("Se verifica daca rezultatul pe CUDA corespunde cu rezultatul pe CPU : ");
	verificaCalcule();
	printHeader("");

	cleanup();

	printf("Apasa ENTER pentru a termina programul\n");
	getchar();

	return 0;
}