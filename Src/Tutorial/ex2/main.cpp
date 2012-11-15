#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <windows.h>


// Parametrii pentru a seta dimensiunile gridului si ale blocurilor
int block_size = 32;
// Dimensiunea vectorului
const int N = 1024;


dim3 dimBlock(block_size, 1,1);
dim3 dimGrid(N / dimBlock.x, N / dimBlock.y,1);

// Matrice locale
float a_h[N][N],b_h[N][N],r_h[N][N];
// Rezultatul de control
float control[N][N];
// Matrice pentru device
float *a_d,*b_d,*r_d;

// Functia de lanasare a kernelului CUDA
extern "C"
cudaError_t launch_actiune_thread(float* a_d, float* b_d,float *c_d,int N,dim3 DIM_GRID, dim3 DIM_BLOCK);

unsigned int timer = 0;

void init()
{
	
	// Aloca memorie - CUDA
	cutilSafeCall(cudaMalloc((void **) &a_d, N*N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &b_d, N*N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &r_d, N*N*sizeof(float)));

	cutilSafeCall(cudaMemset(r_d,0x0,N*N*sizeof(float)));

	// Initializeaza vectori
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
		{
			a_h[i][j] = (float)(i % 13)+1;
			b_h[i][j] = (float)(i % 3)+1;
		}
}


void cleanup()
{
	cutilCheckError(cutStopTimer(timer));
	cutilCheckError(cutDeleteTimer( timer));

	cudaFree(a_d);cudaFree(b_d);cudaFree(r_d);

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
	// Copiaza matricile de prelucrat la device
	cutilSafeCall(cudaMemcpy(a_d, a_h,N*N*sizeof(float),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	cutilSafeCall(cudaMemcpy(b_d, b_h,N*N*sizeof(float),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	// Run Kernel
	cutilSafeCall(launch_actiune_thread(a_d,b_d,r_d,N,dimGrid,dimBlock));
	cutilSafeCall(cudaThreadSynchronize());
	checkCUDAError("invocare kernel");

	// Copiaza rezultatul prelucrat
	cutilSafeCall(cudaMemcpy(r_h,r_d,N*N*sizeof(float),cudaMemcpyDeviceToHost));
	checkCUDAError("cudaMemcpy");
	
	
}

// Se calculeaza matricea produs de control
void computeControl()
{
	int i,j,k,sum=0;
	for (i=0;i<N;i++)
	{
		for (j=0;j<N;j++)
		{
			for (k=0;k<N;k++)
				sum=sum+(a_h[i][k]*b_h[k][j]);
			control[i][j]=sum;
			sum=0;
		}
	}
	return;
}

void verificaCalcule()
{
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
		if(control[i][j] != r_h[i][j])
		{
			printf("INCORECT\n");
			printf("Valoare CUDA : %f\n",r_h[i][j]);
			printf("Valoare LOCALA : %f\n",control[i][j]);
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