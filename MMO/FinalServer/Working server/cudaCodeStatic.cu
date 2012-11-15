
#include<math.h>
#define DIM 2



__device__ bool Cube_CubeTestStatic(float c1_x, float c1_z, float c1_size1, float c1_size2, float c2_x, float c2_z, float c2_size1, float c2_size2)
{
	if(abs(c1_x-c2_x) > (c1_size1+c2_size1)/2) return 0;
	
	if(abs(c1_z-c2_z) > (c1_size2+c2_size2)/2) return 0;

	return 1;
}

__global__ void launch_CubeStatic(float * BBstaticCenter_d,int nrBBstatic, 
						float * BBstaticSize_d, float xFuture, float yFuture, float zFuture, 
						float size1, float size2, int *okStatic_d);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void launch_CubeStatic(float * BBstaticCenter_d,int nrBBstatic, 
						float * BBstaticSize_d, float xFuture, float yFuture, float zFuture, 
						float size1, float size2, int *okStatic_d)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	//unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		
	//Daca cubul de la indicele ID aflat la viitoarea pozitie (x,y,z) se intersecteaza cu orice alt cub
	//dintre cele statice atunci pozitia lui ramane cea curenta altfel pozitia lui devine (x,y,z)
	*okStatic_d = 0;
	if (Cube_CubeTestStatic(BBstaticCenter_d[x], BBstaticCenter_d[1+x],BBstaticSize_d[x],BBstaticSize_d[1+x],
							xFuture,zFuture,size1,size2))
	{
		*okStatic_d = 1;
	}		
}

extern "C"
cudaError_t launch_CubeStatic(float * BBstaticCenter_d,int nrBBstatic, 
						float * BBstaticSize_d, float xFuture, float yFuture, float zFuture, 
						float size1, float size2, int *okStatic_d,dim3 DIM_GRID,dim3 DIM_BLOCK)
{
	launch_CubeStatic <<<DIM_GRID, DIM_BLOCK>>> (BBstaticCenter_d,nrBBstatic,BBstaticSize_d,size1,size2,
												xFuture,yFuture,zFuture,okStatic_d);
	
	return cudaGetLastError();
}