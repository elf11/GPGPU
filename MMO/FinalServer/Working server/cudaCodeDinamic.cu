
#include<math.h>
#define DIM 2



__device__ bool Cube_CubeTestDinamic(float c1_x, float c1_z, float c1_size1, float c1_size2, float c2_x, float c2_z, float c2_size1, float c2_size2)
{
	if(abs(c1_x-c2_x) > (c1_size1+c2_size1)/2) return 0;
	
	if(abs(c1_z-c2_z) > (c1_size2+c2_size2)/2) return 0;

	return 1;
}

__global__ void launch_CubeDinamic(float * BBdinamicCenter_d, int nrBBdinamic, float * BBdinamicSize_d, 
								   float xFuture, float yFuture, float zFuture, int ID, int *okDinamic);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void launch_CubeDinamic(float * BBdinamicCenter_d, int nrBBdinamic, float * BBdinamicSize_d, 
								   float xFuture, float yFuture, float zFuture, int ID, int *okDinamic)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	//unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	//Daca cubul de la indicele ID aflat la viitoarea pozitie (x,y,z) se intersecteaza cu orice alt cub
	//dintre cele dinamice atunci pozitia lui ramane cea curenta altfel pozitia lui devine (x,y,z)
	if (ID != x )
	{
		if (Cube_CubeTestDinamic(BBdinamicCenter_d[x], BBdinamicCenter_d[1+x],BBdinamicCenter_d[x],BBdinamicCenter_d[1+x],
								  xFuture,zFuture,BBdinamicSize_d[ID],BBdinamicSize_d[1+ID]))
		{
			*okDinamic = 1;
		}		
	}
}
extern "C"
cudaError_t launch_CubeDinamic(float * BBdinamicCenter_d, int nrBBdinamic, float * BBdinamicSize_d, 
							   float xFuture, float yFuture, float zFuture, int ID, int *okDinamic,dim3 DIM_GRID,dim3 DIM_BLOCK)
{
	launch_CubeDinamic <<<DIM_GRID, DIM_BLOCK>>> (BBdinamicCenter_d,nrBBdinamic,BBdinamicSize_d,xFuture,yFuture,zFuture,ID,okDinamic);
	
	return cudaGetLastError();
}