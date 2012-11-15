
#include<math.h>
#define DIM 3
#define P_SCALE 30.0f
#define RIGHT P_SCALE
#define LEFT -P_SCALE
#define UP (2*P_SCALE)
#define DOWN 0
#define FRONT P_SCALE
#define BACK -P_SCALE
#define OFFSET 0.01
#define CONE_HEIGHT 1.0
#define CUBE_SIZE 1.0


__device__ bool Cube_CubeTest(float c1_x,float c1_y,float c1_z,float c1_size,
				   float c2_x,float c2_y,float c2_z,float c2_size)
{
	if(abs(c1_x-c2_x) > (c1_size*CUBE_SIZE+c2_size*CUBE_SIZE)/2) return 0;
	
	if(abs(c1_y-c2_y) > (c1_size*CUBE_SIZE+c2_size*CUBE_SIZE)/2) return 0;
	
	if(abs(c1_z-c2_z) > (c1_size*CUBE_SIZE+c2_size*CUBE_SIZE)/2) return 0;

	return 1;
}

__global__ void launch_Cube(float* cube_poz_d,
									  float* cube_speed_d, 
									  float* cube_size_d,
									  int NR_CUBES
										);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void launch_Cube(float* cube_poz_d,
									  float* cube_speed_d, 
									  float cube_size_d,
									  int NR_CUBES
									  )
{
	//calculate position
	
	//unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	
		if(cube_poz_d[y*DIM] >= (RIGHT -OFFSET) || cube_poz_d[y*DIM] <= (LEFT + OFFSET))
			cube_speed_d[y*DIM] = -cube_speed_d[y*DIM];
		if(cube_poz_d[1+y*DIM] >= (UP - OFFSET) || cube_poz_d[1+y*DIM] <= (DOWN + OFFSET))
			cube_speed_d[1+y*DIM] = -cube_speed_d[1+y*DIM];
		if(cube_poz_d[2+y*DIM] >= (FRONT - OFFSET) || cube_poz_d[2+y*DIM] <= (BACK + OFFSET))
			cube_speed_d[2+y*DIM] = -cube_speed_d[2+y*DIM];

		for(int j = (y+1)*DIM  ; j < NR_CUBES ; j=j+DIM) 
		{
			if(Cube_CubeTest(cube_poz_d[y*DIM],cube_poz_d[1+y*DIM],cube_poz_d[2+y*DIM],cube_size_d,cube_poz_d[j],
							 cube_poz_d[1+j],cube_poz_d[2+j],cube_size_d)) 
			{	
				cube_speed_d[j]   = -cube_speed_d[j];
				cube_speed_d[1+j] = -cube_speed_d[1+j];
				cube_speed_d[2+j] = -cube_speed_d[2+j];
			}
		}
	
		cube_poz_d[y*DIM]   += cube_speed_d[y*DIM];
		cube_poz_d[1+y*DIM] += cube_speed_d[1+y*DIM];
		cube_poz_d[2+y*DIM] += cube_speed_d[2+y*DIM];
}

extern "C"
cudaError_t launch_Cube(float* cube_poz_d,
									  float* cube_speed_d, 
									  float cube_size_d,
									  int NR_CUBES,
									  dim3 DIM_GRID, 
									  dim3 DIM_BLOCK)
{
	launch_Cube <<<DIM_GRID, DIM_BLOCK>>> (cube_poz_d,
									  cube_speed_d, 
									  cube_size_d,
									  NR_CUBES);
	
	return cudaGetLastError();
}