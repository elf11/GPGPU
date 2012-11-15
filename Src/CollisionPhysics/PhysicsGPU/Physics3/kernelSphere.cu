

#define DIM 3
#define P_SCALE 30.0f
#define RIGHT P_SCALE
#define LEFT -P_SCALE
#define UP (2*P_SCALE)
#define DOWN 0
#define FRONT P_SCALE
#define BACK -P_SCALE
#define OFFSET 0.01
#define SPHERE_RADIUS 1.0


__device__ bool Sphere_SphereTest(float c1_x,float c1_y,float c1_z,float c1_size,
				   float c2_x,float c2_y,float c2_z,float c2_size)
{
	float dist = (c1_x-c2_x)*(c1_x-c2_x)+(c1_y-c2_y)*(c1_y-c2_y)+(c1_z-c2_z)*(c1_z-c2_z);
	float minDist = c1_size*SPHERE_RADIUS + c2_size*SPHERE_RADIUS;

	return dist <= minDist*minDist;
}

__global__ void launch_Sphere(float* sphere_poz_d,
									  float* sphere_speed_d, 
									  float* sphere_size_d,
									  int NR_SPHERES
										);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void launch_Sphere(float* sphere_poz_d,
									  float* sphere_speed_d, 
									  float sphere_size_d,
									  int NR_SPHERES
									  )
{
	//calculate position
	
	//unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	
		if(sphere_poz_d[y*DIM] >= (RIGHT -OFFSET) || sphere_poz_d[y*DIM] <= (LEFT + OFFSET))
			sphere_speed_d[y*DIM] = -sphere_speed_d[y*DIM];
		if(sphere_poz_d[1+y*DIM] >= (UP - OFFSET) || sphere_poz_d[1+y*DIM] <= (DOWN + OFFSET))
			sphere_speed_d[1+y*DIM] = -sphere_speed_d[1+y*DIM];
		if(sphere_poz_d[2+y*DIM] >= (FRONT - OFFSET) || sphere_poz_d[2+y*DIM] <= (BACK + OFFSET))
			sphere_speed_d[2+y*DIM] = -sphere_speed_d[2+y*DIM];

		for(int j = (y+1)*DIM  ; j < NR_SPHERES ; j=j+DIM) 
		{
			if(Sphere_SphereTest(sphere_poz_d[y*DIM],sphere_poz_d[1+y*DIM],sphere_poz_d[2+y*DIM],sphere_size_d, 
								 sphere_poz_d[j],sphere_poz_d[1+j],sphere_poz_d[2+j],sphere_size_d)) 
			{
				
				sphere_speed_d[j]   = -sphere_speed_d[j];
				sphere_speed_d[1+j] = -sphere_speed_d[1+j];
				sphere_speed_d[2+j] = -sphere_speed_d[2+j];
			}
		}
	
		sphere_poz_d[y*DIM]   += sphere_speed_d[y*DIM];
		sphere_poz_d[1+y*DIM] += sphere_speed_d[1+y*DIM];
		sphere_poz_d[2+y*DIM] += sphere_speed_d[2+y*DIM];
}

extern "C"
cudaError_t launch_Sphere(float* sphere_poz_d,
									  float* sphere_speed_d, 
									  float sphere_size_d,
									  int NR_SPHERES,
									  dim3 DIM_GRID, 
									  dim3 DIM_BLOCK)
{
	launch_Sphere <<<DIM_GRID, DIM_BLOCK>>> (sphere_poz_d,
									  sphere_speed_d, 
									  sphere_size_d,
									  NR_SPHERES);
	
	return cudaGetLastError();
}