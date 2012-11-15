
/*
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
#define CONE_RADIUS 0.5


__device__ bool Cone_ConeTest(float c1_x,float c1_y,float c1_z,float c1_size,
				   float c2_x,float c2_y,float c2_z,float c2_size)
{
	if(c1_y > c2_y){
		float height_dif = c1_y - c2_y;
		if( height_dif > c1_size*CONE_HEIGHT)
			return 0;
		else{
			float new_radius = (c1_size*CONE_RADIUS * height_dif)/(c1_size*CONE_HEIGHT);
			float dist = 
				(c1_x - c2_x) *	(c1_x - c2_x) + 
				(c1_z - c2_z) *	(c1_z - c2_z);
			float minDist = new_radius + c2_size*CONE_RADIUS;
			
			return dist <= minDist * minDist;
		}
	}else{
		float height_dif = c2_y - c1_y;
		if( height_dif > c2_size*CONE_HEIGHT)
			return 0;
		else{
			float new_radius = (c2_size*CONE_RADIUS * height_dif)/(c2_size*CONE_HEIGHT);
			float dist = 
				(c1_x - c2_x) * (c1_x - c2_x) + 
				(c1_z - c2_z) *	(c1_z - c2_z);
			float minDist = new_radius + c1_size*CONE_RADIUS;
			
			return dist <= minDist * minDist;
		}
	}
	return 1;
}

__global__ void launch_Cone(float* cone_poz_d,
									  float* cone_speed_d, 
									  float* cone_size_d,
									  int NR_CONES
										);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void launch_Cone(float* cone_poz_d,
									  float* cone_speed_d, 
									  float* cone_size_d,
									  int NR_CONES
									  )
{
	//calculate position
	
	//unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	
		if(cone_poz_d[y*DIM] >= (RIGHT -OFFSET) || cone_poz_d[y*DIM] <= (LEFT + OFFSET))
			cone_speed_d[y*DIM] = -cone_speed_d[y*DIM];
		if(cone_poz_d[1+y*DIM] >= (UP - OFFSET) || cone_poz_d[1+y*DIM] <= (DOWN + OFFSET))
			cone_speed_d[1+y*DIM] = -cone_speed_d[1+y*DIM];
		if(cone_poz_d[2+y*DIM] >= (FRONT - OFFSET) || cone_poz_d[2+y*DIM] <= (BACK + OFFSET))
			cone_speed_d[2+y*DIM] = -cone_speed_d[2+y*DIM];
		for(int j = (y+1)*DIM  ; j < NR_CONES ; j=j+DIM) {
			if(Cone_ConeTest(
				cone_poz_d[y*DIM],cone_poz_d[1+y*DIM],cone_poz_d[2+y*DIM],cone_size_d[y*DIM],
				cone_poz_d[j],	  cone_poz_d[1+j],	  cone_poz_d[2+j],	  cone_size_d[j])) {
				
				cone_speed_d[j]   = -cone_speed_d[j];
				cone_speed_d[1+j] = -cone_speed_d[1+j];
				cone_speed_d[2+j] = -cone_speed_d[2+j];
			}
		}
	
		cone_poz_d[y*DIM]   += cone_speed_d[y*DIM];
		cone_poz_d[1+y*DIM] += cone_speed_d[1+y*DIM];
		cone_poz_d[2+y*DIM] += cone_speed_d[2+y*DIM];
}

extern "C"
cudaError_t launch_Cone(float* cone_poz_d,
									  float* cone_speed_d, 
									  float* cone_size_d,
									  int NR_CONES,
									  dim3 DIM_GRID, 
									  dim3 DIM_BLOCK)
{
	launch_Cone <<<DIM_GRID, DIM_BLOCK>>> (cone_poz_d,
									  cone_speed_d, 
									  cone_size_d,
									  NR_CONES);
	
	return cudaGetLastError();
}
*/