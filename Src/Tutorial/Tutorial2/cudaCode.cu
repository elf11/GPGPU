__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N)
{
	// Identificarea pozitiei exacte a thread-ului
	int x,y;
	x = (blockIdx.x * blockDim.x + threadIdx.x);
	//y = (blockIdx.y * blockDim.y + threadIdx.y);
	
	r_d[x] = a_d[x]+1.f;
	//for(int j=0;j<N;j++)
		//for(int k=0;k<N;k++)
				//r_d[x] += sqrt(a_d[j]+b_d[j]) / sqrt(a_d[k]+b_d[k]);
	
	
	//Operatii adaugate codului anterior
	//r_d[x] = 0;			
	//for(int j = 0; j < N; j++){
		
		//r_d[x] = a_d[x] +  a_d[j]*a_d[j];
		
	//}
	
	
}

extern "C"
cudaError_t launch_actiune_thread(float* a_d, float* b_d,float *r_d,int N,dim3 DIM_GRID, dim3 DIM_BLOCK)
{
	actiune_thread <<<DIM_GRID, DIM_BLOCK>>> (a_d, b_d,r_d,N);
	
	return cudaGetLastError();
}