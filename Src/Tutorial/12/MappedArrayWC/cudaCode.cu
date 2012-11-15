__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N)
{
	// Identificarea pozitiei exacte a thread-ului
	int x;
	x = (blockIdx.x * blockDim.x + threadIdx.x);
	//if(x < N)
		r_d[x] = a_d[x] + 1.f;
	
	
}

extern "C"
cudaError_t launch_actiune_thread(float* a_d, float* b_d,float *r_d,int N,dim3 DIM_GRID, dim3 DIM_BLOCK)
{
	actiune_thread <<<DIM_GRID, DIM_BLOCK>>> (a_d, b_d,r_d,N);
	
	return cudaGetLastError();
}