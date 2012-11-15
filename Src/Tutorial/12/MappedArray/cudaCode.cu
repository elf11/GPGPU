__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if(idx < N) 
		r_d[idx] = a_d[idx] + 1.f;
	
	
}

extern "C"
cudaError_t launch_actiune_thread(float* a_d, float* b_d,float *r_d,int N,dim3 DIM_GRID, dim3 DIM_BLOCK)
{
	actiune_thread <<<DIM_GRID, DIM_BLOCK>>> (a_d, b_d,r_d,N);
	
	return cudaGetLastError();
}