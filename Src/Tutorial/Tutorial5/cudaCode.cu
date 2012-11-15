__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N)
{
	extern __shared__ float s_data[];
	
	int inOffset = blockDim.x * blockIdx.x;
	int in = inOffset + threadIdx.x;
	
	s_data[blockDim.x - 1 - threadIdx.x] = a_d[in];
	__syncthreads();
	
	int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
	
	int out = outOffset + threadIdx.x;
	r_d[out] = s_data[threadIdx.x];
	
	
}

extern "C"
cudaError_t launch_actiune_thread(float* a_d, float* b_d,float *r_d,int N,dim3 DIM_GRID, dim3 DIM_BLOCK)
{
	actiune_thread <<<DIM_GRID, DIM_BLOCK, 1024>>> (a_d, b_d,r_d,N);
	
	return cudaGetLastError();
}