__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N)
{
	// Identificarea pozitiei exacte a thread-ului
	int inOffset = blockDim.x *blockIdx.x;
	int outOffset = blockDim.x * (gridDim.x -1 - blockIdx.x);
	int in = inOffset + threadIdx.x;
	int out = outOffset + (blockDim.x -1 - threadIdx.x);
	r_d[out] = a_d[in];
	
	
	
}

extern "C"
cudaError_t launch_actiune_thread(float* a_d, float* b_d,float *r_d,int N,dim3 DIM_GRID, dim3 DIM_BLOCK)
{
	actiune_thread <<<DIM_GRID, DIM_BLOCK>>> (a_d, b_d,r_d,N);
	
	return cudaGetLastError();
}