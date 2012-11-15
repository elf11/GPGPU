__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N);

// Kernelul ce se executa pe device-ul CUDA 
__global__ void actiune_thread(float* a_d, float* b_d,float *r_d,int N)
{

	/*
	int i = blockIdx.x*32 + threadIdx.x;
	int j = blockIdx.y;
	float sum = 0.0f;
	
	for (int k = 0; k < N; ++k) 
		sum += a_d[i*N+k] * b_d[k*N+j];
	
	r_d[i*N+j] = sum;
	*/

	// VERSION 2.0
	
	int tx = threadIdx.x; 
	int i = blockIdx.x*32 + tx; 
	int j = blockIdx.y;
	__shared__ float cb[32];
	float sum = 0.0f;
	
	for (int ks = 0; ks < N; ks += 32) {
		
		cb[tx] = a_d[ks+tx+N*j];
		for (int k = ks; k < ks+32; ++k) 
			sum += b_d[i+N*k] * cb[k-ks];
	
	}
	
	r_d[i+N*j] = sum;
	

	//VERSION 3.0
	/*
	int tx = threadIdx.x; 
	int i = blockIdx.x*64 + tx; 
	int j = blockIdx.y;
	__shared__ float cb[32];
	float sum0 = 0.0f, 
		sum1 = 0.0f;

	for (int ks = 0; ks < N; ks += 32) {

		cb[tx] = a_d[ks+tx+N*j];
		__syncthreads();
		for (int k = ks; k < ks+32; ++k) { 
			sum0 += b_d[i+N*k] * cb[k-ks]; 
			sum1 += b_d[i+32+N*k] * cb[k-ks]; }
		__syncthreads();

	}

	r_d[i+N*j] = sum0;
	r_d[i+32+N*j] = sum1;
	*/
}

extern "C"
cudaError_t launch_actiune_thread(float* a_d, float* b_d,float *r_d,int N,dim3 DIM_GRID, dim3 DIM_BLOCK)
{
	actiune_thread <<<DIM_GRID, DIM_BLOCK>>> (a_d, b_d,r_d,N);
	
	return cudaGetLastError();
}