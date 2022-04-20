#include <cuda.h>
#include <cuda_runtime_api.h>
#define DLLEXPORT extern "C" __declspec(dllexport)



__global__ void matAdd(float *A, float *B, float *C, int rowsNum, int colsNum)
{
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = row * colsNum + col;

	if (col < colsNum && row < rowsNum)
		C[idx] = A[idx] + B[idx];
}


DLLEXPORT void cudaMatAdd(float *mat1, float *mat2, float *mat3, int rowsNum, int colsNum)
{   

  	float *p_mat1, *p_mat2, *p_mat3;
	

	//// device memory allocation
	cudaMalloc((void**)&p_mat1, rowsNum * colsNum *sizeof(float));
	cudaMalloc((void**)&p_mat2, rowsNum * colsNum * sizeof(float));
	cudaMalloc((void**)&p_mat3, rowsNum * colsNum * sizeof(float));

	cudaMemcpy(p_mat1, mat1, rowsNum * colsNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(p_mat2, mat2, rowsNum * colsNum * sizeof(float), cudaMemcpyHostToDevice);

  
	int devNo = 0;
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, devNo);

	int tPB = iProp.maxThreadsPerBlock;
	int bPG = (int((rowsNum + colsNum)/2) + tPB - 1) / tPB;
	// printf("Using %d blocks and %d threads\n", bPG, tPB);


	dim3 gridDim(tPB, tPB); 
	dim3 blockDim(bPG, bPG);

	// int BLOCK_SIZE = 512;
    // dim3 gridDim(ceil((float)rowsNum / BLOCK_SIZE), ceil((float)colsNum / BLOCK_SIZE));
	// dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	// dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
 	// dim3 gridDim((int)ceil(int((rowsNum + colsNum)/2)/blockDim.x),(int)ceil(int((rowsNum + colsNum)/2)/blockDim.y));

    matAdd<<<gridDim, blockDim>>>(p_mat1, p_mat2, p_mat3, rowsNum, colsNum); 

    cudaMemcpy(mat3, p_mat3,  rowsNum * colsNum * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(p_mat1);
    cudaFree(p_mat2);
    cudaFree(p_mat3);
}

