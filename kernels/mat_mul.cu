#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdlib>
#include <iostream>
#define DLLEXPORT extern "C" __declspec(dllexport)


// C(rowsNum, colsNum) = A(rowsNum, width) * B(width, colsNum)
__global__ void matMul(float *A, float *B, float *C, int rowsNum, int width, int colsNum)
{
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row < rowsNum && col < colsNum ){
    
        float value = 0;
        for(int k = 0; k < width; k++){
            value += A[row * width + k] * B[k * colsNum + col];
        }
    
        C[row * colsNum + col] = value;
    }
}




DLLEXPORT void cudaMatMul (float *mat1, float *mat2, float *mat3, int mat1RowsNum, int mat1ColsNum, int mat2RowsNum, int mat2ColsNum)
{   
  	float *p_mat1, *p_mat2, *p_mat3;

    //// device memory allocation
    cudaMalloc((void**)&p_mat1, mat1RowsNum * mat1ColsNum * sizeof(float));
    cudaMalloc((void**)&p_mat2, mat2RowsNum * mat2ColsNum * sizeof(float));
    cudaMalloc((void**)&p_mat3, mat1RowsNum * mat2ColsNum * sizeof(float));

    cudaMemcpy(p_mat1, mat1, mat1RowsNum * mat1ColsNum * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p_mat2, mat2, mat2RowsNum * mat2ColsNum * sizeof(float), cudaMemcpyHostToDevice);

    
    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);

    int tPB = iProp.maxThreadsPerBlock;
    int bPG = (int((mat1RowsNum + mat2ColsNum)) + tPB - 1) / tPB; //int((mat1RowsNum + mat2ColsNum))/2

    dim3 gridDim(tPB, tPB);
	dim3 blockDim(bPG, bPG);

    // int BLOCK_SIZE = 16;
    // dim3 gridDim(ceil((float)mat1RowsNum / BLOCK_SIZE), ceil((float)mat2ColsNum / BLOCK_SIZE),1);
	// dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE,1);

    // dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
 	// dim3 gridDim((int)ceil(int((mat1RowsNum + mat2ColsNum)/2)/blockDim.x),(int)ceil(int((mat1RowsNum + mat2ColsNum)/2)/blockDim.y));

    // dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 gridDim(mat2ColsNum / blockDim.x, mat1RowsNum / blockDim.y);

    matMul<<<gridDim, blockDim>>>(p_mat1, p_mat2, p_mat3, mat1RowsNum, mat1ColsNum, mat2ColsNum); 

    cudaMemcpy(mat3, p_mat3, mat1RowsNum * mat2ColsNum * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(p_mat1);
    cudaFree(p_mat2);
    cudaFree(p_mat3);
}





