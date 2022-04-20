#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdlib>
#include <iostream>
#define DLLEXPORT extern "C" __declspec(dllexport)
using namespace std;


// C(rowsNum, colsNum) = A(rowsNum, width) * B(width, colsNum)
void blasMatMul(const float *A, const float *B, float *C, const int rowsNum, const int width, const int colsNum){
    const float alf = 1;
    const float bet = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);
 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colsNum, rowsNum, width ,&alf ,B, colsNum, A, width, &bet, C, colsNum);

    cublasDestroy(handle);
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
   
    blasMatMul(p_mat1, p_mat2, p_mat3,  mat1RowsNum, mat1ColsNum, mat2ColsNum);

    cudaMemcpy(mat3, p_mat3, mat1RowsNum * mat2ColsNum * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(p_mat1);
    cudaFree(p_mat2);
    cudaFree(p_mat3);
}





