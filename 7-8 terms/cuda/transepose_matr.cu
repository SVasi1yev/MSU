#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024
#define BLOCKSIZE 32
#define GRIDSIZE 32

__global__ void transeposeMatr(float* init_matr, float* res_matr, int n)
{
    __shared__ int array[BLOCKSIZE][BLOCKSIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = bx * BLOCKSIZE + tx;
    int gy = by * BLOCKSIZE + ty;
    array[ty][tx] = init_matr[gy * n + gx];
    __syncthreads();
    res_matr[gx * n + gy] = array[ty][tx];
}

int main()
{
    const int n = SIZE;
    float *init_matr, *res_matr, *test_matr;
    cudaMallocHost((void **) &init_matr, sizeof(float) * n * n);
    cudaMallocHost((void **) &res_matr, sizeof(float) * n * n);
    cudaMallocHost((void **) &test_matr, sizeof(float) * n * n);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            init_matr[i * n + j] = i * n + j;
            test_matr[j * n + i] = i * n + j;
        }
    }

    float *d_init_matr, *d_res_matr;
    cudaMalloc((void **) &d_init_matr, sizeof(float) * n * n);
    cudaMalloc((void **) &d_res_matr, sizeof(float) * n * n);

    cudaMemcpy(d_init_matr, init_matr, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    transeposeMatr<<<dim3(GRIDSIZE, GRIDSIZE), dim3(BLOCKSIZE, BLOCKSIZE), BLOCKSIZE * BLOCKSIZE * sizeof(float)>>>(d_init_matr, d_res_matr, n);

    cudaMemcpy(res_matr, d_res_matr, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(test_matr[i * n + j] != res_matr[i * n + j])
            {
                printf("incorrect!");
                return 0;
            }
        }
    }
    printf("correct!");
    return 0;
}