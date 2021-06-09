//
//  main.cpp
//  
//
//  Created by Elijah Afanasiev on 25.09.2018.
//
//

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void unified_samle(int size = 1048576)
{
    printf("infied_sample\n");

    int n = size;

    int nBytes = n * sizeof(float);
    float *a, *b;
    float *c;

    cudaMallocManaged(&a, nBytes);
    cudaMallocManaged(&b, nBytes);
    cudaMallocManaged(&c, nBytes);

    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    vectorAddGPU<<<grid, block>>>(a, b, c, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

void pinned_samle(int size = 1048576)
{
    printf("pinned_sample\n");

    int n = size;

    int nBytes = n * sizeof(float);
    float *a, *b;
    float *c;

    cudaMallocHost(&a, nBytes);
    cudaMallocHost(&b, nBytes);
    cudaMallocHost(&c, nBytes);

    float *a_d, *b_d;
    float *c_d;

    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    // printf("Allocating device memory on host..\n");
    
    cudaMalloc((void **)&a_d, nBytes);
    cudaMalloc((void **)&b_d, nBytes);
    cudaMalloc((void **)&c_d, nBytes);

    // printf("Copying to device..\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudaMemcpy(a_d, a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, nBytes, cudaMemcpyHostToDevice);
    
    // printf("Doing GPU Vector add\n");
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

void usual_sample(int size = 1048576)
{
    printf("usual sample\n");

    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results

    a = (float *)malloc(nBytes);
    b = (float *)malloc(nBytes);
    c = (float *)malloc(nBytes);
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    
    // printf("Allocating device memory on host..\n");

    cudaMalloc((void **)&a_d, nBytes);
    cudaMalloc((void **)&b_d, nBytes);
    cudaMalloc((void **)&c_d, nBytes);
    
    // printf("Copying to device..\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    cudaMemcpy(a_d, a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, nBytes, cudaMemcpyHostToDevice);
    
    // printf("Doing GPU Vector add\n");
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a);
    free(b);
    free(c);
}


int main(int argc, char **argv)
{
    usual_sample(atoi(argv[1]));
    pinned_samle(atoi(argv[1]));
    unified_samle(atoi(argv[1]));
    
    return 0;
}
