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
#include <omp.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

int deviceCount;

__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void sample_vec_add(int size = 1048576)
{
    printf("Sample\n");

    int n = size;
    
    int nBytes = n*sizeof(int);
    
    float *a, *b;  // host data
    float *c;  // results
    float *test;
    
    cudaMallocHost(&a, nBytes);
    cudaMallocHost(&b, nBytes);
    cudaMallocHost(&c, nBytes);
    cudaMallocHost(&test, nBytes);
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    for(int i=0;i<n;i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
        test[i] = a[i] + b[i];
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double time_ = omp_get_wtime();
    cudaEventRecord(start);
    
    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));
    
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);

    cudaMemcpy(c, c_d, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    time_ = omp_get_wtime() - time_;
    time_ *= 1000;
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda time: %f ms\nomp time: %f ms\n", milliseconds, time_);
    
    cudaThreadSynchronize();
    
    for(int i = 0; i < n; i++)
    {
        if(c[i] != test[i])
        {
            printf("INCORRECT\n");
            break;
        }
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(test);
}

void streams_vec_add(int size = 1048576)
{
    printf("Streams\n");
    cudaSetDevice(1);
    float* temp;
    cudaMallocHost(&temp, sizeof(float));
    cudaFree(temp);
    cudaSetDevice(0);

    int n = size;
    int nBytes = n * sizeof(float);

    float *a, *b;
    float *c;
    float *test;
    
    cudaMallocHost(&a, nBytes);
    cudaMallocHost(&b, nBytes);
    cudaMallocHost(&c, nBytes);
    cudaMallocHost(&test, nBytes);

    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x/deviceCount));
    
    for(int i=0;i<n;i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
        test[i] = a[i] + b[i];
    }

    omp_set_num_threads(deviceCount);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    double time_ = omp_get_wtime();

    #pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        cudaSetDevice(thread_num);

        float *a_d, *b_d;
        float *c_d;
        
        cudaMalloc((void **)&a_d, nBytes / deviceCount);
        cudaMalloc((void **)&b_d, nBytes / deviceCount);
        cudaMalloc((void **)&c_d, nBytes / deviceCount);

        cudaMemcpy(a_d, a + thread_num * n / deviceCount, nBytes / deviceCount, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b + thread_num * n / deviceCount, nBytes / deviceCount, cudaMemcpyHostToDevice);

        vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);

        cudaMemcpy(c + thread_num * n / deviceCount, c_d, nBytes / deviceCount, cudaMemcpyDeviceToHost);

        cudaFree(a_d);
        cudaFree(b_d);
        cudaFree(c_d);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    time_ = omp_get_wtime() - time_;
    time_ *= 1000;
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda time: %f ms\nomp time: %f ms\n", milliseconds, time_);

    cudaThreadSynchronize();

    for(int i = 0; i < n; i++)
    {
        if(c[i] != test[i])
        {
            printf("INCORRECT!\n");
            break;
        }
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(test);
}


int main(int argc, char **argv)
{

    cudaGetDeviceCount(&deviceCount);
    sample_vec_add(atoi(argv[1]));
    streams_vec_add(atoi(argv[1]));

    return 0;
}
