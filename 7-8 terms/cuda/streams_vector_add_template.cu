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

void sample_vec_add(int size = 1048576)
{
    printf("sample\n");

    int n = size;
    
    int nBytes = n*sizeof(float);
    
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
    
    printf("Allocating device memory on host..\n");
    
    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));
    
    //printf("Copying to device..\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    
    //printf("Doing GPU Vector add\n");
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);

    cudaMemcpy(c, c_d, nBytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();

    for(int i = 0; i < 20; i++)
    {
        printf("%f\n", c[i]);
    }
    for(int i = 0; i < n; i++)
    {
        if(c[i] != test[i])
        {
            printf("INCORRECT!\n");
            break;
        }
    }
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(test);
}

void streams_vec_add(int size = 1048576)
{
    printf("streams\n");

    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    float *test;
    
    cudaMallocHost(&a, nBytes);
    cudaMallocHost(&b, nBytes);
    cudaMallocHost(&c, nBytes);
    cudaMallocHost(&test, nBytes);
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)n/(2 * block.x) + 1);
    
    for(int i=0;i<n;i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
        test[i] = a[i] + b[i];
    }
    
    printf("Allocating device memory on host..\n");
    
    cudaMalloc((void **)&a_d, nBytes);
    cudaMalloc((void **)&b_d, nBytes);
    cudaMalloc((void **)&c_d, nBytes);

    float *a_offset = a + n / 2, *b_offset = b + n / 2, *c_offset = c + n / 2;
    float *a_d_offset = a_d + n / 2, *b_d_offset = b_d + n / 2, *c_d_offset = c_d + n / 2;

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // printf("Copying to device..\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudaMemcpyAsync(a_d, a, nBytes / 2, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(b_d, b, nBytes / 2, cudaMemcpyHostToDevice, stream1);

    cudaMemcpyAsync(a_d_offset, a_offset, nBytes / 2, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(b_d_offset, b_offset, nBytes / 2, cudaMemcpyHostToDevice, stream2);
    
    // printf("Doing GPU Vector add\n");
    vectorAddGPU<<<grid, block, 0, stream1>>>(a_d, b_d, c_d, n / 2);
    vectorAddGPU<<<grid, block, 0, stream2>>>(a_d_offset, b_d_offset, c_d_offset, n / 2);

    cudaMemcpyAsync(c, c_d, nBytes / 2, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(c_offset, c_d_offset, nBytes / 2, cudaMemcpyDeviceToHost, stream2);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();

    for(int i = 0; i < 20; i++)
    {
        printf("%f\n", c[i]);
    }
    for(int i = 0; i < n; i++)
    {
        if(c[i] != test[i])
        {
            printf("INCORRECT!\n");
            break;
        }
    }
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(test);
}


int main(int argc, char **argv)
{
    sample_vec_add(atoi(argv[1]));
    streams_vec_add(atoi(argv[1]));

    return 0;
}
