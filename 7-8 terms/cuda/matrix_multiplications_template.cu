#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cublas_v2.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void gpu_matrix_mult_global(float *a, float *b, float *c, int m, int n, int k)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0;
    int ia = n * (BLOCK_SIZE * by + ty);
    int ib = BLOCK_SIZE * bx + tx;
    for(int i = 0; i < n; i++)
    {
        sum += a[ia + i] * b[ib + i * n];
    }
    c[ia + ib] = sum;
}

__global__ void gpu_square_matrix_mult(float *a, float *b, float *c, int n)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1, bBegin = BLOCK_SIZE * bx;
    int aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;

    float sum = 0;

    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    for(int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[tx][ty] = a[ia + n * ty + tx];
        bs[tx][ty] = b[ib + n * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += as[k][ty] * bs[tx][k];
        }
        __syncthreads();
    }
    c[aBegin + bBegin + ty * n + tx] = sum;
}

void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    m = atoi(argv[1]);
    n = atoi(argv[1]);
    k = atoi(argv[1]);
    
    // allocate memory in host RAM, h_cc is used to store CPU result
    float *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(float)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(float)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(float)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(float)*m*k);
    
    // random initialize matrix A
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_a[i * n + j] = rand() % 1024;
        }
    }
    
    // random initialize matrix B
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            h_b[i * k + j] = rand() % 1024;
        }
    }
    
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    
    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start to count execution time of GPU version
    // Allocate memory space on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*m*n);
    cudaMalloc((void **) &d_b, sizeof(float)*n*k);
    cudaMalloc((void **) &d_c, sizeof(float)*m*k);
    
    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    
    int block_num = n / BLOCK_SIZE;
    
    cudaEventRecord(start, 0);
    
    if(atoi(argv[2]) == 1)
    {
        gpu_square_matrix_mult<<<dim3(block_num, block_num), dim3(BLOCK_SIZE, BLOCK_SIZE), 32 * 32 * sizeof(float)>>>(d_a, d_b, d_c, n);
    }
    else if(atoi(argv[2]) == 2)
    {
        gpu_matrix_mult_global<<<dim3(block_num, block_num), dim3(BLOCK_SIZE, BLOCK_SIZE), 32 * 32 * sizeof(float)>>>(d_a, d_b, d_c, m, n, k);
    }
    
    // Transefr results from device to host
    cudaDeviceSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(h_c, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
    
    // start the CPU version
    cudaEventRecord(start, 0);
    
    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);
    
    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if(h_cc[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
    }

    double flop = 2 * n * n;
    flop /= gpu_elapsed_time_ms;
    flop *= n;
    flop /= 1e6;
    
    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
        printf("permormance = %f\n", flop);
    }
    else
    {
        printf("incorrect results\n");
    }
    
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    
    return 0;
}
