#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

__global__ void staticReverse(int *d, int n)
{
  __shared__ int array[SIZE];
  int idx = threadIdx.x;
  array[idx] = d[idx];
  __syncthreads();
  d[SIZE - 1 - idx] = array[idx];
}

__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int array[];
  int idx = threadIdx.x;
  array[idx] = d[idx];
  __syncthreads();
  d[SIZE - 1 - idx] = array[idx];
}

int main(void)
{
  const int n = SIZE; // FIX ME TO max possible size
  int* a = new int[n]; 
  int* r = new int[n]; 
  int* d = new int[n]; // FIX ME TO dynamic arrays if neccesary

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 

  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1, SIZE, SIZE * sizeof(int)>>>(d_d, n); // FIX kernel execution params
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1, SIZE, SIZE * sizeof(int)>>>(d_d, n); // FIX kernel executon params
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}
