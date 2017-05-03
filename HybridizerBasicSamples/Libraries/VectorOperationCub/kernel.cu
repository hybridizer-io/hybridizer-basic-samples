
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "VectorOperationCuda.h"
#include "ErrorChecking.h"
#include <stdio.h>


template<typename T>
__global__ void set(T* a, int N, T val) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x)
		a[i] = val;
}

int main()
{
	typedef double real;
	const int N = 100000;
	hybridizer::complex<real> * a, *b;

	ERROR_CHECK(cudaMalloc(&a, N * sizeof(hybridizer::complex<real>)));
	ERROR_CHECK(cudaMalloc(&b, N * sizeof(hybridizer::complex<real>)));

	set<> << <20, 256 >> > (a, N, { 1.0, 1.0 });
	set<> << <20, 256 >> > (b, N, { 1.0, -1.0 });
	ERROR_CHECK(cudaDeviceSynchronize());

	printf("first\n");
	hybridizer::complex<real> sum = hybridizer::ScalarProduct(a, b, N);

	ERROR_CHECK(cudaDeviceSynchronize());

	printf("second\n");
	sum = hybridizer::ScalarProduct(a, b, N);

	::printf("%lf , %lf\n", sum.data.x, sum.data.y);
}