#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <map>
#include "hybridizer_cuda_reduction.h"

double* g_reduce_double_device;
float* g_reduce_float_device;

__device__ float __fAtomicAdd(float *p, float val);
namespace hybridizer {
#if defined(__CUDA_ARCH__)
#include <device_functions.h>

#if (__CUDA_ARCH__ < 600)
	__forceinline __device__ double atomicAdd(double* address, double val)
	{
		unsigned long long int* address_as_ull =
			(unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));
		} while (assumed != old);
		return __longlong_as_double(old);
	}
#endif
#endif


	__forceinline __device__ double warpReduceSum(double val) {
#pragma unroll
		for (int offset = warpSize / 2; offset > 0; offset /= 2) {
			val += __shfl_down(val, offset);
		}

		return val;
	}

	__forceinline __device__ float warpReduceSum(float val) {
#pragma unroll
		for (int offset = warpSize / 2; offset > 0; offset /= 2) {
			val += __shfl_down(val, offset);
		}

		return val;
	}

	__forceinline __device__ double blockReduceSum(double val) {
		static __shared__ double shared[32];
		int lane = threadIdx.x % 32;
		int wid = threadIdx.x / 32;
		val = warpReduceSum(val);
		if (lane == 0) shared[wid] = val;
		__syncthreads();
		val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
		if (wid == 0) {
			val = warpReduceSum(val);
		}

		return val;
	}


	__forceinline __device__ float blockReduceSum(float val) {
		static __shared__ double shared[32];
		int lane = threadIdx.x % 32;
		int wid = threadIdx.x / 32;
		val = warpReduceSum(val);
		if (lane == 0) shared[wid] = val;
		__syncthreads();
		val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
		if (wid == 0) {
			val = warpReduceSum(val);
		}

		return val;
	}

	__global__ void deviceReduceAdd(double *in, double* out, int N) {
		double sum = 0.0;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			sum += in[i];
		}

		sum = blockReduceSum(sum);
		if (threadIdx.x == 0) {
			atomicAdd(out, sum);
		}
	}

	__global__ void deviceScalarProd(double *x, double* y, double* out, int N) {
		double sum = 0.0;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			sum += x[i] * y[i];
		}

		sum = blockReduceSum(sum);
		if (threadIdx.x == 0) {
			atomicAdd(out, sum);
		}
	}

	__global__ void deviceScalarProd(float *x, float* y, float* out, int N) {
		float sum = 0.0F;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			sum += x[i] * y[i];
		}

		sum = blockReduceSum(sum);
		if (threadIdx.x == 0) {
			__fAtomicAdd(out, sum);
		}
	}
}
extern "C" {
	hyb_inline void init() {
		if (g_reduce_double_device == NULL) {
			cudaMalloc(&g_reduce_double_device, sizeof(double));
			cudaMalloc(&g_reduce_float_device, sizeof(float));
		}
	}

	double ReduceAdd_double(double* x, int N) {
		init();
		double result = 0.0;
		cudaMemset(g_reduce_double_device, 0, sizeof(double));
		hybridizer::deviceReduceAdd << < 16 * g_multiprocessor_count, 128 >> > (x, g_reduce_double_device, N);
		cudaMemcpy(&result, g_reduce_double_device, sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		return result;
	}

	double ScalarProd_double(double* x, double* y, int N) {
		init();
		double result = 0.0;
		cudaMemset(g_reduce_double_device, 0, sizeof(double));
		hybridizer::deviceScalarProd << < 16 * g_multiprocessor_count, 128 >> > (x, y, g_reduce_double_device, N);
		cudaMemcpy(&result, g_reduce_double_device, sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		return result;
	}

	float ScalarProd_float(float* x, float* y, int N) {
		init();
		float result = 0.0F;
		cudaMemset(g_reduce_float_device, 0, sizeof(float));
		hybridizer::deviceScalarProd << < 16 * g_multiprocessor_count, 128 >> > (x, y, g_reduce_float_device, N);
		cudaMemcpy(&result, g_reduce_float_device, sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		return result;
	}
}