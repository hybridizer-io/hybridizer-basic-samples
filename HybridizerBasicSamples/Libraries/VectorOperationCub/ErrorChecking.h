#pragma once
#ifndef __ERROR_CHECKING__
#define __ERROR_CHECKING__
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>


inline void CheckError(cudaError_t code, const char* file, int line) {
	if (code != cudaSuccess) {
		::fprintf(stdout, "CUDA RUNTIME ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(code));
		abort();
	}
}

#define ERROR_CHECK(code) do { CheckError((code), __FILE__, __LINE__); } while(0)
#define ERROR_PEEK do { CheckError((cudaPeekAtLastError()), __FILE__, __LINE__); } while(0)


#endif // __ERROR_CHECKING__