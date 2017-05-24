// dllmain.cpp : Defines the entry point for the DLL application.
#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif

#include "hybridizer_cuda_reduction.h"
#include <cuda_runtime.h>

int g_multiprocessor_count = 0;

#ifndef TRUE
#define TRUE 1
#endif

#if defined(_WIN32) || defined(_WIN64)
BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
)
#else
__attribute__((constructor)) int start()
#endif
{
#if defined(_WIN32) || defined(_WIN64)
	switch (ul_reason_for_call)
	{
	case DLL_THREAD_ATTACH:
		break;
	case DLL_PROCESS_ATTACH:
#endif
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		g_multiprocessor_count = prop.multiProcessorCount;

#if defined(_WIN32) || defined(_WIN64)
		break;

	case DLL_THREAD_DETACH:
		break;
	case DLL_PROCESS_DETACH:
		break;
	}
#endif
	return TRUE;
}

