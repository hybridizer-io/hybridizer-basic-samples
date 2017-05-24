// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the WIN32PROJECT1_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// WIN32PROJECT1_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.

#if defined(_WIN32) || defined(_WIN64)
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __attribute__((visibility("default")))
#endif

#if __GNUC__
#define hyb_inline __attribute__((always_inline))  inline
#else
#if defined(__clang__) || defined(__GNUC__)
#define hyb_inline inline __attribute__((always_inline))
#else
#define hyb_inline __forceinline inline
#endif
#endif

extern int g_multiprocessor_count;

extern "C" {
	DLL_API double ReduceAdd_double(double* input, int N);
	DLL_API double ScalarProd_double(double* x, double* y, int N);
	DLL_API float ScalarProd_float(float* x, float* y, int N);
}