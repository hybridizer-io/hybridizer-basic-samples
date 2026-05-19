#ifndef __LLAMA_CSHARP_INTRINSICS__
#define __LLAMA_CSHARP_INTRINSICS__

// Use the specific BlockReduce header rather than <cub/cub.cuh>: the umbrella
// header errors out under NVRTC (CCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK),
// which is the path hybridizer takes when JIT-compiling kernels.
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>

// float atomicMax — CUDA does not provide it natively (only integer overloads).
// CAS-loop on the int reinterpretation of the float bit pattern. Pattern from
// the Hybridizer basic-samples / 6.Advanced / GenericReduction sample. Used by
// the [IntrinsicFunction("atomicMax")] stub in LlamaCsharp.Math.Atomics — the
// Hybridizer transcoder emits a call to this function in the generated CUDA
// code. Found via -I. (project root) which is on the nvcc command line.
__device__ __forceinline__ float atomicMax(float* address, float val)
{
    int* addr_as_i = reinterpret_cast<int*>(address);
    int old = *addr_as_i;
    int assumed;

    while (true) {
        assumed = old;
        float old_val = __int_as_float(assumed);

        if (old_val >= val)
            break;

        int new_bits = __float_as_int(val);
        old = atomicCAS(addr_as_i, assumed, new_bits);

        if (old == assumed)
            break;
    }

    return __int_as_float(old);
}

// ---------------------------------------------------------------------------
// CUB BlockReduce wrappers (iter 7.A.6).
//
// Hybridizer can't emit `__shared__ typename cub::BlockReduce<...>::TempStorage`
// directly, so each (BLOCK_DIM, op) pair gets a hand-written `__device__`
// helper here. The C# side calls these via [IntrinsicFunction] stubs declared
// in Math/CubReduce.cs.
//
// All helpers broadcast the reduced value to every thread (CUB returns it only
// on thread 0). The TempStorage and the broadcast slot are declared `__shared__`
// inside the helper, which under CUDA semantics is a single per-block static
// allocation — reuse across multiple calls in the same kernel is safe as long
// as each call completes (BlockReduce internally syncthreads; the explicit
// __syncthreads below covers the broadcast write).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float cub_block_reduce_sum_128(float v)
{
    typedef cub::BlockReduce<float, 128> BR;
    __shared__ typename BR::TempStorage tmp;
    __shared__ float bcast;
    float r = BR(tmp).Sum(v);
    if (threadIdx.x == 0) bcast = r;
    __syncthreads();
    return bcast;
}

__device__ __forceinline__ float cub_block_reduce_sum_64(float v)
{
    typedef cub::BlockReduce<float, 64> BR;
    __shared__ typename BR::TempStorage tmp;
    __shared__ float bcast;
    float r = BR(tmp).Sum(v);
    if (threadIdx.x == 0) bcast = r;
    __syncthreads();
    return bcast;
}

__device__ __forceinline__ float cub_block_reduce_max_64(float v)
{
    typedef cub::BlockReduce<float, 64> BR;
    __shared__ typename BR::TempStorage tmp;
    __shared__ float bcast;
    float r = BR(tmp).Reduce(v, cuda::maximum<>{});
    if (threadIdx.x == 0) bcast = r;
    __syncthreads();
    return bcast;
}

// ---------------------------------------------------------------------------
// Q8 vectorized byte load (iter 7.A.8).
//
// Each thread reads 4 contiguous Q8 bytes per inner-loop iteration. PTX
// emits a single ld.global.v4.u8 instruction instead of 4 separate byte
// loads — cuts LSU instruction count 4× on the matvec hot path. C# side:
// [IntrinsicFunction("q8_load_uchar4")] in Math/Q8VecLoad.cs.
//
// We take a `const void*` rather than a `ResidentArrayGeneric<unsigned char>*`
// because intrinsics.cuh is included before hybridizer.includes.cuda.h in
// the generated translation unit, so the Hybridizer struct isn't visible
// here. The data pointer (`tab`) lives at offset 8 of the 32-byte resident
// struct (vtable/typeid at 0..7, tab at 8..15, _count at 16..23, _status
// at 24..27). The Hybridizer-generated call site implicitly converts the
// typed array pointer to `const void*`.
//
// Caller ensures the resulting (data + idx) address is 4-byte aligned —
// TinyLlama row offsets are 32-byte aligned and the kernel steps idx by 4
// from a 4-byte base.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int q8_load_uchar4(const void* arr, int idx)
{
    const unsigned char* data =
        *reinterpret_cast<const unsigned char* const*>(
            static_cast<const char*>(arr) + 8);
    uchar4 v = *reinterpret_cast<const uchar4*>(data + idx);
    return  (unsigned int)v.x
         | ((unsigned int)v.y << 8)
         | ((unsigned int)v.z << 16)
         | ((unsigned int)v.w << 24);
}

// ---------------------------------------------------------------------------
// Float vectorized load (iter 7.A.9).
//
// Companion to q8_load_uchar4: reads 4 contiguous floats as one
// ld.global.v4.f32 (16 bytes per thread per instruction) and writes them
// to four out-pointers. Used by MatVecMulCoopRowCubVec4 to vectorize the
// activation-side reads that previously emitted 4× LD.E per inner-loop
// iteration.
//
// Same const-void* + offset-8 trick as q8_load_uchar4 — intrinsics.cuh
// is included before the Hybridizer FloatResidentArray struct definition
// in the generated TU, so we can't reference the typed struct here. The
// data pointer lives at offset 8 of the 32-byte resident struct.
// Caller ensures idx is a multiple of 4 (the kernel steps tid*4 by
// blockDim*4, so always satisfied).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void f32_load_v4(const void* arr, int idx,
                                             float* x, float* y, float* z, float* w)
{
    const float* data =
        *reinterpret_cast<const float* const*>(
            static_cast<const char*>(arr) + 8);
    float4 v = *reinterpret_cast<const float4*>(data + idx);
    *x = v.x;
    *y = v.y;
    *z = v.z;
    *w = v.w;
}

// ---------------------------------------------------------------------------
// CUDA-graph wrappers (iter 7.A.7.b).
//
// Hybridizer.Runtime.CUDAImports doesn't expose the cudaGraph_* / capture APIs
// we need to collapse the ~378 per-token kernel launches into a single
// cudaGraphLaunch. We host-side wrap them here as plain extern "C" symbols and
// dlopen them from C# via NativeLibrary.GetExport, same pattern as the
// _ExternCWrapperStream_CUDA kernel wrappers.
//
// All return the integer cudaError_t (0 = success). Pointer outputs use the
// idiomatic out-param pattern. The chosen capture mode is Relaxed: legacy
// default-stream calls anywhere on the device (e.g. host-side cuda.Memcpy in
// the unrelated drain timer) won't abort the capture.
// ---------------------------------------------------------------------------

// NVRTC (used by hybridizer for JIT-compiled kernels) doesn't see the
// cudaStream*/cudaGraph* host APIs and rejects unannotated host functions.
// These wrappers are only consumed by the static nvcc -> shared-library path
// (see CompileCUDA in Directory.Build.targets), so skip them under NVRTC.
#ifndef __CUDACC_RTC__

// DLL_PUBLIC is defined for us by hybridizer.wrappers.cu in that TU, but
// host_wrappers.cu (the static-build TU that pulls these wrappers into the
// satellite DLL — see Directory.Build.targets CompileCUDA target) only
// #includes this header, so define it locally if not already set.
#ifndef DLL_PUBLIC
  #if defined _WIN32 || defined __CYGWIN__
    #define DLL_PUBLIC __declspec(dllexport)
  #else
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
  #endif
#endif

extern "C" {

DLL_PUBLIC int llama_stream_create(cudaStream_t* streamOut)
{
    return (int)cudaStreamCreate(streamOut);
}

DLL_PUBLIC int llama_stream_destroy(cudaStream_t stream)
{
    return (int)cudaStreamDestroy(stream);
}

DLL_PUBLIC int llama_stream_synchronize(cudaStream_t stream)
{
    return (int)cudaStreamSynchronize(stream);
}

DLL_PUBLIC int llama_graph_begin_capture(cudaStream_t stream)
{
    return (int)cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
}

DLL_PUBLIC int llama_graph_end_capture(cudaStream_t stream, cudaGraph_t* graphOut)
{
    return (int)cudaStreamEndCapture(stream, graphOut);
}

DLL_PUBLIC int llama_graph_instantiate(cudaGraphExec_t* execOut, cudaGraph_t graph)
{
    return (int)cudaGraphInstantiate(execOut, graph, nullptr, nullptr, 0);
}

DLL_PUBLIC int llama_graph_launch(cudaGraphExec_t exec, cudaStream_t stream)
{
    return (int)cudaGraphLaunch(exec, stream);
}

DLL_PUBLIC int llama_graph_destroy(cudaGraph_t graph)
{
    return (int)cudaGraphDestroy(graph);
}

DLL_PUBLIC int llama_graph_exec_destroy(cudaGraphExec_t exec)
{
    return (int)cudaGraphExecDestroy(exec);
}

} // extern "C"

#endif // __CUDACC_RTC__

#endif
