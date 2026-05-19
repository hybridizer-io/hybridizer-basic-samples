using System.Collections.Generic;
using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Utils;

/// <summary>
/// One-time materialisation of the 32-byte native struct that the generated
/// CUDA wrappers expect when a parameter is declared as
/// <see cref="FloatResidentArray"/> or <see cref="ResidentArrayGeneric{T}"/>.
///
/// Bypasses Hybridizer's HybRunner-emitted marshaller, which serialises the
/// same struct on every kernel call (cudaMalloc + cudaMemcpy + cudaFree per
/// param per call). For the LlamaTransformer resident path that's the bulk
/// of CUDA API time per the nsys profile — ~6× cudaMemcpy and ~3× cudaMalloc
/// per kernel call, dominating wall-clock.
///
/// Layout matches the C++ side as emitted in <c>hybridizer.includes.cuda.h</c>
/// for FloatResidentArray / ResidentArrayGeneric&lt;T&gt; (both 32 bytes):
///   offset 0..7   : hybridobject union (_typeid or vtable; kernels we use
///                   read only `tab`, so the value isn't load-bearing).
///   offset 8..15  : tab (float* / void*) — the device data pointer.
///   offset 16..23 : _count (int64).
///   offset 24..31 : _status (int) + 4 bytes padding.
///
/// Entries live for the lifetime of the AppDomain; we never free the cached
/// device struct because the underlying resident array also lives that long.
/// </summary>
public static unsafe class ResidentStructCache
{
    private const int NativeStructSize = 32;

    // Keyed by reference identity of the managed resident array.
    private static readonly Dictionary<object, nint> _cache = new();

    public static nint Get(FloatResidentArray arr)
    {
        if (_cache.TryGetValue(arr, out nint cached))
            return cached;
        // The marshaller normally does this in SerializeResidentArray. We're
        // bypassing it, so we must perform the H→D upload ourselves when the
        // array is in the "host-fills-first" state (Q8 weights, RoPE tables).
        // Skipping this is what makes the kernel read garbage and produce
        // all-zero output downstream.
        if (arr.Status == ResidentArrayStatus.DeviceNeedsRefresh)
            arr.RefreshDevice();
        return Materialise(arr, arr.DevicePointer, arr.Count);
    }

    public static nint Get(ResidentArrayGeneric<byte> arr)
    {
        if (_cache.TryGetValue(arr, out nint cached))
            return cached;
        if (arr.Status == ResidentArrayStatus.DeviceNeedsRefresh)
            arr.RefreshDevice();
        return Materialise(arr, arr.DevicePointer, arr.Count);
    }

    public static nint Get(ResidentArrayGeneric<int> arr)
    {
        if (_cache.TryGetValue(arr, out nint cached))
            return cached;
        if (arr.Status == ResidentArrayStatus.DeviceNeedsRefresh)
            arr.RefreshDevice();
        return Materialise(arr, arr.DevicePointer, arr.Count);
    }

    private static nint Materialise(object key, nint tab, long count)
    {
        cuda.ERROR_CHECK(cuda.Malloc(out nint devStruct, (size_t)NativeStructSize));

        // Build the host-side image of the native struct.
        byte* buf = stackalloc byte[NativeStructSize];
        for (int i = 0; i < NativeStructSize; i++) buf[i] = 0;
        *(long*)(buf + 8) = (long)tab;
        *(long*)(buf + 16) = count;

        cuda.ERROR_CHECK(cuda.Memcpy(devStruct, (nint)buf, (size_t)NativeStructSize, cudaMemcpyKind.cudaMemcpyHostToDevice));

        _cache[key] = devStruct;
        return devStruct;
    }
}
