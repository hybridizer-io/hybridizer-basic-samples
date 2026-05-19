using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Math;

/// <summary>
/// Atomic primitives used by Hybridizer reduction kernels.
///
/// Each method is decorated with <c>[IntrinsicFunction]</c>: the transcoder
/// replaces the call site with the named native intrinsic on CUDA / OMP, and
/// the managed C# body stays as the fallback that runs in the managed
/// dual-path (or in any context where the kernel isn't transcoded).
///
/// CUDA mapping:
///   - <c>atomicAdd(float*, float)</c> — built-in since compute 2.0, no extra
///     header needed.
///   - <c>atomicMax(float*, float)</c> — not built-in; we ship a CAS-loop
///     implementation in <c>intrinsics.cuh</c> (via <c>__int_as_float</c> /
///     <c>__float_as_int</c> bit reinterpretation). Pattern from the
///     6.Advanced / GenericReduction Hybridizer sample.
///
/// OMP behaviour: <c>hybridizer.omp.h</c> defines <c>atomicAdd</c> /
/// <c>atomicMax</c> as plain non-atomic macros — using these intrinsics in an
/// OMP kernel races. Reduction kernels therefore split per flavor and the OMP
/// body uses a sequential loop instead (see [[reference-hybridizer-per-flavor-entrypoints]]).
///
/// We deliberately avoid <see cref="AtomicExpr"/> here — the Hybridizer
/// maintainer flagged its lambda-driven path as buggy; direct intrinsic
/// stubs are the supported pattern.
/// </summary>
[IntrinsicInclude("intrinsics.cuh")]
internal static class Atomics
{
    /// <summary>
    /// Atomic floating-point add. CUDA: native <c>atomicAdd(float*, float)</c>.
    /// Managed fallback: CAS loop via <see cref="Interlocked.CompareExchange(ref float, float, float)"/>.
    /// </summary>
    [IntrinsicFunction("atomicAdd")]
    public static float Add(ref float target, float val)
    {
        while (true)
        {
            float currentValue = target;
            float newValue = currentValue + val;
            float original = Interlocked.CompareExchange(ref target, newValue, currentValue);
            if (original == currentValue)
                return newValue;
        }
    }

    /// <summary>
    /// Atomic floating-point max. CUDA: ships its own <c>atomicMax(float*, float)</c>
    /// via <c>intrinsics.cuh</c>. Managed fallback: CAS loop.
    /// </summary>
    [IntrinsicFunction("atomicMax")]
    public static float Max(ref float target, float val)
    {
        while (true)
        {
            float currentValue = target;
            float newValue = currentValue > val ? currentValue : val;
            float original = Interlocked.CompareExchange(ref target, newValue, currentValue);
            if (original == currentValue)
                return newValue;
        }
    }

    /// <summary>
    /// Atomic integer min. CUDA: built-in <c>atomicMin(int*, int)</c>, no
    /// helper header needed. Used by the argmax tie-breaker (smallest index
    /// among entries equal to the max value).
    /// </summary>
    [IntrinsicFunction("atomicMin")]
    public static int Min(ref int target, int val)
    {
        while (true)
        {
            int currentValue = target;
            int newValue = currentValue < val ? currentValue : val;
            int original = Interlocked.CompareExchange(ref target, newValue, currentValue);
            if (original == currentValue)
                return newValue;
        }
    }
}
