using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Math;

/// <summary>
/// Int-typed device kernels used by the GPU sampling path
/// (<see cref="LlamaCsharp.Model.LlamaTransformer.ForwardArgmaxGpu"/> and
/// successors). All CUDA-only — the host buffer forward path doesn't go
/// through these.
/// </summary>
public class IntKernels
{
    /// <summary>
    /// Smoke kernel for the <see cref="ResidentArrayGeneric{T}"/>-of-int round
    /// trip: writes <paramref name="value"/> into <c>dst[0]</c>. Single-element
    /// touch; verifies that the bypass struct cache + the generated wrapper
    /// understand the int element type before any real kernel reads/writes
    /// device int memory.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void WriteOneInt(ResidentArrayGeneric<int> dst, int value)
    {
        Parallel.For(0, 1, _ => { dst[0] = value; });
    }

    /// <summary>
    /// Append <paramref name="src"/>[0] to a fixed-size ring buffer:
    /// <c>ring[counter[0] % ringSize] = src[0]; counter[0]++</c>. Single-thread
    /// body via <c>Parallel.For(0, 1, ...)</c> — Hybridizer maps that to a
    /// launch where only the (0,0) thread executes, so no atomics needed.
    /// Used by the deferred-print path: after the per-token argmax writes the
    /// new id into <c>_nextTokenIdResident</c>, this kernel appends it to the
    /// ring + bumps the counter without any host roundtrip.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void AppendDeviceInt(
        ResidentArrayGeneric<int> ring,
        ResidentArrayGeneric<int> counter,
        ResidentArrayGeneric<int> src,
        int ringSize)
    {
        Parallel.For(0, 1, _ =>
        {
            int p = counter[0];
            ring[p % ringSize] = src[0];
            counter[0] = p + 1;
        });
    }

    /// <summary>
    /// Increment a one-element device int slot in place: <c>slot[0]++</c>.
    /// Used by the deferred-decode path to bump the position counter
    /// device-side so the per-token forward can be CUDA-graph-captured
    /// without re-uploading the host int every step (iter 7.A.7.a).
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void BumpDeviceInt(ResidentArrayGeneric<int> slot)
    {
        Parallel.For(0, 1, _ =>
        {
            slot[0] = slot[0] + 1;
        });
    }
}
