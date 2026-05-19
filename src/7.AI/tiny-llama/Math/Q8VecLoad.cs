using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Math;

/// <summary>
/// Q8 vectorized byte load (iter 7.A.8). One <c>[IntrinsicFunction]</c> stub:
/// the Hybridizer transcoder lowers the call site to a hand-written
/// <c>__device__</c> helper in <c>intrinsics.cuh</c> that issues a single
/// <c>ld.global.v4.u8</c> (one LSU instruction → 4 bytes per thread) instead
/// of 4 individual byte loads.
///
/// Same plumbing pattern as <see cref="CubReduce"/> and <see cref="Atomics"/>:
/// CUDA-only, the calling kernel carries <c>[HybridizerIgnore("OMP")]</c>, so
/// the managed body is the safe fallback that runs only on the
/// Managed/OMP dual-path tests and never on the production CUDA path.
/// </summary>
[IntrinsicInclude("intrinsics.cuh")]
internal static class Q8VecLoad
{
    /// <summary>
    /// Read 4 contiguous bytes starting at <paramref name="idx"/> as a packed
    /// little-endian <c>uint</c> (byte 0 in bits 0..7, byte 1 in bits 8..15, …).
    /// Caller is responsible for ensuring <paramref name="idx"/> is 4-byte
    /// aligned — misaligned access on the CUDA path is undefined behaviour.
    /// </summary>
    [IntrinsicFunction("q8_load_uchar4")]
    public static uint LoadPacked4(ResidentArrayGeneric<byte> arr, int idx)
    {
        uint b0 = arr[idx];
        uint b1 = arr[idx + 1];
        uint b2 = arr[idx + 2];
        uint b3 = arr[idx + 3];
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}
