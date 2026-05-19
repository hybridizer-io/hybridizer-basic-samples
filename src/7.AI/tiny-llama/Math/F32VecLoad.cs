using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Math;

/// <summary>
/// Float vectorized load (iter 7.A.9). Same pattern as
/// <see cref="Q8VecLoad"/>: one <c>[IntrinsicFunction]</c> stub that the
/// Hybridizer transcoder lowers to a hand-written <c>__device__</c> helper
/// in <c>intrinsics.cuh</c>. The helper issues a single
/// <c>ld.global.v4.f32</c> (one LSU instruction → 4 floats per thread)
/// instead of 4 individual <c>LD.E</c> 32-bit loads.
///
/// Used by <see cref="Q8Kernels.MatVecMulCoopRowCubVec4"/> to vectorize the
/// activation-side reads alongside the Q8 weight-side reads, pushing the
/// matvec from ~75 % DRAM throughput toward the 90 % range per the Nsight
/// Compute report (3.ncu-rep, 2026-05-18 PM).
/// </summary>
[IntrinsicInclude("intrinsics.cuh")]
internal static class F32VecLoad
{
    /// <summary>
    /// Read 4 contiguous floats starting at <paramref name="idx"/> into
    /// <paramref name="x"/>..<paramref name="w"/>. Caller is responsible for
    /// 16-byte alignment (idx must be a multiple of 4 floats); the kernels
    /// stepping <c>tid * 4 + k * blockDim * 4</c> satisfy this trivially.
    /// </summary>
    [IntrinsicFunction("f32_load_v4")]
    public static void LoadVec4(FloatResidentArray arr, int idx,
                                 out float x, out float y, out float z, out float w)
    {
        x = arr[idx];
        y = arr[idx + 1];
        z = arr[idx + 2];
        w = arr[idx + 3];
    }
}
