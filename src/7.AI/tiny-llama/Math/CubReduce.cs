using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Math;

/// <summary>
/// Block-wide reductions backed by <c>cub::BlockReduce</c>. Each method is an
/// <c>[IntrinsicFunction]</c> stub: the Hybridizer transcoder replaces the
/// call site with a hand-written <c>__device__</c> helper defined in
/// <c>intrinsics.cuh</c>, which wraps the matching CUB template instantiation
/// and broadcasts the result to every thread in the block.
///
/// Same plumbing pattern as <see cref="Atomics"/>. The helpers are CUDA-only;
/// the kernels that call them are decorated with
/// <c>[HybridizerIgnore("OMP")]</c>, so the managed bodies are never reached
/// at runtime and just return the input value as a no-op fallback.
/// </summary>
[IntrinsicInclude("intrinsics.cuh")]
internal static class CubReduce
{
    /// <summary>Block-wide sum for blockDim.x = 128. All threads receive the sum.</summary>
    [IntrinsicFunction("cub_block_reduce_sum_128")]
    public static float SumBlock128(float v) => v;

    /// <summary>Block-wide sum for blockDim.x = 64. All threads receive the sum.</summary>
    [IntrinsicFunction("cub_block_reduce_sum_64")]
    public static float SumBlock64(float v) => v;

    /// <summary>Block-wide max for blockDim.x = 64. All threads receive the max.</summary>
    [IntrinsicFunction("cub_block_reduce_max_64")]
    public static float MaxBlock64(float v) => v;
}
