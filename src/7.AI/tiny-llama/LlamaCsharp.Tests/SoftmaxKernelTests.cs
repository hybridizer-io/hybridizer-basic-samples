using LlamaCsharp.Math;
using LlamaCsharp.Utils;
using Xunit;
using Xunit.Abstractions;

namespace LlamaCsharp.Tests;

/// <summary>
/// Step 6 — Softmax port via <see cref="GpuBackend.Softmax"/>. Verifies that
/// the three-phase (find-max → exp+sum → normalize) parallel-atomic softmax
/// agrees with the managed sequential reference for the typical input shapes
/// that the LM-head / Attention path will hit.
/// </summary>
[Trait("Category", "Hybridizer")]
public class SoftmaxKernelTests
{
    private readonly ITestOutputHelper _output;
    public SoftmaxKernelTests(ITestOutputHelper output) { _output = output; }

    public static IEnumerable<object[]> Sizes => new[]
    {
        new object[] { 8 },
        new object[] { 64 },
        new object[] { 2048 },   // ≈ TinyLlama context length
        new object[] { 32000 },  // ≈ TinyLlama vocab size (LM head softmax)
    };

    private static float[] BuildInput(int size, int seed)
    {
        var rng = new Random(seed);
        var x = new float[size];
        for (int i = 0; i < size; i++)
            x[i] = (float)(rng.NextDouble() * 20.0 - 10.0); // [-10, 10]
        return x;
    }

    private static float[] Reference(float[] x)
    {
        var copy = (float[])x.Clone();
        ParallelMath.Softmax(copy, copy.Length);
        return copy;
    }

    private static void AssertClose(float[] expected, float[] actual, float rtol, float atol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = atol + rtol * MathF.Abs(expected[i]);
            Assert.True(diff <= tol,
                $"index {i}: expected {expected[i]}, actual {actual[i]}, diff {diff} > {tol}");
        }
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void Managed_MatchesReference(int size)
    {
        var input = BuildInput(size, seed: size * 13);
        var reference = Reference(input);
        var actual = (float[])input.Clone();

        GpuBackend.Activate(ComputeBackend.Managed);
        try
        {
            GpuBackend.Softmax(actual, size);
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }

        AssertClose(reference, actual, rtol: 1e-5f, atol: 1e-7f);
    }

    [SkippableTheory]
    [MemberData(nameof(Sizes))]
    public void Cuda_MatchesReference(int size)
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built");

        var input = BuildInput(size, seed: size * 13);
        var reference = Reference(input);
        var actual = (float[])input.Clone();

        GpuBackend.Activate(ComputeBackend.Cuda);
        try
        {
            GpuBackend.Softmax(actual, size);
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }

        _output.WriteLine($"CUDA softmax({size}) ran");
        AssertClose(reference, actual, rtol: 5e-5f, atol: 1e-6f);
    }

    [SkippableTheory]
    [MemberData(nameof(Sizes))]
    public void Omp_MatchesReference(int size)
    {
        Skip.IfNot(SatelliteLoader.OmpAvailable(), "OMP satellite not built");

        var input = BuildInput(size, seed: size * 13);
        var reference = Reference(input);
        var actual = (float[])input.Clone();

        GpuBackend.Activate(ComputeBackend.Omp);
        try
        {
            GpuBackend.Softmax(actual, size);
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }

        _output.WriteLine($"OMP softmax({size}) ran");
        AssertClose(reference, actual, rtol: 5e-5f, atol: 1e-6f);
    }
}
