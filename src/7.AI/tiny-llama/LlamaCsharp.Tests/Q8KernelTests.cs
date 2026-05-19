using Hybridizer.Runtime.CUDAImports;
using LlamaCsharp.Math;
using LlamaCsharp.Utils;
using Xunit;
using Xunit.Abstractions;

namespace LlamaCsharp.Tests;

/// <summary>
/// Step-5 first-real-kernel port: Q8_0 row-matvec on both Hybridizer flavors.
/// Compares each flavor against the managed dequantize-then-MatVecMul reference.
/// </summary>
[Trait("Category", "Hybridizer")]
public class Q8KernelTests
{
    private readonly ITestOutputHelper _output;
    public Q8KernelTests(ITestOutputHelper output) { _output = output; }

    // Shapes that exercise both a TinyLlama-ish embed_dim and a Wide-ish hidden_dim.
    public static IEnumerable<object[]> Shapes => new[]
    {
        new object[] { 16, 64 },     // small smoke
        new object[] { 64, 128 },    // small
        new object[] { 256, 2048 },  // ≈ embed_dim × hidden col-stride
        new object[] { 32, 5632 },   // matches TinyLlama hidden_dim col-stride
    };

    private static float[] LinSpace(int n, float start, float step, int seed)
    {
        var rng = new Random(seed);
        var arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = start + i * step + (float)(rng.NextDouble() - 0.5);
        return arr;
    }

    private static (byte[] values, float[] scales, float[] vec, float[] reference) BuildInputs(int rows, int cols)
    {
        var rawMat = LinSpace(rows * cols, -0.5f, 0.0009f, seed: rows * 31 + cols);
        var vec = LinSpace(cols, 0.4f, -0.003f, seed: rows + cols * 7);
        byte[] q8 = Q8Helpers.QuantizeMatrix(rawMat, rows, cols);
        var (values, scales) = Q8Helpers.SplitMatrix(q8, rows, cols);
        float[] dequantized = Q8Helpers.DequantizeMatrix(q8, rows, cols);
        var reference = new float[rows];
        ParallelMath.MatVecMul(reference, dequantized, vec, rows, cols);
        return (values, scales, vec, reference);
    }

    private static void AssertClose(float[] expected, float[] actual, float rtol, float atol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = atol + rtol * MathF.Abs(expected[i]);
            Assert.True(diff <= tol, $"index {i}: expected {expected[i]}, actual {actual[i]}, diff {diff} > {tol}");
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Managed_DualPathMatchesDequantizedReference(int rows, int cols)
    {
        var (values, scales, vec, reference) = BuildInputs(rows, cols);
        int blocksPerRow = cols / Q8Helpers.BlockElements;
        var result = new float[rows];
        Q8Kernels.MatVecMul(result, values, scales, vec, rows, blocksPerRow);
        AssertClose(reference, result, rtol: 5e-5f, atol: 1e-4f);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_MatchesManagedReference(int rows, int cols)
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built");

        var (values, scales, vec, reference) = BuildInputs(rows, cols);
        int blocksPerRow = cols / Q8Helpers.BlockElements;
        var result = new float[rows];

        cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);
        HybRunner runner = SatelliteLoader.LoadCuda()
            .SetDistrib(prop.multiProcessorCount * 16, 128);
        dynamic wrapped = runner.Wrap(new Q8Kernels());
        wrapped.MatVecMul(result, values, scales, vec, rows, blocksPerRow);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());

        _output.WriteLine($"CUDA Q8 matvec ran ({rows}×{cols})");
        AssertClose(reference, result, rtol: 5e-5f, atol: 1e-4f);
    }
}
