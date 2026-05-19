using Hybridizer.Runtime.CUDAImports;
using LlamaCsharp.Math;
using LlamaCsharp.Utils;
using Xunit;
using Xunit.Abstractions;

namespace LlamaCsharp.Tests;

/// <summary>
/// Step-4 smoke tests: prove the Hybridizer build/load/launch pipeline works
/// end-to-end on both flavors with the trivial SAXPY kernel from
/// <see cref="Smoke.Saxpy"/>. Skips cleanly if the satellite isn't built.
/// </summary>
[Trait("Category", "Hybridizer")]
public class SmokeKernelTests
{
    private readonly ITestOutputHelper _output;

    public SmokeKernelTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private const int N = 1 << 14;
    private const float A = 2.5f;

    private static (float[] x, float[] y, float[] expected) MakeInputs()
    {
        var rng = new Random(42);
        var x = new float[N];
        var y = new float[N];
        var expected = new float[N];
        for (int i = 0; i < N; i++)
        {
            x[i] = (float)rng.NextDouble();
            y[i] = (float)rng.NextDouble();
            expected[i] = A * x[i] + y[i];
        }
        return (x, y, expected);
    }

    private static void AssertSaxpy(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = 1e-5f + 1e-4f * MathF.Abs(expected[i]);
            Assert.True(diff <= tol, $"index {i}: expected {expected[i]}, actual {actual[i]}");
        }
    }

    [Fact]
    public void Managed_DualPathRunsAndMatchesReference()
    {
        // Same C# body, called directly (no satellite involved).
        var (x, y, expected) = MakeInputs();
        Smoke.Saxpy(N, A, x, y);
        AssertSaxpy(expected, y);
    }

    [SkippableFact]
    public void Cuda_SaxpyMatchesManaged()
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(),
            "CUDA satellite (*_CUDA.dll) not next to test assembly — build LlamaCsharp with <CompileCUDA>enable</CompileCUDA>.");

        var (x, y, expected) = MakeInputs();
        cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);
        HybRunner runner = SatelliteLoader.LoadCuda()
            .SetDistrib(prop.multiProcessorCount * 16, 128);
        dynamic wrapped = runner.Wrap(new Smoke());
        wrapped.Saxpy(N, A, x, y);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        _output.WriteLine($"CUDA satellite loaded, SAXPY ran on {prop.multiProcessorCount} SMs");
        AssertSaxpy(expected, y);
    }

    [SkippableFact]
    public void Omp_SaxpyMatchesManaged()
    {
        Skip.IfNot(SatelliteLoader.OmpAvailable(),
            "OMP satellite (lib*_OMP.so) not next to test assembly — build LlamaCsharp with <CompileOMP>enable</CompileOMP>.");

        var (x, y, expected) = MakeInputs();
        HybRunner runner = SatelliteLoader.LoadOmp();
        dynamic wrapped = runner.Wrap(new Smoke());
        wrapped.Saxpy(N, A, x, y);
        _output.WriteLine("OMP satellite loaded, SAXPY ran");
        AssertSaxpy(expected, y);
    }
}
