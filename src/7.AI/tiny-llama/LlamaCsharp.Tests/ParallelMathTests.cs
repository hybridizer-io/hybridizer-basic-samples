using LlamaCsharp.Math;
using Xunit;

namespace LlamaCsharp.Tests;

public class ParallelMathTests
{
    private static float[] LinSpace(int n, float start, float step, int seed = 0)
    {
        var rng = new Random(seed);
        var arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = start + i * step + (float)(rng.NextDouble() - 0.5);
        return arr;
    }

    private static float NaiveDot(float[] a, int aOffset, float[] b, int bOffset, int n)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += (double)a[aOffset + i] * b[bOffset + i];
        return (float)sum;
    }

    private static void AssertAllClose(float[] expected, float[] actual, float rtol = 1e-4f, float atol = 1e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float thresh = atol + rtol * MathF.Abs(expected[i]);
            Assert.True(diff <= thresh,
                $"index {i}: expected {expected[i]}, actual {actual[i]}, diff {diff} > {thresh}");
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(32)]
    [InlineData(2048)]
    public void DotProductSimd_MatchesNaive(int length)
    {
        var a = LinSpace(length + 5, -1.0f, 0.013f, seed: 42);
        var b = LinSpace(length + 5, 0.5f, -0.007f, seed: 99);
        float expected = NaiveDot(a, 2, b, 1, length);
        float actual = ParallelMath.DotProductSimd(a, 2, b, 1, length);
        float tol = 1e-5f + 1e-4f * MathF.Abs(expected);
        Assert.True(MathF.Abs(expected - actual) <= tol,
            $"expected {expected}, actual {actual}, diff {MathF.Abs(expected - actual)} > {tol}");
    }

    [Fact]
    public void MatVecMul_F32_MatchesNaive()
    {
        const int rows = 17;
        const int cols = 73;
        var mat = LinSpace(rows * cols, -0.3f, 0.011f, seed: 1);
        var vec = LinSpace(cols, 0.2f, -0.013f, seed: 2);

        var expected = new float[rows];
        for (int i = 0; i < rows; i++)
            expected[i] = NaiveDot(mat, i * cols, vec, 0, cols);

        var actual = new float[rows];
        ParallelMath.MatVecMul(actual, mat, vec, rows, cols);
        AssertAllClose(expected, actual);
    }

    [Fact]
    public void FusedMatVecMul3_F32_MatchesIndividual()
    {
        const int cols = 64;
        int[] rowCounts = { 13, 7, 5 };
        var mats = rowCounts.Select((r, i) => LinSpace(r * cols, -0.1f * (i + 1), 0.007f, seed: i + 10)).ToArray();
        var vec = LinSpace(cols, 0.4f, -0.009f, seed: 7);

        var out1 = new float[rowCounts[0]];
        var out2 = new float[rowCounts[1]];
        var out3 = new float[rowCounts[2]];
        ParallelMath.FusedMatVecMul3(
            out1, mats[0], rowCounts[0],
            out2, mats[1], rowCounts[1],
            out3, mats[2], rowCounts[2],
            vec, cols);

        var ref1 = new float[rowCounts[0]];
        var ref2 = new float[rowCounts[1]];
        var ref3 = new float[rowCounts[2]];
        ParallelMath.MatVecMul(ref1, mats[0], vec, rowCounts[0], cols);
        ParallelMath.MatVecMul(ref2, mats[1], vec, rowCounts[1], cols);
        ParallelMath.MatVecMul(ref3, mats[2], vec, rowCounts[2], cols);

        AssertAllClose(ref1, out1);
        AssertAllClose(ref2, out2);
        AssertAllClose(ref3, out3);
    }

    [Fact]
    public void FusedMatVecMul2_F32_MatchesIndividual()
    {
        const int cols = 96;
        int[] rowCounts = { 23, 11 };
        var mat1 = LinSpace(rowCounts[0] * cols, 0.05f, 0.003f, seed: 100);
        var mat2 = LinSpace(rowCounts[1] * cols, -0.05f, 0.005f, seed: 200);
        var vec = LinSpace(cols, 0.1f, -0.002f, seed: 300);

        var out1 = new float[rowCounts[0]];
        var out2 = new float[rowCounts[1]];
        ParallelMath.FusedMatVecMul2(out1, mat1, rowCounts[0], out2, mat2, rowCounts[1], vec, cols);

        var ref1 = new float[rowCounts[0]];
        var ref2 = new float[rowCounts[1]];
        ParallelMath.MatVecMul(ref1, mat1, vec, rowCounts[0], cols);
        ParallelMath.MatVecMul(ref2, mat2, vec, rowCounts[1], cols);

        AssertAllClose(ref1, out1);
        AssertAllClose(ref2, out2);
    }

    [Fact]
    public void RmsNorm_MatchesNaive()
    {
        const int size = 2048;
        var input = LinSpace(size, 0.3f, -0.001f, seed: 11);
        var weight = LinSpace(size, 1.0f, 0.0001f, seed: 22);
        const float eps = 1e-5f;

        double sumSq = 0;
        for (int i = 0; i < size; i++)
            sumSq += (double)input[i] * input[i];
        double scale = 1.0 / System.Math.Sqrt(sumSq / size + eps);

        var expected = new float[size];
        for (int i = 0; i < size; i++)
            expected[i] = (float)((double)input[i] * scale * weight[i]);

        var actual = new float[size];
        ParallelMath.RmsNorm(actual, input, weight, size, eps);
        AssertAllClose(expected, actual, rtol: 5e-4f);
    }

    [Fact]
    public void RmsNorm_NonVectorTail_Works()
    {
        const int size = 13;
        var input = LinSpace(size, 0.7f, -0.05f, seed: 33);
        var weight = new float[size];
        Array.Fill(weight, 1.0f);

        double sumSq = 0;
        for (int i = 0; i < size; i++) sumSq += (double)input[i] * input[i];
        double scale = 1.0 / System.Math.Sqrt(sumSq / size + 1e-5);

        var expected = new float[size];
        for (int i = 0; i < size; i++)
            expected[i] = (float)((double)input[i] * scale);

        var actual = new float[size];
        ParallelMath.RmsNorm(actual, input, weight, size, 1e-5f);
        AssertAllClose(expected, actual);
    }

    [Fact]
    public void Softmax_Small_SumsToOne()
    {
        var x = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 1.5f };
        ParallelMath.Softmax(x, 0, x.Length);

        float sum = 0;
        foreach (var v in x) sum += v;
        Assert.Equal(1.0f, sum, tolerance: 1e-6f);

        Assert.True(x.All(v => v >= 0));
        int argmax = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[argmax]) argmax = i;
        Assert.Equal(3, argmax);
    }

    [Fact]
    public void Softmax_Large_Parallel_SumsToOne()
    {
        const int size = 10000;
        var x = LinSpace(size, -5.0f, 0.001f, seed: 55);

        ParallelMath.Softmax(x, size);

        double sum = 0;
        for (int i = 0; i < size; i++) sum += x[i];
        Assert.Equal(1.0, sum, tolerance: 1e-4);
        Assert.True(x.All(v => v >= 0));
    }

    [Fact]
    public void Softmax_Offset_Variant()
    {
        var x = new float[10];
        for (int i = 0; i < x.Length; i++) x[i] = 100.0f;
        x[3] = 0.0f;
        x[4] = 1.0f;
        x[5] = 2.0f;
        x[6] = 3.0f;
        x[7] = 4.0f;

        ParallelMath.Softmax(x, offset: 3, size: 5);

        float sum = 0;
        for (int i = 3; i < 8; i++) sum += x[i];
        Assert.Equal(1.0f, sum, tolerance: 1e-6f);

        Assert.Equal(100.0f, x[0]);
        Assert.Equal(100.0f, x[1]);
        Assert.Equal(100.0f, x[2]);
        Assert.Equal(100.0f, x[8]);
        Assert.Equal(100.0f, x[9]);
    }

    [Fact]
    public void ElementWiseMul_MatchesNaive()
    {
        const int size = 1234;
        var a = LinSpace(size, -1.0f, 0.003f, seed: 1);
        var b = LinSpace(size, 0.5f, -0.001f, seed: 2);
        var expected = a.Zip(b, (x, y) => x * y).ToArray();
        var actual = new float[size];
        ParallelMath.ElementWiseMul(actual, a, b, size);
        AssertAllClose(expected, actual);
    }

    [Fact]
    public void Silu_MatchesNaive()
    {
        var x = new float[] { -2f, -1f, 0f, 1f, 2f, 3.14f };
        var expected = x.Select(v => v * (1.0f / (1.0f + MathF.Exp(-v)))).ToArray();
        ParallelMath.Silu(x, x.Length);
        AssertAllClose(expected, x, rtol: 1e-6f);
    }

    [Fact]
    public void FusedSiluElementWiseMul_Small_MatchesNaive()
    {
        const int size = 64;
        var gate = LinSpace(size, -1.0f, 0.05f, seed: 1);
        var up = LinSpace(size, 0.5f, -0.02f, seed: 2);

        var expected = new float[size];
        for (int i = 0; i < size; i++)
        {
            float silu = gate[i] * (1.0f / (1.0f + MathF.Exp(-gate[i])));
            expected[i] = silu * up[i];
        }

        ParallelMath.FusedSiluElementWiseMul(gate, up, size);
        AssertAllClose(expected, gate);
    }

    [Fact]
    public void FusedSiluElementWiseMul_LargeParallel_MatchesNaive()
    {
        const int size = 16384;
        var gate = LinSpace(size, -1.0f, 0.0001f, seed: 1);
        var up = LinSpace(size, 0.5f, -0.00007f, seed: 2);

        var expected = new float[size];
        for (int i = 0; i < size; i++)
        {
            float silu = gate[i] * (1.0f / (1.0f + MathF.Exp(-gate[i])));
            expected[i] = silu * up[i];
        }

        ParallelMath.FusedSiluElementWiseMul(gate, up, size);
        AssertAllClose(expected, gate);
    }

    [Fact]
    public void Accumulate_MatchesNaive()
    {
        const int size = 1000;
        var a = LinSpace(size, 0.0f, 0.01f, seed: 1);
        var b = LinSpace(size, 5.0f, -0.01f, seed: 2);
        var expected = a.Zip(b, (x, y) => x + y).ToArray();

        ParallelMath.Accumulate(a, b, size);
        AssertAllClose(expected, a);
    }

    [Fact]
    public void Argmax_FindsMaxIndex()
    {
        var x = new float[] { 0.1f, 0.5f, -2.0f, 3.14f, 1.5f, 3.14f - 0.0001f };
        Assert.Equal(3, ParallelMath.Argmax(x, x.Length));
    }

    [Fact]
    public void ApplyRope_RotatesPairs_AsExpected()
    {
        const int headDim = 4;
        const int numHeads = 1;
        const int numKvHeads = 1;
        const int pairs = 2;
        const int contextLen = 1;

        var cosTable = new float[contextLen * pairs];
        var sinTable = new float[contextLen * pairs];
        cosTable[0] = MathF.Cos(0.5f); sinTable[0] = MathF.Sin(0.5f);
        cosTable[1] = MathF.Cos(0.25f); sinTable[1] = MathF.Sin(0.25f);

        var q = new float[] { 1.0f, 0.0f, 0.0f, 1.0f };
        var k = new float[] { 0.5f, 0.5f, 1.0f, 0.0f };

        var qExpected = new float[4];
        qExpected[0] = q[0] * cosTable[0] - q[1] * sinTable[0];
        qExpected[1] = q[0] * sinTable[0] + q[1] * cosTable[0];
        qExpected[2] = q[2] * cosTable[1] - q[3] * sinTable[1];
        qExpected[3] = q[2] * sinTable[1] + q[3] * cosTable[1];

        var kExpected = new float[4];
        kExpected[0] = k[0] * cosTable[0] - k[1] * sinTable[0];
        kExpected[1] = k[0] * sinTable[0] + k[1] * cosTable[0];
        kExpected[2] = k[2] * cosTable[1] - k[3] * sinTable[1];
        kExpected[3] = k[2] * sinTable[1] + k[3] * cosTable[1];

        ParallelMath.ApplyRope(q, k, headDim, numHeads, numKvHeads, cosTable, sinTable, ropeOffset: 0, ropePairCount: pairs);

        AssertAllClose(qExpected, q, rtol: 1e-5f);
        AssertAllClose(kExpected, k, rtol: 1e-5f);
    }
}
