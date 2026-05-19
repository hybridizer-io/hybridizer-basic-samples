using LlamaCsharp.Math;
using Xunit;

namespace LlamaCsharp.Tests;

public class Q8MathTests
{
    private static float[] LinSpace(int n, float start, float step, int seed = 0)
    {
        var rng = new Random(seed);
        var arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = start + i * step + (float)(rng.NextDouble() - 0.5);
        return arr;
    }

    private static void AssertAllClose(float[] expected, float[] actual, float rtol, float atol)
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

    [Fact]
    public void CopyDequantizedRowQ8_0_RoundTripsKnownBlock()
    {
        // One block: scale = 1.0, values = -3, -2, -1, 0, 1, 2, 3, ..., 28
        var src = new float[Q8Helpers.BlockElements];
        for (int i = 0; i < src.Length; i++)
            src[i] = i - 3;
        var bytes = Q8Helpers.QuantizeRow(src, src.Length);

        var dequantized = new float[src.Length];
        ParallelMath.CopyDequantizedRowQ8_0(dequantized, 0, bytes, rowIndex: 0, cols: src.Length);

        // Scale here is 28/127 ≈ 0.22; rounding error <= 0.5 * scale per element
        AssertAllClose(src, dequantized, rtol: 0f, atol: 28f / 127f * 0.5f + 1e-5f);
    }

    [Fact]
    public void DotProductQ8_0_MatchesDequantizedDot()
    {
        // 3 blocks = 96 elements
        const int cols = 96;
        var rawRow = LinSpace(cols, -1.5f, 0.013f, seed: 1);
        var vector = LinSpace(cols, 0.3f, -0.007f, seed: 2);

        byte[] q8Row = Q8Helpers.QuantizeRow(rawRow, cols);
        float[] dequantizedRow = Q8Helpers.DequantizeRow(q8Row, cols);

        // Reference: naive double dot product on the dequantized row (this is what the Q8 kernel must reproduce)
        double expected = 0;
        for (int i = 0; i < cols; i++)
            expected += (double)dequantizedRow[i] * vector[i];

        float actual = ParallelMath.DotProductQ8_0(q8Row, 0, vector, 0, cols);
        Assert.True(MathF.Abs((float)expected - actual) <= 1e-3f + 1e-4f * (float)System.Math.Abs(expected),
            $"expected {expected}, actual {actual}");
    }

    [Fact]
    public void DotProductQ8_0_PartialTrailingBlock_Works()
    {
        // 32 + 17 = 49 elements: full block + partial block
        const int cols = 49;
        var rawRow = LinSpace(cols, -1.0f, 0.05f, seed: 3);
        var vector = LinSpace(cols, 0.4f, -0.01f, seed: 4);

        byte[] q8Row = Q8Helpers.QuantizeRow(rawRow, cols);
        float[] dequantizedRow = Q8Helpers.DequantizeRow(q8Row, cols);

        double expected = 0;
        for (int i = 0; i < cols; i++)
            expected += (double)dequantizedRow[i] * vector[i];

        float actual = ParallelMath.DotProductQ8_0(q8Row, 0, vector, 0, cols);
        Assert.True(MathF.Abs((float)expected - actual) <= 1e-4f + 1e-4f * (float)System.Math.Abs(expected),
            $"expected {expected}, actual {actual}");
    }

    [Fact]
    public void MatVecMulQ8_0_MatchesDequantizedMatVec()
    {
        const int rows = 23;
        const int cols = 128;
        var rawMat = LinSpace(rows * cols, -0.5f, 0.0007f, seed: 10);
        var vec = LinSpace(cols, 0.6f, -0.005f, seed: 11);

        byte[] q8Mat = Q8Helpers.QuantizeMatrix(rawMat, rows, cols);
        float[] dequantizedMat = Q8Helpers.DequantizeMatrix(q8Mat, rows, cols);

        var expected = new float[rows];
        ParallelMath.MatVecMul(expected, dequantizedMat, vec, rows, cols);

        var actual = new float[rows];
        ParallelMath.MatVecMulQ8_0(actual, q8Mat, vec, rows, cols);

        AssertAllClose(expected, actual, rtol: 5e-5f, atol: 1e-4f);
    }

    [Fact]
    public void FusedMatVecMul3Q8_0_MatchesIndividual()
    {
        const int cols = 96;
        int[] rowCounts = { 11, 7, 5 };
        var mats = new float[3][];
        var q8 = new byte[3][];
        for (int k = 0; k < 3; k++)
        {
            mats[k] = LinSpace(rowCounts[k] * cols, -0.4f * (k + 1), 0.003f * (k + 1), seed: 100 + k);
            q8[k] = Q8Helpers.QuantizeMatrix(mats[k], rowCounts[k], cols);
        }
        var vec = LinSpace(cols, 0.25f, -0.002f, seed: 200);

        var out1 = new float[rowCounts[0]];
        var out2 = new float[rowCounts[1]];
        var out3 = new float[rowCounts[2]];
        ParallelMath.FusedMatVecMul3Q8_0(
            out1, q8[0], rowCounts[0],
            out2, q8[1], rowCounts[1],
            out3, q8[2], rowCounts[2],
            vec, cols);

        var ref1 = new float[rowCounts[0]];
        var ref2 = new float[rowCounts[1]];
        var ref3 = new float[rowCounts[2]];
        ParallelMath.MatVecMulQ8_0(ref1, q8[0], vec, rowCounts[0], cols);
        ParallelMath.MatVecMulQ8_0(ref2, q8[1], vec, rowCounts[1], cols);
        ParallelMath.MatVecMulQ8_0(ref3, q8[2], vec, rowCounts[2], cols);

        AssertAllClose(ref1, out1, rtol: 0f, atol: 1e-6f);
        AssertAllClose(ref2, out2, rtol: 0f, atol: 1e-6f);
        AssertAllClose(ref3, out3, rtol: 0f, atol: 1e-6f);
    }

    [Fact]
    public void FusedMatVecMul2Q8_0_MatchesIndividual()
    {
        const int cols = 128;
        int[] rowCounts = { 19, 9 };
        var m1 = LinSpace(rowCounts[0] * cols, 0.05f, 0.001f, seed: 300);
        var m2 = LinSpace(rowCounts[1] * cols, -0.05f, 0.002f, seed: 400);
        var q1 = Q8Helpers.QuantizeMatrix(m1, rowCounts[0], cols);
        var q2 = Q8Helpers.QuantizeMatrix(m2, rowCounts[1], cols);
        var vec = LinSpace(cols, 0.15f, -0.001f, seed: 500);

        var out1 = new float[rowCounts[0]];
        var out2 = new float[rowCounts[1]];
        ParallelMath.FusedMatVecMul2Q8_0(out1, q1, rowCounts[0], out2, q2, rowCounts[1], vec, cols);

        var ref1 = new float[rowCounts[0]];
        var ref2 = new float[rowCounts[1]];
        ParallelMath.MatVecMulQ8_0(ref1, q1, vec, rowCounts[0], cols);
        ParallelMath.MatVecMulQ8_0(ref2, q2, vec, rowCounts[1], cols);

        AssertAllClose(ref1, out1, rtol: 0f, atol: 1e-6f);
        AssertAllClose(ref2, out2, rtol: 0f, atol: 1e-6f);
    }

    [Fact]
    public void CopyDequantizedRowQ8_0_ReadsCorrectRowFromMultiRowMatrix()
    {
        const int rows = 4;
        const int cols = 64;
        var rawMat = LinSpace(rows * cols, -1.0f, 0.01f, seed: 777);
        var q8 = Q8Helpers.QuantizeMatrix(rawMat, rows, cols);

        // Pull row 2 with CopyDequantizedRowQ8_0
        var dst = new float[cols];
        ParallelMath.CopyDequantizedRowQ8_0(dst, 0, q8, rowIndex: 2, cols: cols);

        // Reference: dequantize the whole matrix and slice row 2
        var deq = Q8Helpers.DequantizeMatrix(q8, rows, cols);
        var expected = deq.AsSpan(2 * cols, cols).ToArray();

        AssertAllClose(expected, dst, rtol: 0f, atol: 1e-6f);
    }
}
