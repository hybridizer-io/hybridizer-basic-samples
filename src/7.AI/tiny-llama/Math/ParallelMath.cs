using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace LlamaCsharp.Math;

/// <summary>
/// Massively parallel mathematical operations for transformer inference.
/// 
/// OPTIMIZATION STRATEGY (vs naive Parallel.For on every op):
///   1. FUSED MatVecMul: QKV and Gate+Up computed in a SINGLE parallel pass
///      -> eliminates 2-3 barrier syncs per layer (the #1 source of idle CPU)
///   2. SIMD inner loops: Vector&lt;float&gt; for dot products and element ops
///      -> 4-8x speedup on the inner loop (AVX2/SSE)
///   3. Sequential small ops: RMSNorm, SiLU, Accumulate on dim<=8192 are
///      faster sequential+SIMD than Parallel.For (overhead > work)
/// </summary>
public static class ParallelMath
{
    // Below this size, sequential + SIMD is faster than Parallel.For
    private const int ParallelThreshold = 8192;

    // SIMD vector width (typically 4 for SSE, 8 for AVX2)
    private static readonly int VectorWidth = Vector<float>.Count;

    // ====================================================================
    // CORE: Matrix-Vector Multiply (single output)
    // ====================================================================

    /// <summary>
    /// Matrix-Vector multiplication: result[i] = dot(matrix[i*cols..], vector[0..cols])
    /// Parallelized over rows, SIMD-accelerated inner dot product.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MatVecMul(float[] result, float[] matrix, float[] vector, int rows, int cols)
    {
        Parallel.For(0, rows, i =>
        {
            result[i] = DotProductSimd(matrix, i * cols, vector, 0, cols);
        });
    }

    // ====================================================================
    // FUSED: Multiple MatVecMuls in ONE parallel pass
    // ====================================================================

    /// <summary>
    /// Fused triple MatVecMul: computes Q, K, V projections in a single parallel pass.
    /// Instead of 3 sequential parallel loops (each with a barrier sync), this distributes
    /// ALL rows (Q rows + K rows + V rows) across threads in one pass.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void FusedMatVecMul3(
        float[] out1, float[] mat1, int rows1,
        float[] out2, float[] mat2, int rows2,
        float[] out3, float[] mat3, int rows3,
        float[] vector, int cols)
    {
        int totalRows = rows1 + rows2 + rows3;

        Parallel.For(0, totalRows, i =>
        {
            float[] outArr;
            float[] mat;
            int localRow;

            if (i < rows1)
            {
                outArr = out1;
                mat = mat1;
                localRow = i;
            }
            else if (i < rows1 + rows2)
            {
                outArr = out2;
                mat = mat2;
                localRow = i - rows1;
            }
            else
            {
                outArr = out3;
                mat = mat3;
                localRow = i - rows1 - rows2;
            }

            outArr[localRow] = DotProductSimd(mat, localRow * cols, vector, 0, cols);
        });
    }

    /// <summary>
    /// Fused double MatVecMul: computes Gate and Up projections in a single parallel pass.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void FusedMatVecMul2(
        float[] out1, float[] mat1, int rows1,
        float[] out2, float[] mat2, int rows2,
        float[] vector, int cols)
    {
        int totalRows = rows1 + rows2;

        Parallel.For(0, totalRows, i =>
        {
            float[] outArr;
            float[] mat;
            int localRow;

            if (i < rows1)
            {
                outArr = out1;
                mat = mat1;
                localRow = i;
            }
            else
            {
                outArr = out2;
                mat = mat2;
                localRow = i - rows1;
            }

            outArr[localRow] = DotProductSimd(mat, localRow * cols, vector, 0, cols);
        });
    }

    // ====================================================================
    // FUSED: SiLU + ElementWise multiply in one pass
    // ====================================================================

    /// <summary>
    /// Fused SiLU activation + element-wise multiply: gate[i] = SiLU(gate[i]) * up[i]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void FusedSiluElementWiseMul(float[] gate, float[] up, int size)
    {
        if (size >= ParallelThreshold)
        {
            Parallel.For(0, size, i =>
            {
                float val = gate[i];
                gate[i] = val * (1.0f / (1.0f + MathF.Exp(-val))) * up[i];
            });
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                float val = gate[i];
                gate[i] = val * (1.0f / (1.0f + MathF.Exp(-val))) * up[i];
            }
        }
    }

    // ====================================================================
    // RMS Normalization (sequential + SIMD for typical dims <= 8192)
    // ====================================================================

    /// <summary>
    /// RMS Normalization: output[i] = (input[i] / rms) * weight[i]
    /// where rms = sqrt(mean(input²) + eps)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RmsNorm(float[] output, float[] input, float[] weight, int size, float eps)
    {
        float sumSquares = 0f;
        int j = 0;

        if (Vector.IsHardwareAccelerated && size >= VectorWidth)
        {
            var sumVec = Vector<float>.Zero;
            int limit = size - (size % VectorWidth);
            for (; j < limit; j += VectorWidth)
            {
                var v = new Vector<float>(input, j);
                sumVec += v * v;
            }

            for (int vi = 0; vi < VectorWidth; vi++)
                sumSquares += sumVec[vi];
        }

        for (; j < size; j++)
            sumSquares += input[j] * input[j];

        float scale = 1.0f / MathF.Sqrt(sumSquares / size + eps);

        j = 0;
        if (Vector.IsHardwareAccelerated && size >= VectorWidth)
        {
            var scaleVec = new Vector<float>(scale);
            int limit = size - (size % VectorWidth);
            for (; j < limit; j += VectorWidth)
            {
                var inp = new Vector<float>(input, j);
                var w = new Vector<float>(weight, j);
                (inp * scaleVec * w).CopyTo(output, j);
            }
        }

        for (; j < size; j++)
            output[j] = input[j] * scale * weight[j];
    }

    // ====================================================================
    // Softmax
    // ====================================================================

    /// <summary>
    /// In-place softmax over x[offset..offset+size].
    /// Sequential and used for per-head attention scores.
    /// </summary>
    public static void Softmax(float[] x, int offset, int size)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < size; i++)
            if (x[offset + i] > max) max = x[offset + i];

        float sum = 0f;
        for (int i = 0; i < size; i++)
        {
            x[offset + i] = MathF.Exp(x[offset + i] - max);
            sum += x[offset + i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < size; i++)
            x[offset + i] *= invSum;
    }

    /// <summary>
    /// In-place softmax over x[0..size].
    /// Parallel for large arrays (vocab logits).
    /// </summary>
    public static void Softmax(float[] x, int size)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < size; i++)
            if (x[i] > max) max = x[i];

        if (size > ParallelThreshold)
        {
            // Single-cell heap slot for atomic combine — replaces a managed-only
            // lock + closure-captured local with the [[reference-hybridizer-reductions]]
            // canonical atomic-add pattern. `Interlocked.Add` over float requires a CAS
            // loop in .NET 8 (no native overload), which is also exactly what would
            // lower to `atomicAdd` on CUDA via the .builtins mapping.
            float[] sumBox = new float[1];

            Parallel.For(0, size,
                () => 0f,
                (i, state, localSum) =>
                {
                    x[i] = MathF.Exp(x[i] - max);
                    return localSum + x[i];
                },
                localSum => AtomicAddFloat(ref sumBox[0], localSum));

            float invSum = 1.0f / sumBox[0];
            Parallel.For(0, size, i => { x[i] *= invSum; });
        }
        else
        {
            float sum = 0f;
            for (int i = 0; i < size; i++)
            {
                x[i] = MathF.Exp(x[i] - max);
                sum += x[i];
            }

            float invSum = 1.0f / sum;
            for (int i = 0; i < size; i++)
                x[i] *= invSum;
        }
    }

    // ====================================================================
    // Atomic helpers
    // ====================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AtomicAddFloat(ref float target, float value)
    {
        float oldVal, newVal;
        do
        {
            oldVal = target;
            newVal = oldVal + value;
        } while (Interlocked.CompareExchange(ref target, newVal, oldVal) != oldVal);
    }

    // ====================================================================
    // Element-wise operations (sequential + SIMD for small dims)
    // ====================================================================

    /// <summary>
    /// Element-wise multiplication: result[i] = a[i] * b[i]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ElementWiseMul(float[] result, float[] a, float[] b, int size)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated && size >= VectorWidth)
        {
            int limit = size - (size % VectorWidth);
            for (; j < limit; j += VectorWidth)
            {
                var va = new Vector<float>(a, j);
                var vb = new Vector<float>(b, j);
                (va * vb).CopyTo(result, j);
            }
        }

        for (; j < size; j++)
            result[j] = a[j] * b[j];
    }

    /// <summary>
    /// In-place SiLU (Swish): x[i] = x[i] * sigmoid(x[i])
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Silu(float[] x, int size)
    {
        for (int i = 0; i < size; i++)
        {
            float val = x[i];
            x[i] = val * (1.0f / (1.0f + MathF.Exp(-val)));
        }
    }

    /// <summary>
    /// In-place accumulation: a[i] += b[i]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Accumulate(float[] a, float[] b, int size)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated && size >= VectorWidth)
        {
            int limit = size - (size % VectorWidth);
            for (; j < limit; j += VectorWidth)
            {
                var va = new Vector<float>(a, j);
                var vb = new Vector<float>(b, j);
                (va + vb).CopyTo(a, j);
            }
        }

        for (; j < size; j++)
            a[j] += b[j];
    }

    // ====================================================================
    // RoPE - fused Q+K in one Parallel.For
    // ====================================================================

    /// <summary>
    /// Apply Rotary Positional Embedding (RoPE) to Q and K using precomputed cos/sin tables.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyRope(
        float[] q,
        float[] k,
        int headDim,
        int numHeads,
        int numKvHeads,
        float[] cosTable,
        float[] sinTable,
        int ropeOffset,
        int ropePairCount)
    {
        int totalHeads = numHeads + numKvHeads;

        Parallel.For(0, totalHeads, idx =>
        {
            bool isQuery = idx < numHeads;
            int h = isQuery ? idx : idx - numHeads;
            float[] target = isQuery ? q : k;
            int offset = h * headDim;

            for (int pair = 0; pair < ropePairCount; pair++)
            {
                int i = pair * 2;
                float cos = cosTable[ropeOffset + pair];
                float sin = sinTable[ropeOffset + pair];

                float v0 = target[offset + i];
                float v1 = target[offset + i + 1];
                target[offset + i] = v0 * cos - v1 * sin;
                target[offset + i + 1] = v0 * sin + v1 * cos;
            }
        });
    }

    // ====================================================================
    // Q8_0 kernels
    // ====================================================================

    private const int Q8BlockElements = 32;
    private const int Q8BlockSize = 34;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MatVecMulQ8_0(float[] result, byte[] matrix, float[] vector, int rows, int cols)
    {
        int rowStride = ((cols + Q8BlockElements - 1) / Q8BlockElements) * Q8BlockSize;
        Parallel.For(0, rows, i =>
        {
            result[i] = DotProductQ8_0(matrix, i * rowStride, vector, 0, cols);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void FusedMatVecMul3Q8_0(
        float[] out1, byte[] mat1, int rows1,
        float[] out2, byte[] mat2, int rows2,
        float[] out3, byte[] mat3, int rows3,
        float[] vector, int cols)
    {
        int totalRows = rows1 + rows2 + rows3;
        int rowStride = ((cols + Q8BlockElements - 1) / Q8BlockElements) * Q8BlockSize;

        Parallel.For(0, totalRows, i =>
        {
            float[] outArr;
            byte[] mat;
            int localRow;

            if (i < rows1)
            {
                outArr = out1;
                mat = mat1;
                localRow = i;
            }
            else if (i < rows1 + rows2)
            {
                outArr = out2;
                mat = mat2;
                localRow = i - rows1;
            }
            else
            {
                outArr = out3;
                mat = mat3;
                localRow = i - rows1 - rows2;
            }

            outArr[localRow] = DotProductQ8_0(mat, localRow * rowStride, vector, 0, cols);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void FusedMatVecMul2Q8_0(
        float[] out1, byte[] mat1, int rows1,
        float[] out2, byte[] mat2, int rows2,
        float[] vector, int cols)
    {
        int totalRows = rows1 + rows2;
        int rowStride = ((cols + Q8BlockElements - 1) / Q8BlockElements) * Q8BlockSize;

        Parallel.For(0, totalRows, i =>
        {
            float[] outArr;
            byte[] mat;
            int localRow;

            if (i < rows1)
            {
                outArr = out1;
                mat = mat1;
                localRow = i;
            }
            else
            {
                outArr = out2;
                mat = mat2;
                localRow = i - rows1;
            }

            outArr[localRow] = DotProductQ8_0(mat, localRow * rowStride, vector, 0, cols);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CopyDequantizedRowQ8_0(float[] dst, int dstOffset, byte[] matrix, int rowIndex, int cols)
    {
        int rowStride = ((cols + Q8BlockElements - 1) / Q8BlockElements) * Q8BlockSize;
        int matrixOffset = rowIndex * rowStride;
        int written = 0;

        while (written < cols)
        {
            ushort scaleBits = (ushort)(matrix[matrixOffset] | (matrix[matrixOffset + 1] << 8));
            float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);
            int blockCount = System.Math.Min(Q8BlockElements, cols - written);
            int valuesOffset = matrixOffset + 2;

            for (int i = 0; i < blockCount; i++)
                dst[dstOffset + written + i] = scale * (sbyte)matrix[valuesOffset + i];

            written += blockCount;
            matrixOffset += Q8BlockSize;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float DotProductQ8_0(byte[] matrix, int matrixOffset, float[] vector, int vectorOffset, int length)
    {
        float sum = 0f;
        int processed = 0;

        while (processed < length)
        {
            ushort scaleBits = (ushort)(matrix[matrixOffset] | (matrix[matrixOffset + 1] << 8));
            float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);
            int blockCount = System.Math.Min(Q8BlockElements, length - processed);
            int valuesOffset = matrixOffset + 2;
            float blockSum = 0f;

            if (Vector.IsHardwareAccelerated && blockCount >= Vector<sbyte>.Count)
            {
                ReadOnlySpan<sbyte> qSpan = MemoryMarshal.Cast<byte, sbyte>(matrix.AsSpan(valuesOffset, blockCount));
                int floatWidth = Vector<float>.Count;
                int chunkWidth = Vector<sbyte>.Count;
                int i = 0;
                var sum0 = Vector<float>.Zero;
                var sum1 = Vector<float>.Zero;
                var sum2 = Vector<float>.Zero;
                var sum3 = Vector<float>.Zero;

                for (; i <= blockCount - chunkWidth; i += chunkWidth)
                {
                    var qVec = new Vector<sbyte>(qSpan.Slice(i, chunkWidth));
                    Vector.Widen(qVec, out Vector<short> s0, out Vector<short> s1);
                    Vector.Widen(s0, out Vector<int> i00, out Vector<int> i01);
                    Vector.Widen(s1, out Vector<int> i10, out Vector<int> i11);

                    int baseOffset = vectorOffset + processed + i;
                    sum0 += Vector.ConvertToSingle(i00) * new Vector<float>(vector, baseOffset);
                    sum1 += Vector.ConvertToSingle(i01) * new Vector<float>(vector, baseOffset + floatWidth);
                    sum2 += Vector.ConvertToSingle(i10) * new Vector<float>(vector, baseOffset + 2 * floatWidth);
                    sum3 += Vector.ConvertToSingle(i11) * new Vector<float>(vector, baseOffset + 3 * floatWidth);
                }

                for (int vi = 0; vi < floatWidth; vi++)
                    blockSum += sum0[vi] + sum1[vi] + sum2[vi] + sum3[vi];

                for (; i < blockCount; i++)
                    blockSum += qSpan[i] * vector[vectorOffset + processed + i];
            }
            else
            {
                for (int i = 0; i < blockCount; i++)
                    blockSum += (sbyte)matrix[valuesOffset + i] * vector[vectorOffset + processed + i];
            }

            sum += scale * blockSum;
            processed += blockCount;
            matrixOffset += Q8BlockSize;
        }

        return sum;
    }

    // ====================================================================
    // Utilities
    // ====================================================================

    /// <summary>
    /// Argmax: returns the index of the maximum value.
    /// </summary>
    public static int Argmax(float[] x, int size)
    {
        int maxIdx = 0;
        float maxVal = x[0];
        for (int i = 1; i < size; i++)
        {
            if (x[i] > maxVal)
            {
                maxVal = x[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /// <summary>
    /// Copy elements from source to destination.
    /// </summary>
    public static void Copy(float[] dst, int dstOffset, float[] src, int srcOffset, int count)
    {
        Array.Copy(src, srcOffset, dst, dstOffset, count);
    }

    // ====================================================================
    // SIMD-accelerated dot product (inner hot loop)
    // ====================================================================

    /// <summary>
    /// SIMD-accelerated dot product between two float arrays at given offsets.
    /// This is the innermost hot loop of the entire inference engine.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float DotProductSimd(float[] a, int aOffset, float[] b, int bOffset, int length)
    {
        float sum = 0f;
        int j = 0;

        if (Vector.IsHardwareAccelerated && length >= VectorWidth)
        {
            var sumVec = Vector<float>.Zero;
            int limit = length - (length % VectorWidth);
            for (; j < limit; j += VectorWidth)
            {
                var va = new Vector<float>(a, aOffset + j);
                var vb = new Vector<float>(b, bOffset + j);
                sumVec += va * vb;
            }

            for (int vi = 0; vi < VectorWidth; vi++)
                sum += sumVec[vi];
        }

        for (; j < length; j++)
            sum += a[aOffset + j] * b[bOffset + j];

        return sum;
    }
}
