namespace LlamaCsharp.Tests;

/// <summary>
/// Q8_0 block layout (matches LlamaCsharp.Math.ParallelMath):
///   - 2 bytes: FP16 scale (little-endian)
///   - 32 bytes: int8 values
///   - 34 bytes per block
/// A row of `cols` elements stores ceil(cols/32) blocks contiguously.
/// </summary>
internal static class Q8Helpers
{
    public const int BlockElements = 32;
    public const int BlockBytes = 34;

    public static int RowStride(int cols) =>
        ((cols + BlockElements - 1) / BlockElements) * BlockBytes;

    /// <summary>Quantize one float row into a Q8_0 byte row using per-block scale = max|x|/127.</summary>
    public static byte[] QuantizeRow(float[] floats, int cols)
    {
        int rowStride = RowStride(cols);
        var bytes = new byte[rowStride];
        QuantizeRowInto(bytes, 0, floats, 0, cols);
        return bytes;
    }

    public static void QuantizeRowInto(byte[] dst, int dstOffset, float[] src, int srcOffset, int cols)
    {
        int written = 0;
        int offset = dstOffset;
        while (written < cols)
        {
            int blockCount = System.Math.Min(BlockElements, cols - written);

            float maxAbs = 0f;
            for (int i = 0; i < blockCount; i++)
            {
                float a = MathF.Abs(src[srcOffset + written + i]);
                if (a > maxAbs) maxAbs = a;
            }

            float scale = maxAbs / 127.0f;
            float invScale = scale > 0f ? 1.0f / scale : 0f;

            Half scaleHalf = (Half)scale;
            ushort scaleBits = BitConverter.HalfToUInt16Bits(scaleHalf);
            dst[offset] = (byte)(scaleBits & 0xFF);
            dst[offset + 1] = (byte)((scaleBits >> 8) & 0xFF);

            for (int i = 0; i < blockCount; i++)
            {
                float q = src[srcOffset + written + i] * invScale;
                int qi = (int)MathF.Round(q, MidpointRounding.AwayFromZero);
                if (qi > 127) qi = 127;
                if (qi < -128) qi = -128;
                dst[offset + 2 + i] = (byte)(sbyte)qi;
            }

            // Pad-block tail: if blockCount < 32 (last partial block), zero the remaining slots
            for (int i = blockCount; i < BlockElements; i++)
                dst[offset + 2 + i] = 0;

            written += blockCount;
            offset += BlockBytes;
        }
    }

    /// <summary>Dequantize a Q8_0 row using the same convention as ParallelMath.CopyDequantizedRowQ8_0.</summary>
    public static float[] DequantizeRow(byte[] bytes, int cols)
    {
        var dst = new float[cols];
        int written = 0;
        int offset = 0;
        while (written < cols)
        {
            ushort scaleBits = (ushort)(bytes[offset] | (bytes[offset + 1] << 8));
            float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);
            int blockCount = System.Math.Min(BlockElements, cols - written);
            for (int i = 0; i < blockCount; i++)
                dst[written + i] = scale * (sbyte)bytes[offset + 2 + i];

            written += blockCount;
            offset += BlockBytes;
        }
        return dst;
    }

    /// <summary>Quantize a full row-major float matrix to Q8_0.</summary>
    public static byte[] QuantizeMatrix(float[] floats, int rows, int cols)
    {
        int rowStride = RowStride(cols);
        var bytes = new byte[rows * rowStride];
        for (int r = 0; r < rows; r++)
            QuantizeRowInto(bytes, r * rowStride, floats, r * cols, cols);
        return bytes;
    }

    /// <summary>
    /// Decompose an interleaved Q8_0 byte[] matrix into (values, scales) — the
    /// layout the Hybridizer kernel <see cref="LlamaCsharp.Math.Q8Kernels.MatVecMul"/>
    /// expects. Thin wrapper around <see cref="LlamaCsharp.Math.Q8Decode.SplitMatrix"/>
    /// so production and test code share one decoder.
    /// </summary>
    public static (byte[] Values, float[] Scales) SplitMatrix(byte[] bytes, int rows, int cols) =>
        LlamaCsharp.Math.Q8Decode.SplitMatrix(bytes, rows, cols);

    /// <summary>Dequantize a full row-major Q8_0 matrix back to floats.</summary>
    public static float[] DequantizeMatrix(byte[] bytes, int rows, int cols)
    {
        var floats = new float[rows * cols];
        int rowStride = RowStride(cols);
        for (int r = 0; r < rows; r++)
        {
            var row = DequantizeRow(bytes.AsSpan(r * rowStride, rowStride).ToArray(), cols);
            Array.Copy(row, 0, floats, r * cols, cols);
        }
        return floats;
    }
}
