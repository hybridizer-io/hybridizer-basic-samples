namespace LlamaCsharp.Math;

/// <summary>
/// Host-side helpers for the Q8_0 split layout consumed by <see cref="Q8Kernels"/>.
/// The interleaved GGUF layout (2-byte FP16 scale + 32 int8 values per 34-byte block)
/// is decoded once at model load into two arrays: byte[] values (32 sbytes/block) and
/// float[] scales (one FP32 per block). This keeps FP16 decode out of the kernel hot
/// path — see [[reference-hybridizer-quant-split-layout]].
/// </summary>
public static class Q8Decode
{
    public const int BlockElements = 32;
    public const int InterleavedBlockBytes = 34;

    public static int InterleavedRowStride(int cols) =>
        ((cols + BlockElements - 1) / BlockElements) * InterleavedBlockBytes;

    /// <summary>
    /// Decompose an interleaved Q8_0 byte[] matrix into the split layout:
    /// values[r * cols + b * 32 + i] = sbyte stored as byte
    /// scales[r * blocksPerRow + b] = FP32 (pre-decoded from FP16)
    /// Requires cols % 32 == 0 (true for all TinyLlama tensors).
    /// </summary>
    public static (byte[] Values, float[] Scales) SplitMatrix(byte[] interleaved, int rows, int cols)
    {
        if (cols % BlockElements != 0)
            throw new ArgumentException($"cols ({cols}) must be a multiple of {BlockElements} for the split layout");

        int blocksPerRow = cols / BlockElements;
        var values = new byte[rows * cols];
        var scales = new float[rows * blocksPerRow];
        int rowStride = InterleavedRowStride(cols);

        for (int r = 0; r < rows; r++)
        {
            int srcRow = r * rowStride;
            int dstValuesRow = r * cols;
            int dstScalesRow = r * blocksPerRow;
            for (int b = 0; b < blocksPerRow; b++)
            {
                int blockOff = srcRow + b * InterleavedBlockBytes;
                ushort scaleBits = (ushort)(interleaved[blockOff] | (interleaved[blockOff + 1] << 8));
                scales[dstScalesRow + b] = (float)BitConverter.UInt16BitsToHalf(scaleBits);
                Array.Copy(interleaved, blockOff + 2, values, dstValuesRow + b * BlockElements, BlockElements);
            }
        }
        return (values, scales);
    }

    /// <summary>
    /// Dequantize one row of the split-layout matrix into dst[dstOffset..dstOffset+cols].
    /// </summary>
    public static void DequantizeRow(byte[] values, float[] scales, int rowIndex, int cols, float[] dst, int dstOffset)
    {
        int blocksPerRow = cols / BlockElements;
        int valuesRowOffset = rowIndex * cols;
        int scalesRowOffset = rowIndex * blocksPerRow;
        for (int b = 0; b < blocksPerRow; b++)
        {
            float scale = scales[scalesRowOffset + b];
            int blockOff = valuesRowOffset + b * BlockElements;
            int dstBlockOff = dstOffset + b * BlockElements;
            for (int i = 0; i < BlockElements; i++)
                dst[dstBlockOff + i] = scale * (sbyte)values[blockOff + i];
        }
    }
}
