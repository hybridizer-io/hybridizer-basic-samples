namespace LlamaCsharp.Gguf;

/// <summary>
/// Represents a tensor stored in a GGUF file, including its metadata
/// and methods for dequantization to float[].
/// </summary>
public class GgufTensor
{
    public string Name { get; set; } = "";
    public uint NDimensions { get; set; }
    public ulong[] Dimensions { get; set; } = Array.Empty<ulong>();
    public GgmlType Type { get; set; }
    public ulong Offset { get; set; }

    /// <summary>
    /// Total number of elements in this tensor.
    /// </summary>
    public ulong ElementCount
    {
        get
        {
            ulong count = 1;
            for (int i = 0; i < NDimensions; i++)
                count *= Dimensions[i];
            return count;
        }
    }

    /// <summary>
    /// Size in bytes of the raw tensor data in the GGUF file.
    /// </summary>
    public ulong ByteSize
    {
        get
        {
            ulong elements = ElementCount;
            return Type switch
            {
                GgmlType.F32 => elements * 4,
                GgmlType.F16 => elements * 2,
                // Q8_0: blocks of 32 elements, each block = 2 bytes (f16 scale) + 32 bytes (int8 weights) = 34 bytes
                GgmlType.Q8_0 => (elements / 32) * 34,
                // Q4_0: blocks of 32 elements, each block = 2 bytes (f16 scale) + 16 bytes (4-bit weights) = 18 bytes
                GgmlType.Q4_0 => (elements / 32) * 18,
                _ => throw new NotSupportedException($"Unsupported tensor type: {Type}")
            };
        }
    }

    /// <summary>
    /// Dequantize the raw tensor data to float[].
    /// Uses Parallel.For for maximum CPU utilization.
    /// </summary>
    public float[] ToFloat32Array(byte[] rawData)
    {
        int numElements = (int)ElementCount;
        float[] result = new float[numElements];

        switch (Type)
        {
            case GgmlType.F32:
                DequantizeF32(rawData, result, numElements);
                break;
            case GgmlType.F16:
                DequantizeF16(rawData, result, numElements);
                break;
            case GgmlType.Q8_0:
                DequantizeQ8_0(rawData, result, numElements);
                break;
            case GgmlType.Q4_0:
                DequantizeQ4_0(rawData, result, numElements);
                break;
            default:
                throw new NotSupportedException($"Unsupported tensor type for dequantization: {Type}");
        }

        return result;
    }

    /// <summary>
    /// F32: Direct copy from bytes to float array.
    /// </summary>
    private static void DequantizeF32(byte[] rawData, float[] result, int numElements)
    {
        Buffer.BlockCopy(rawData, 0, result, 0, numElements * 4);
    }

    /// <summary>
    /// F16: Convert each Half (2 bytes) to float.
    /// Parallelized over element index.
    /// </summary>
    private static void DequantizeF16(byte[] rawData, float[] result, int numElements)
    {
        Parallel.For(0, numElements, i =>
        {
            ushort bits = BitConverter.ToUInt16(rawData, i * 2);
            result[i] = (float)BitConverter.UInt16BitsToHalf(bits);
        });
    }

    /// <summary>
    /// Q8_0: Block-based symmetric quantization.
    /// Each block: 32 weights, stored as 2 bytes (f16 scale) + 32 bytes (int8 values).
    /// Dequantization: w_i = scale * q_i
    /// Parallelized over blocks.
    /// </summary>
    private static void DequantizeQ8_0(byte[] rawData, float[] result, int numElements)
    {
        int numBlocks = numElements / 32;
        const int blockSize = 34; // 2 bytes scale + 32 bytes int8

        Parallel.For(0, numBlocks, block =>
        {
            int byteOffset = block * blockSize;

            // Read f16 scale
            ushort scaleBits = BitConverter.ToUInt16(rawData, byteOffset);
            float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);

            int elemOffset = block * 32;

            // Dequantize 32 int8 values
            for (int i = 0; i < 32; i++)
            {
                sbyte q = (sbyte)rawData[byteOffset + 2 + i];
                result[elemOffset + i] = scale * q;
            }
        });
    }

    /// <summary>
    /// Q4_0: Block-based symmetric 4-bit quantization.
    /// Each block: 32 weights, stored as 2 bytes (f16 scale) + 16 bytes (4-bit values packed).
    /// Dequantization: w_i = scale * (q_i - 8)  [unsigned 4-bit, bias of 8]
    /// Parallelized over blocks.
    /// </summary>
    private static void DequantizeQ4_0(byte[] rawData, float[] result, int numElements)
    {
        int numBlocks = numElements / 32;
        const int blockSize = 18; // 2 bytes scale + 16 bytes packed 4-bit

        Parallel.For(0, numBlocks, block =>
        {
            int byteOffset = block * blockSize;

            // Read f16 scale
            ushort scaleBits = BitConverter.ToUInt16(rawData, byteOffset);
            float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);

            int elemOffset = block * 32;

            // Dequantize 32 values from 16 packed bytes (2 x 4-bit per byte)
            for (int i = 0; i < 16; i++)
            {
                byte packed = rawData[byteOffset + 2 + i];

                // Low nibble (first element): unsigned 0-15, subtract 8 to center
                int lo = (packed & 0x0F) - 8;
                // High nibble (second element): unsigned 0-15, subtract 8 to center
                int hi = ((packed >> 4) & 0x0F) - 8;

                result[elemOffset + i] = scale * lo;
                result[elemOffset + i + 16] = scale * hi;
            }
        });
    }

    public override string ToString()
    {
        string dims = string.Join(" × ", Dimensions.Select(d => d.ToString()));
        return $"{Name} [{dims}] ({Type}, {ByteSize:N0} bytes)";
    }
}
