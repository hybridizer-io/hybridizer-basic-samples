using System.Text;

namespace LlamaCsharp.Gguf;

/// <summary>
/// Reads and parses GGUF binary files according to the GGUF v3 specification.
/// Extracts metadata key-value pairs, tensor information, and raw tensor data.
/// </summary>
public class GgufReader
{
    public uint Version { get; private set; }
    public ulong TensorCount { get; private set; }
    public ulong MetadataKvCount { get; private set; }
    public List<GgufMetadataKv> Metadata { get; private set; } = new();
    public List<GgufTensor> TensorInfos { get; private set; } = new();

    /// <summary>
    /// Absolute offset in the file where tensor data begins (after header + tensor infos + alignment padding).
    /// </summary>
    public long TensorDataOffset { get; private set; }

    private readonly string _filePath;

    public GgufReader(string filePath)
    {
        _filePath = filePath;
    }

    /// <summary>
    /// Parse the GGUF file: header, metadata, tensor infos.
    /// </summary>
    public void Load()
    {
        using var fs = File.OpenRead(_filePath);
        using var reader = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);

        // --- Magic number ---
        uint magic = reader.ReadUInt32();
        if (magic != 0x46554747) // "GGUF" in little-endian
            throw new InvalidDataException($"Invalid GGUF magic number: 0x{magic:X8} (expected 0x46554747)");

        // --- Version ---
        Version = reader.ReadUInt32();
        if (Version < 2 || Version > 3)
            throw new InvalidDataException($"Unsupported GGUF version: {Version} (expected 2 or 3)");

        // --- Tensor count ---
        TensorCount = reader.ReadUInt64();

        // --- Metadata KV count ---
        MetadataKvCount = reader.ReadUInt64();

        Console.WriteLine($"  GGUF v{Version} | {TensorCount} tensors | {MetadataKvCount} metadata entries");

        // --- Read metadata ---
        for (ulong i = 0; i < MetadataKvCount; i++)
        {
            var kv = ReadMetadataKv(reader);
            Metadata.Add(kv);
        }

        // --- Read tensor infos ---
        for (ulong i = 0; i < TensorCount; i++)
        {
            var ti = ReadTensorInfo(reader);
            TensorInfos.Add(ti);
        }

        // --- Calculate tensor data offset (aligned) ---
        long currentPos = fs.Position;
        uint alignment = GetAlignment();
        TensorDataOffset = AlignOffset(currentPos, alignment);
    }

    /// <summary>
    /// Read the raw bytes for a specific tensor from the file.
    /// </summary>
    public byte[] ReadTensorData(GgufTensor tensor)
    {
        int size = (int)tensor.ByteSize;
        byte[] data = new byte[size];

        using var fs = File.OpenRead(_filePath);
        fs.Seek(TensorDataOffset + (long)tensor.Offset, SeekOrigin.Begin);
        int bytesRead = fs.Read(data, 0, size);
        if (bytesRead != size)
            throw new IOException($"Failed to read tensor '{tensor.Name}': expected {size} bytes, got {bytesRead}");

        return data;
    }

    /// <summary>
    /// Get a metadata value by key, or null if not found.
    /// </summary>
    public GgufMetadataKv? GetMetadata(string key)
    {
        return Metadata.FirstOrDefault(m => m.Key == key);
    }

    /// <summary>
    /// Get the alignment value from metadata, defaulting to 32.
    /// </summary>
    public uint GetAlignment()
    {
        var kv = GetMetadata("general.alignment");
        return kv != null ? kv.GetUInt32() : 32u;
    }

    // ====================================================================
    // Private parsing methods
    // ====================================================================

    private static GgufMetadataKv ReadMetadataKv(BinaryReader reader)
    {
        var kv = new GgufMetadataKv();
        kv.Key = ReadString(reader);
        kv.ValueType = (GgufMetadataValueType)reader.ReadUInt32();
        kv.Value = ReadMetadataValue(reader, kv.ValueType);
        return kv;
    }

    private static object ReadMetadataValue(BinaryReader reader, GgufMetadataValueType type)
    {
        switch (type)
        {
            case GgufMetadataValueType.UInt8:
                return reader.ReadByte();
            case GgufMetadataValueType.Int8:
                return reader.ReadSByte();
            case GgufMetadataValueType.UInt16:
                return reader.ReadUInt16();
            case GgufMetadataValueType.Int16:
                return reader.ReadInt16();
            case GgufMetadataValueType.UInt32:
                return reader.ReadUInt32();
            case GgufMetadataValueType.Int32:
                return reader.ReadInt32();
            case GgufMetadataValueType.Float32:
                return reader.ReadSingle();
            case GgufMetadataValueType.Bool:
                return reader.ReadByte() != 0;
            case GgufMetadataValueType.String:
                return ReadString(reader);
            case GgufMetadataValueType.UInt64:
                return reader.ReadUInt64();
            case GgufMetadataValueType.Int64:
                return reader.ReadInt64();
            case GgufMetadataValueType.Float64:
                return reader.ReadDouble();
            case GgufMetadataValueType.Array:
                return ReadArray(reader);
            default:
                throw new InvalidDataException($"Unknown metadata value type: {type}");
        }
    }

    private static string ReadString(BinaryReader reader)
    {
        ulong length = reader.ReadUInt64();
        if (length > 1_000_000)
            throw new InvalidDataException($"String length too large: {length}");
        byte[] bytes = reader.ReadBytes((int)length);
        return Encoding.UTF8.GetString(bytes);
    }

    private static object[] ReadArray(BinaryReader reader)
    {
        var elementType = (GgufMetadataValueType)reader.ReadUInt32();
        ulong count = reader.ReadUInt64();

        if (count > 100_000_000)
            throw new InvalidDataException($"Array count too large: {count}");

        var result = new object[count];
        for (ulong i = 0; i < count; i++)
        {
            result[i] = ReadMetadataValue(reader, elementType);
        }
        return result;
    }

    private static GgufTensor ReadTensorInfo(BinaryReader reader)
    {
        var tensor = new GgufTensor();
        tensor.Name = ReadString(reader);
        tensor.NDimensions = reader.ReadUInt32();
        tensor.Dimensions = new ulong[tensor.NDimensions];
        for (int i = 0; i < tensor.NDimensions; i++)
        {
            tensor.Dimensions[i] = reader.ReadUInt64();
        }
        tensor.Type = (GgmlType)reader.ReadUInt32();
        tensor.Offset = reader.ReadUInt64();
        return tensor;
    }

    private static long AlignOffset(long offset, uint alignment)
    {
        long remainder = offset % alignment;
        if (remainder == 0) return offset;
        return offset + (alignment - remainder);
    }
}
