namespace LlamaCsharp.Gguf;

/// <summary>
/// GGML tensor data types as defined in the GGUF specification.
/// </summary>
public enum GgmlType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

/// <summary>
/// GGUF metadata value types.
/// </summary>
public enum GgufMetadataValueType : uint
{
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// <summary>
/// A metadata key-value pair from a GGUF file.
/// </summary>
public class GgufMetadataKv
{
    public string Key { get; set; } = "";
    public GgufMetadataValueType ValueType { get; set; }
    public object Value { get; set; } = null!;

    public string GetString() => (string)Value;

    public uint GetUInt32()
    {
        return Value switch
        {
            uint u => u,
            int i => (uint)i,
            ulong ul => (uint)ul,
            long l => (uint)l,
            _ => Convert.ToUInt32(Value)
        };
    }

    public int GetInt32()
    {
        return Value switch
        {
            int i => i,
            uint u => (int)u,
            long l => (int)l,
            ulong ul => (int)ul,
            _ => Convert.ToInt32(Value)
        };
    }

    public ulong GetUInt64()
    {
        return Value switch
        {
            ulong ul => ul,
            long l => (ulong)l,
            uint u => u,
            int i => (ulong)i,
            _ => Convert.ToUInt64(Value)
        };
    }

    public float GetFloat32()
    {
        return Value switch
        {
            float f => f,
            double d => (float)d,
            _ => Convert.ToSingle(Value)
        };
    }

    public string[] GetStringArray()
    {
        if (Value is object[] arr)
            return arr.Cast<string>().ToArray();
        throw new InvalidCastException($"Metadata '{Key}' is not a string array");
    }

    public float[] GetFloat32Array()
    {
        if (Value is object[] arr)
            return arr.Select(Convert.ToSingle).ToArray();
        throw new InvalidCastException($"Metadata '{Key}' is not a float array");
    }

    public int[] GetInt32Array()
    {
        if (Value is object[] arr)
            return arr.Select(Convert.ToInt32).ToArray();
        throw new InvalidCastException($"Metadata '{Key}' is not an int array");
    }

    public override string ToString() => $"{Key} ({ValueType}) = {Value}";
}
