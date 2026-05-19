using Hybridizer.Runtime.CUDAImports;
using LlamaCsharp.Gguf;
using LlamaCsharp.Math;

namespace LlamaCsharp.Model;

/// <summary>
/// Matrix weight that keeps Q8_0 tensors compressed in memory and computes
/// matvecs directly on the quantized bytes.
/// </summary>
public sealed class WeightMatrix
{
    private readonly float[]? _floatData;
    private readonly byte[]? _q8Values;
    private readonly float[]? _q8Scales;
    private readonly int _blocksPerRow;

    // CUDA device-resident copies of the Q8_0 weight tables, lazily created the
    // first time MatVecMul runs under ComputeBackend.Cuda. Holding both forms
    // costs ~3% extra host RAM per Q8 tensor; we keep them so a backend switch
    // back to Managed/OMP is still possible (and so we can `Dispose` cleanly).
    private ResidentArrayGeneric<byte>? _q8ResValues;
    private FloatResidentArray? _q8ResScales;

    public int Rows { get; }
    public int Cols { get; }
    public GgmlType Type { get; }
    public bool IsQ8_0 => Type == GgmlType.Q8_0 && _q8Values != null;

    private WeightMatrix(int rows, int cols, GgmlType type, float[]? floatData, byte[]? q8Values, float[]? q8Scales)
    {
        Rows = rows;
        Cols = cols;
        Type = type;
        _floatData = floatData;
        _q8Values = q8Values;
        _q8Scales = q8Scales;
        _blocksPerRow = q8Values != null ? cols / Q8Decode.BlockElements : 0;
    }

    public static WeightMatrix FromTensor(GgufReader reader, GgufTensor tensor)
    {
        int cols = (int)tensor.Dimensions[0];
        int rows = tensor.NDimensions > 1 ? (int)tensor.Dimensions[1] : 1;
        byte[] raw = reader.ReadTensorData(tensor);

        if (tensor.Type == GgmlType.Q8_0 && tensor.NDimensions >= 2)
        {
            var (values, scales) = Q8Decode.SplitMatrix(raw, rows, cols);
            return new WeightMatrix(rows, cols, tensor.Type, null, values, scales);
        }

        return new WeightMatrix(rows, cols, tensor.Type, tensor.ToFloat32Array(raw), null, null);
    }

    public void MatVecMul(float[] result, float[] vector)
    {
        if (IsQ8_0)
        {
            if (GpuBackend.Mode == ComputeBackend.Cuda)
            {
                if (_q8ResValues == null)
                    PromoteToResident();
                GpuBackend.DispatchMatVecMulResident(result, _q8ResValues!, _q8ResScales!, vector, Rows, _blocksPerRow);
                return;
            }

            GpuBackend.DispatchMatVecMul(result, _q8Values!, _q8Scales!, vector, Rows, _blocksPerRow);
            return;
        }

        ParallelMath.MatVecMul(result, _floatData!, vector, Rows, Cols);
    }

    /// <summary>
    /// Bridge overload (step 3): host float[] vector, device-resident result.
    /// CUDA-only — the float[] / Managed / OMP paths still call the original
    /// <see cref="MatVecMul(float[], float[])"/> overload.
    /// </summary>
    public void MatVecMul(FloatResidentArray result, float[] vector)
    {
        if (!IsQ8_0)
            throw new NotSupportedException("Resident matvec is only implemented for Q8_0 weights on the CUDA path.");
        if (_q8ResValues == null)
            PromoteToResident();
        GpuBackend.DispatchMatVecMulResidentOut(result, _q8ResValues!, _q8ResScales!, vector, Rows, _blocksPerRow);
    }

    /// <summary>
    /// Fully-resident matvec (step 5 once everything is resident): both vector
    /// and result on device. CUDA-only.
    /// </summary>
    public void MatVecMul(FloatResidentArray result, FloatResidentArray vector)
    {
        if (!IsQ8_0)
            throw new NotSupportedException("Resident matvec is only implemented for Q8_0 weights on the CUDA path.");
        if (_q8ResValues == null)
            PromoteToResident();
        GpuBackend.DispatchMatVecMulFullyResident(result, _q8ResValues!, _q8ResScales!, vector, Rows, _blocksPerRow);
    }

    /// <summary>
    /// Bulk-copy <c>_q8Values</c> / <c>_q8Scales</c> into CUDA-resident arrays
    /// via <see cref="FloatResidentArray.HostPointer"/> / <see cref="ResidentArrayGeneric{T}.HostPointer"/>
    /// (auto-allocates the host buffer on first access). Marks status
    /// <see cref="ResidentArrayStatus.DeviceNeedsRefresh"/> so the first kernel
    /// call triggers a single H→D <c>cudaMemcpy</c> of the whole table — the
    /// last time these bytes will cross the bus until process exit.
    /// </summary>
    private unsafe void PromoteToResident()
    {
        var resValues = new ResidentArrayGeneric<byte>(_q8Values!.Length);
        fixed (byte* src = _q8Values)
        {
            Buffer.MemoryCopy(src, (void*)resValues.HostPointer, _q8Values.Length, _q8Values.Length);
        }
        resValues.Status = ResidentArrayStatus.DeviceNeedsRefresh;

        var resScales = new FloatResidentArray(_q8Scales!.Length);
        long scalesBytes = (long)_q8Scales.Length * sizeof(float);
        fixed (float* src = _q8Scales)
        {
            Buffer.MemoryCopy(src, (void*)resScales.HostPointer, scalesBytes, scalesBytes);
        }
        resScales.Status = ResidentArrayStatus.DeviceNeedsRefresh;

        _q8ResValues = resValues;
        _q8ResScales = resScales;
    }

    public void CopyRowTo(float[] destination, int rowIndex)
    {
        if (IsQ8_0)
        {
            Q8Decode.DequantizeRow(_q8Values!, _q8Scales!, rowIndex, Cols, destination, 0);
            return;
        }

        Array.Copy(_floatData!, rowIndex * Cols, destination, 0, Cols);
    }

    /// <summary>
    /// Resident-destination version of <see cref="CopyRowTo"/>: dequantizes
    /// one Q8_0 row directly into a <see cref="FloatResidentArray"/> via a
    /// device kernel. Used by the CUDA token-embedding lookup once
    /// <c>_x</c> is promoted in step 5. The Q8 weight buffers must already
    /// be resident (CUDA backend triggers promotion lazily on first matvec).
    /// </summary>
    public void CopyRowToResident(FloatResidentArray destination, int rowIndex)
    {
        if (!IsQ8_0)
            throw new NotSupportedException("Resident row copy is only implemented for Q8_0 weights on the CUDA path.");
        if (_q8ResValues == null)
            PromoteToResident();
        GpuBackend.DispatchDequantizeRowFullyResident(destination, _q8ResValues!, _q8ResScales!, rowIndex, Cols, _blocksPerRow);
    }

    /// <summary>
    /// Resident-destination row copy with the row index read from a
    /// device-resident int slot. Sibling of <see cref="CopyRowToResident"/>
    /// used by the GPU greedy sampler to feed the next-token id from the
    /// previous step's argmax straight into the embedding lookup with no
    /// H↔D roundtrip on the token id.
    /// </summary>
    public void CopyRowToResidentByDeviceIdx(FloatResidentArray destination, ResidentArrayGeneric<int> rowIdx)
    {
        if (!IsQ8_0)
            throw new NotSupportedException("Resident row copy is only implemented for Q8_0 weights on the CUDA path.");
        if (_q8ResValues == null)
            PromoteToResident();
        GpuBackend.DispatchDequantizeRowByDeviceIdxFullyResident(destination, _q8ResValues!, _q8ResScales!, rowIdx, Cols, _blocksPerRow);
    }

    public static void FusedMatVecMul3(
        float[] out1, WeightMatrix m1,
        float[] out2, WeightMatrix m2,
        float[] out3, WeightMatrix m3,
        float[] vector)
    {
        m1.MatVecMul(out1, vector);
        m2.MatVecMul(out2, vector);
        m3.MatVecMul(out3, vector);
    }

    public static void FusedMatVecMul2(
        float[] out1, WeightMatrix m1,
        float[] out2, WeightMatrix m2,
        float[] vector)
    {
        m1.MatVecMul(out1, vector);
        m2.MatVecMul(out2, vector);
    }
}

/// <summary>
/// Weights for a single transformer layer.
/// </summary>
public class LayerWeights
{
    public float[] AttnNormWeight = null!;      // [embed_dim]
    public WeightMatrix Wq = null!;             // [num_heads * head_dim, embed_dim]
    public WeightMatrix Wk = null!;             // [num_kv_heads * head_dim, embed_dim]
    public WeightMatrix Wv = null!;             // [num_kv_heads * head_dim, embed_dim]
    public WeightMatrix Wo = null!;             // [embed_dim, num_heads * head_dim]
    public float[] FfnNormWeight = null!;       // [embed_dim]
    public WeightMatrix W1 = null!;             // gate: [hidden_dim, embed_dim]
    public WeightMatrix W2 = null!;             // down: [embed_dim, hidden_dim]
    public WeightMatrix W3 = null!;             // up: [hidden_dim, embed_dim]
}

/// <summary>
/// All weights for a LLaMA model.
/// Large Q8_0 matrices stay compressed in memory to reduce RAM traffic.
/// </summary>
public class LlamaWeights
{
    public WeightMatrix TokenEmbedding = null!;
    public LayerWeights[] Layers = null!;
    public float[] OutputNormWeight = null!;
    public WeightMatrix OutputWeight = null!;

    public static LlamaWeights FromGguf(GgufReader reader, LlamaConfig config)
    {
        var weights = new LlamaWeights();
        weights.Layers = new LayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            weights.Layers[i] = new LayerWeights();

        var tensorDict = new Dictionary<string, GgufTensor>();
        foreach (var t in reader.TensorInfos)
            tensorDict[t.Name] = t;

        int loaded = 0;
        int total = tensorDict.Count;

        GgufTensor GetTensor(string name)
        {
            if (!tensorDict.TryGetValue(name, out var tensor))
                throw new InvalidDataException($"Missing tensor: {name}");

            loaded++;
            Console.Write($"\r  Loading tensors: {loaded}/{total} ({100 * loaded / total}%)   ");
            return tensor;
        }

        WeightMatrix LoadMatrix(string name) => WeightMatrix.FromTensor(reader, GetTensor(name));

        float[] LoadVector(string name)
        {
            GgufTensor tensor = GetTensor(name);
            byte[] raw = reader.ReadTensorData(tensor);
            return tensor.ToFloat32Array(raw);
        }

        WeightMatrix? TryLoadMatrix(string name)
        {
            if (!tensorDict.ContainsKey(name))
                return null;
            return LoadMatrix(name);
        }

        weights.TokenEmbedding = LoadMatrix("token_embd.weight");
        weights.OutputNormWeight = LoadVector("output_norm.weight");
        weights.OutputWeight = TryLoadMatrix("output.weight") ?? weights.TokenEmbedding;

        for (int l = 0; l < config.NumLayers; l++)
        {
            string prefix = $"blk.{l}";
            var layer = weights.Layers[l];

            layer.AttnNormWeight = LoadVector($"{prefix}.attn_norm.weight");
            layer.Wq = LoadMatrix($"{prefix}.attn_q.weight");
            layer.Wk = LoadMatrix($"{prefix}.attn_k.weight");
            layer.Wv = LoadMatrix($"{prefix}.attn_v.weight");
            layer.Wo = LoadMatrix($"{prefix}.attn_output.weight");

            layer.FfnNormWeight = LoadVector($"{prefix}.ffn_norm.weight");
            layer.W1 = LoadMatrix($"{prefix}.ffn_gate.weight");
            layer.W2 = LoadMatrix($"{prefix}.ffn_down.weight");
            layer.W3 = LoadMatrix($"{prefix}.ffn_up.weight");
        }

        Console.WriteLine($"\r  Loading tensors: {loaded}/{total} (100%) â€” Done!       ");
        return weights;
    }
}
