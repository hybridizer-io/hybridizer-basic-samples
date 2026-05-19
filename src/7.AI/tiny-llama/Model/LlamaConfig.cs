using LlamaCsharp.Gguf;

namespace LlamaCsharp.Model;

/// <summary>
/// LLaMA model hyperparameters extracted from GGUF metadata.
/// </summary>
public class LlamaConfig
{
    public int VocabSize { get; set; }
    public int EmbeddingDim { get; set; }       // llama.embedding_length
    public int HiddenDim { get; set; }          // llama.feed_forward_length
    public int NumLayers { get; set; }          // llama.block_count
    public int NumHeads { get; set; }           // llama.attention.head_count
    public int NumKvHeads { get; set; }         // llama.attention.head_count_kv (for GQA)
    public int HeadDim { get; set; }            // embedding_dim / num_heads
    public int ContextLength { get; set; }      // llama.context_length
    public float RmsNormEps { get; set; }       // llama.attention.layer_norm_rms_epsilon
    public float RopeFreqBase { get; set; }     // llama.rope.freq_base
    public int RopeDimCount { get; set; }       // llama.rope.dimension_count

    /// <summary>
    /// Number of query heads that share a single KV head (for Grouped Query Attention).
    /// </summary>
    public int GqaGroupSize => NumHeads / NumKvHeads;

    /// <summary>
    /// Create a LlamaConfig from GGUF metadata.
    /// </summary>
    public static LlamaConfig FromGguf(GgufReader reader)
    {
        var config = new LlamaConfig();

        // Determine architecture prefix
        var arch = reader.GetMetadata("general.architecture")?.GetString() ?? "llama";
        Console.WriteLine($"  Architecture: {arch}");

        config.EmbeddingDim = (int)GetUInt64OrUInt32(reader, $"{arch}.embedding_length");
        config.NumLayers = (int)GetUInt64OrUInt32(reader, $"{arch}.block_count");
        config.NumHeads = (int)GetUInt64OrUInt32(reader, $"{arch}.attention.head_count");
        config.ContextLength = (int)GetUInt64OrUInt32(reader, $"{arch}.context_length");

        // Feed forward length (optional — some models don't store it)
        var ffnMeta = reader.GetMetadata($"{arch}.feed_forward_length");
        config.HiddenDim = ffnMeta != null ? (int)ffnMeta.GetUInt64() : config.EmbeddingDim * 4;

        // KV heads (for GQA): defaults to num_heads if not specified
        var kvHeadsMeta = reader.GetMetadata($"{arch}.attention.head_count_kv");
        config.NumKvHeads = kvHeadsMeta != null ? (int)GetUInt64OrUInt32Value(kvHeadsMeta) : config.NumHeads;

        // Head dimension
        config.HeadDim = config.EmbeddingDim / config.NumHeads;

        // RMS norm epsilon
        var rmsMeta = reader.GetMetadata($"{arch}.attention.layer_norm_rms_epsilon");
        config.RmsNormEps = rmsMeta?.GetFloat32() ?? 1e-5f;

        // RoPE
        var ropeFreqMeta = reader.GetMetadata($"{arch}.rope.freq_base");
        config.RopeFreqBase = ropeFreqMeta?.GetFloat32() ?? 10000.0f;

        var ropeDimMeta = reader.GetMetadata($"{arch}.rope.dimension_count");
        config.RopeDimCount = ropeDimMeta != null ? (int)GetUInt64OrUInt32Value(ropeDimMeta) : config.HeadDim;

        // Vocab size: determined from tokenizer tokens array
        var tokensMeta = reader.GetMetadata("tokenizer.ggml.tokens");
        if (tokensMeta != null && tokensMeta.Value is object[] tokensArray)
        {
            config.VocabSize = tokensArray.Length;
        }
        else
        {
            // Fallback: infer from token_embd tensor dimensions
            var embdTensor = reader.TensorInfos.FirstOrDefault(t => t.Name == "token_embd.weight");
            config.VocabSize = embdTensor != null ? (int)embdTensor.Dimensions[0] : 32000;
        }

        return config;
    }

    /// <summary>
    /// Try to read as uint64 first, fallback to uint32 (some models use one or the other).
    /// </summary>
    private static ulong GetUInt64OrUInt32(GgufReader reader, string key)
    {
        var meta = reader.GetMetadata(key)
            ?? throw new InvalidDataException($"Required metadata key missing: {key}");
        return GetUInt64OrUInt32Value(meta);
    }

    private static ulong GetUInt64OrUInt32Value(GgufMetadataKv meta)
    {
        return meta.Value switch
        {
            ulong ul => ul,
            uint u => u,
            long l => (ulong)l,
            int i => (ulong)i,
            _ => Convert.ToUInt64(meta.Value)
        };
    }

    public void PrintSummary()
    {
        Console.WriteLine($"  Vocab size:     {VocabSize:N0}");
        Console.WriteLine($"  Embedding dim:  {EmbeddingDim}");
        Console.WriteLine($"  Hidden dim:     {HiddenDim}");
        Console.WriteLine($"  Num layers:     {NumLayers}");
        Console.WriteLine($"  Num heads:      {NumHeads} (KV heads: {NumKvHeads}, GQA group: {GqaGroupSize})");
        Console.WriteLine($"  Head dim:       {HeadDim}");
        Console.WriteLine($"  Context length: {ContextLength:N0}");
        Console.WriteLine($"  RoPE freq base: {RopeFreqBase:F1}");
        Console.WriteLine($"  RoPE dim count: {RopeDimCount}");
        Console.WriteLine($"  RMS norm eps:   {RmsNormEps}");
    }
}
