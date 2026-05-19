using Hybridizer.Runtime.CUDAImports;
using LlamaCsharp.Math;
using LlamaCsharp.Utils;

namespace LlamaCsharp.Model;

/// <summary>
/// LLaMA transformer forward pass implementation.
/// Performs full autoregressive inference with KV-cache,
/// Grouped Query Attention (GQA), RoPE, and SwiGLU FFN.
/// </summary>
public class LlamaTransformer
{
    private readonly LlamaConfig _config;
    private readonly LlamaWeights _weights;

    // KV-Cache: stores key and value vectors for all previous positions
    // Shape: [layer][position * kv_dim + head * head_dim + d]
    private readonly float[][] _keyCache;
    private readonly float[][] _valueCache;

    // Precomputed RoPE tables: [position * ropePairCount + pair]
    private readonly float[] _ropeCos;
    private readonly float[] _ropeSin;
    private readonly int _ropePairCount;

    // Pre-allocated working buffers (to avoid GC pressure during inference)
    private readonly float[] _x;          // Current hidden state [embed_dim]
    private readonly float[] _xNorm;      // Normalized hidden state [embed_dim]
    private readonly float[] _q;          // Query vector [num_heads * head_dim]
    private readonly float[] _k;          // Key vector [num_kv_heads * head_dim]
    private readonly float[] _v;          // Value vector [num_kv_heads * head_dim]
    private readonly float[] _attnOut;    // Attention output [num_heads * head_dim]
    private readonly float[] _attnProj;   // After Wo projection [embed_dim]
    private readonly float[] _gate;       // FFN gate (W1 output) [hidden_dim]
    private readonly float[] _up;         // FFN up (W3 output) [hidden_dim]
    private readonly float[] _ffnOut;     // FFN output (W2 output) [embed_dim]
    private readonly float[] _logits;     // Output logits [vocab_size]
    // Per-head attention scores scratch [numHeads * contextLen]. Owned here so
    // the GPU attention kernel can take it as a parameter (vs. the managed
    // [ThreadStatic] scratch in Attention.cs). The GPU path uses [0..numHeads*seqLen)
    // each step; the rest of the buffer stays unused.
    private readonly float[] _attentionScores;

    // Device-resident mirrors of the KV cache + scoresScratch, allocated only
    // when GpuBackend.Mode == Cuda at ctor time. With these promoted the
    // multi-MB KV cache stops crossing PCIe on every Attention call — the
    // dominant bandwidth source after the Attention port.
    private readonly FloatResidentArray[]? _keyCacheResident;
    private readonly FloatResidentArray[]? _valueCacheResident;
    private readonly FloatResidentArray? _attentionScoresResident;
    // Step 2 of activation promotion: RoPE tables are computed once in
    // BuildRopeCache() and read 22× per token (~11 MB/token of redundant
    // PCIe traffic when host-only). Resident, populated from host via
    // HostPointer bulk-copy + DeviceNeedsRefresh.
    private readonly FloatResidentArray? _ropeCosResident;
    private readonly FloatResidentArray? _ropeSinResident;
    // Step 3: FFN-block activations resident. _gate / _up written by Gate/Up
    // matvec (float[] vector bridge), modified in-place by SwiGLU, then
    // _gate read by W2 → _ffnOut, all device-side. _ffnOut bridges back to
    // host _x via AccumulateResidentB.
    private readonly FloatResidentArray? _gateResident;
    private readonly FloatResidentArray? _upResident;
    private readonly FloatResidentArray? _ffnOutResident;
    // Step 4: attention-block intermediates resident. _q/_k/_v written by
    // QKV matvecs (float[] vec bridge), rotated by ApplyRopeFullyResident,
    // _k/_v copied into the resident cache by WriteKvCacheSliceResidentSrc,
    // Attention reads resident q + caches and writes resident _attnOut, Wo
    // matvec reads resident _attnOut and writes resident _attnProj, then
    // AccumulateResidentB closes into the still-host _x.
    private readonly FloatResidentArray? _qResident;
    private readonly FloatResidentArray? _kResident;
    private readonly FloatResidentArray? _vResident;
    private readonly FloatResidentArray? _attnOutResident;
    private readonly FloatResidentArray? _attnProjResident;
    // Step 5 (final activation promotion): _x, _xNorm, _logits resident.
    // _xResident is the running hidden state, written by token embedding
    // and Accumulate, read by RmsNorm and output matvec. _xNormResident
    // is the RmsNorm output buffer feeding QKV / Gate / Up matvecs.
    // _logitsResident is the output matvec result; we RefreshHost it once
    // per Forward() and copy back into the existing _logits float[] so the
    // sampler signature stays unchanged.
    private readonly FloatResidentArray? _xResident;
    private readonly FloatResidentArray? _xNormResident;
    private readonly FloatResidentArray? _logitsResident;
    // RmsNorm bypass: each norm weight gets its own FloatResidentArray (host-
    // filled at ctor from the existing float[] in LayerWeights / LlamaWeights,
    // then DeviceNeedsRefresh-uploaded on first kernel use). _rmsSumBoxResident
    // is a persistent 1-element device buffer for the sum-of-squares reducer
    // output; we cudaMemsetAsync it to 0 before each RmsNorm reduce and the
    // fused broadcast kernel reads it on device to compute the scale —
    // zero D→H per RmsNorm since Iter 4a.
    private readonly FloatResidentArray[]? _attnNormWeightResident;
    private readonly FloatResidentArray[]? _ffnNormWeightResident;
    private readonly FloatResidentArray? _outputNormWeightResident;
    private readonly FloatResidentArray? _rmsSumBoxResident;
    // GPU-greedy sampler scratch (Iter 7.B.2). _nextTokenIdResident holds the
    // chosen token id on device; persistent so future iterations can feed it
    // directly into a device-indexed embedding lookup without H→D plumbing.
    // _argmaxMaxBoxDev is a 4-byte device float seeded to -∞ before each
    // argmax call (raw cudaMalloc — never read on host except via the dispatch).
    private readonly ResidentArrayGeneric<int>? _nextTokenIdResident;
    private readonly nint _argmaxMaxBoxDev;
    // Deferred-print ring (Iter 7.B.3b). The per-token AppendDeviceInt kernel
    // writes each chosen id into _tokenIdsRingResident at slot
    // (_tokenIdsCountResident[0] % TokenIdRingSize) and bumps the counter —
    // all on device. The host calls DrainPendingTokens periodically to copy
    // a slice back for printing / EOS detection.
    //
    // Sizing: must cover (peak tok/s) × (drain interval). With Iter 7.A's
    // 87 tok/s on TinyLlama Q8_0 and the 200 ms drain interval in Program.cs
    // that's ~18 tokens between drains; 1024 gives a 50× safety margin to
    // absorb timing jitter and any temporary host-side drain stalls (e.g. a
    // GC pause). 1024 × 4 B = 4 KB of device memory — negligible.
    private const int TokenIdRingSize = 1024;
    private readonly ResidentArrayGeneric<int>? _tokenIdsRingResident;
    private readonly ResidentArrayGeneric<int>? _tokenIdsCountResident;
    private int _lastDrainedCount;
    private readonly bool _useResidentKv;

    // Iter 7.A.7.a — device-side decode position. Lives in a 1-element
    // ResidentArrayGeneric<int>. Seeded once from the host at the boundary
    // between prompt-eval and decode; bumped each decode step by a tiny
    // BumpDeviceInt kernel device-side, so the per-token forward has no host
    // int arg whose value would lock a captured CUDA graph (iter 7.A.7.b).
    private readonly ResidentArrayGeneric<int>? _positionResident;

    // Iter 7.A.7.b — captured cudaGraph for the decode body. Populated on the
    // second ForwardArgmaxDeferred call by stream-capturing the body; reused on
    // every subsequent call via cudaGraphLaunch. Collapses ~378 per-token
    // kernel launches into one. nint.Zero means "not captured yet".
    // The first call is a non-captured warmup: it runs the body normally so
    // every ResidentStructCache entry the body touches is materialised
    // (Materialise does sync cudaMalloc+cudaMemcpy of a 32-byte device struct,
    // which aborts capture even in Relaxed mode). After warmup the cache is
    // hot and the captured body only contains kernel launches.
    private nint _decodeGraph;
    private nint _decodeGraphExec;
    private bool _decodeWarmupDone;
    // Graph capture is on by default — Windows re-measurement after the
    // cudaGraphInstantiate fix landed at 115 tok/s vs the 98 tok/s direct
    // baseline. Set LLAMA_DISABLE_GRAPH=1 to keep the non-default capture
    // stream but skip BeginCapture/Instantiate/Launch — useful for A/B
    // diagnostics and required when profiling (per-kernel sync aborts
    // capture). See porting-to-hybridizer.md for the iter 7.A.7.b history.
    private readonly bool _disableGraph =
        System.Environment.GetEnvironmentVariable("LLAMA_DISABLE_GRAPH") == "1";

    public LlamaTransformer(LlamaConfig config, LlamaWeights weights)
    {
        _config = config;
        _weights = weights;

        int kvDim = config.NumKvHeads * config.HeadDim;

        _keyCache = new float[config.NumLayers][];
        _valueCache = new float[config.NumLayers][];
        for (int l = 0; l < config.NumLayers; l++)
        {
            _keyCache[l] = new float[config.ContextLength * kvDim];
            _valueCache[l] = new float[config.ContextLength * kvDim];
        }

        _x = new float[config.EmbeddingDim];
        _xNorm = new float[config.EmbeddingDim];
        _q = new float[config.NumHeads * config.HeadDim];
        _k = new float[kvDim];
        _v = new float[kvDim];
        _attnOut = new float[config.NumHeads * config.HeadDim];
        _attnProj = new float[config.EmbeddingDim];
        _gate = new float[config.HiddenDim];
        _up = new float[config.HiddenDim];
        _ffnOut = new float[config.EmbeddingDim];
        _logits = new float[config.VocabSize];
        _attentionScores = new float[config.NumHeads * config.ContextLength];

        _ropePairCount = config.RopeDimCount / 2;
        _ropeCos = new float[config.ContextLength * _ropePairCount];
        _ropeSin = new float[config.ContextLength * _ropePairCount];
        BuildRopeCache();

        // Resident KV cache + scratch — only useful on CUDA (OMP host==device
        // so no PCIe to save; Managed obviously). Promotion is decided once at
        // ctor time based on the active GpuBackend mode; the transformer is
        // bound to one backend for its lifetime.
        _useResidentKv = GpuBackend.Mode == ComputeBackend.Cuda;
        if (_useResidentKv)
        {
            _keyCacheResident = new FloatResidentArray[config.NumLayers];
            _valueCacheResident = new FloatResidentArray[config.NumLayers];
            for (int l = 0; l < config.NumLayers; l++)
            {
                _keyCacheResident[l] = new FloatResidentArray(config.ContextLength * kvDim);
                _valueCacheResident[l] = new FloatResidentArray(config.ContextLength * kvDim);
                // Force device allocation upfront so the first kernel call has
                // a valid pointer and isn't competing for a fresh cudaMalloc
                // alongside other resident-array first-uses.
                _ = _keyCacheResident[l].DevicePointer;
                _ = _valueCacheResident[l].DevicePointer;
                // Mark as "device is the truth" so the marshaller never tries
                // to upload an empty host buffer over our kernel writes.
                _keyCacheResident[l].Status = ResidentArrayStatus.HostNeedsRefresh;
                _valueCacheResident[l].Status = ResidentArrayStatus.HostNeedsRefresh;
            }
            _attentionScoresResident = new FloatResidentArray(config.NumHeads * config.ContextLength);
            _ = _attentionScoresResident.DevicePointer;
            _attentionScoresResident.Status = ResidentArrayStatus.HostNeedsRefresh;

            // RoPE tables are host-fills-first (computed in BuildRopeCache);
            // bulk-copy via HostPointer + DeviceNeedsRefresh so the marshaller
            // does one H→D upload on the first ApplyRope call and the device
            // copy persists thereafter.
            _ropeCosResident = new FloatResidentArray(_ropeCos.Length);
            _ropeSinResident = new FloatResidentArray(_ropeSin.Length);
            unsafe
            {
                long bytes = (long)_ropeCos.Length * sizeof(float);
                fixed (float* src = _ropeCos)
                {
                    Buffer.MemoryCopy(src, (void*)_ropeCosResident.HostPointer, bytes, bytes);
                }
                fixed (float* src = _ropeSin)
                {
                    Buffer.MemoryCopy(src, (void*)_ropeSinResident.HostPointer, bytes, bytes);
                }
            }
            _ropeCosResident.Status = ResidentArrayStatus.DeviceNeedsRefresh;
            _ropeSinResident.Status = ResidentArrayStatus.DeviceNeedsRefresh;

            // FFN-block activations — written by kernels, never read on host.
            // Eager DevicePointer + HostNeedsRefresh per the resident-array
            // init pattern for write-first arrays.
            _gateResident = new FloatResidentArray(config.HiddenDim);
            _upResident = new FloatResidentArray(config.HiddenDim);
            _ffnOutResident = new FloatResidentArray(config.EmbeddingDim);
            _ = _gateResident.DevicePointer;
            _ = _upResident.DevicePointer;
            _ = _ffnOutResident.DevicePointer;
            _gateResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _upResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _ffnOutResident.Status = ResidentArrayStatus.HostNeedsRefresh;

            // Attention-block intermediates — same write-first pattern.
            _qResident = new FloatResidentArray(config.NumHeads * config.HeadDim);
            _kResident = new FloatResidentArray(kvDim);
            _vResident = new FloatResidentArray(kvDim);
            _attnOutResident = new FloatResidentArray(config.NumHeads * config.HeadDim);
            _attnProjResident = new FloatResidentArray(config.EmbeddingDim);
            _ = _qResident.DevicePointer;
            _ = _kResident.DevicePointer;
            _ = _vResident.DevicePointer;
            _ = _attnOutResident.DevicePointer;
            _ = _attnProjResident.DevicePointer;
            _qResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _kResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _vResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _attnOutResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _attnProjResident.Status = ResidentArrayStatus.HostNeedsRefresh;

            // Hidden state + logits resident (step 5). _x is written by token
            // embedding lookup (DequantizeRowFullyResident) and Accumulate,
            // read by RmsNorm and the output matvec; _xNorm cycles between
            // RmsNorm and the QKV/Gate/Up matvecs; _logits is the output of
            // the final matvec, RefreshHost'd once per Forward.
            _xResident = new FloatResidentArray(config.EmbeddingDim);
            _xNormResident = new FloatResidentArray(config.EmbeddingDim);
            _logitsResident = new FloatResidentArray(config.VocabSize);
            _ = _xResident.DevicePointer;
            _ = _xNormResident.DevicePointer;
            _ = _logitsResident.DevicePointer;
            _xResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _xNormResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _logitsResident.Status = ResidentArrayStatus.HostNeedsRefresh;

            // RmsNorm bypass: promote per-layer norm weights + a persistent
            // sumBox for the reducer (cudaMemsetAsync to 0 each call; the
            // fused broadcast kernel reads it on device — no D→H).
            _attnNormWeightResident = new FloatResidentArray[config.NumLayers];
            _ffnNormWeightResident = new FloatResidentArray[config.NumLayers];
            for (int l = 0; l < config.NumLayers; l++)
            {
                _attnNormWeightResident[l] = MakeResidentFromHost(weights.Layers[l].AttnNormWeight);
                _ffnNormWeightResident[l] = MakeResidentFromHost(weights.Layers[l].FfnNormWeight);
            }
            _outputNormWeightResident = MakeResidentFromHost(weights.OutputNormWeight);

            _rmsSumBoxResident = new FloatResidentArray(1);
            _ = _rmsSumBoxResident.DevicePointer;
            _rmsSumBoxResident.Status = ResidentArrayStatus.HostNeedsRefresh;

            // GPU greedy sampling scratch — see ForwardArgmaxGpu.
            _nextTokenIdResident = new ResidentArrayGeneric<int>(1);
            _ = _nextTokenIdResident.DevicePointer;
            _nextTokenIdResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            cuda.ERROR_CHECK(cuda.Malloc(out _argmaxMaxBoxDev, (size_t)sizeof(float)));

            // Deferred-print ring + counter (Iter 7.B.3b). Counter starts at 0
            // via a one-shot WriteOneInt — ring slot contents are don't-cares
            // until the corresponding counter value is reached.
            _tokenIdsRingResident = new ResidentArrayGeneric<int>(TokenIdRingSize);
            _tokenIdsCountResident = new ResidentArrayGeneric<int>(1);
            _ = _tokenIdsRingResident.DevicePointer;
            _ = _tokenIdsCountResident.DevicePointer;
            _tokenIdsRingResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            _tokenIdsCountResident.Status = ResidentArrayStatus.HostNeedsRefresh;
            CudaInvoke.WriteOneInt(_tokenIdsCountResident, 0);
            _lastDrainedCount = 0;

            // Decode-position counter — seeded later via SeedDecodePosition
            // when the host knows promptTokens.Count, then bumped device-side
            // each ForwardArgmaxDeferred step.
            _positionResident = new ResidentArrayGeneric<int>(1);
            _ = _positionResident.DevicePointer;
            _positionResident.Status = ResidentArrayStatus.HostNeedsRefresh;
        }
    }

    /// <summary>
    /// Bulk-copy a host <c>float[]</c> (norm weight) into a freshly allocated
    /// <see cref="FloatResidentArray"/>. The DeviceNeedsRefresh status causes
    /// the first kernel use to trigger the H→D upload, after which the device
    /// buffer is the source of truth for the rest of the run.
    /// </summary>
    private static unsafe FloatResidentArray MakeResidentFromHost(float[] hostWeight)
    {
        var res = new FloatResidentArray(hostWeight.Length);
        long bytes = (long)hostWeight.Length * sizeof(float);
        fixed (float* src = hostWeight)
        {
            Buffer.MemoryCopy(src, (void*)res.HostPointer, bytes, bytes);
        }
        // Explicit RefreshDevice — we'll pass the raw DevicePointer to kernels
        // (bypassing ResidentStructCache.Get which would otherwise trigger the
        // upload). One H→D copy at ctor time, then the device buffer is the
        // source of truth.
        res.Status = ResidentArrayStatus.DeviceNeedsRefresh;
        res.RefreshDevice();
        return res;
    }

    /// <summary>
    /// Run one forward pass for a single token at the given position.
    /// Returns logits over the vocabulary.
    /// </summary>
    public float[] Forward(int token, int position)
    {
        if (_useResidentKv)
            return ForwardResident(token, position);
        return ForwardHost(token, position);
    }

    /// <summary>
    /// CUDA-resident forward pass: every activation lives on the device for
    /// the entire layer loop; only the final logits are <c>RefreshHost</c>'d
    /// and copied into the public <c>_logits</c> float[] so the sampler can
    /// read them with no API change.
    /// </summary>
    private unsafe float[] ForwardResident(int token, int position)
    {
        ForwardResidentCore(token, position);

        // Single D→H copy per forward — sampler reads the public _logits array.
        _logitsResident!.RefreshHost();
        long bytes = (long)_logits.Length * sizeof(float);
        fixed (float* dst = _logits)
        {
            Buffer.MemoryCopy((void*)_logitsResident!.HostPointer, dst, bytes, bytes);
        }
        return _logits;
    }

    /// <summary>
    /// CUDA-resident forward pass with on-device greedy argmax. Same layer
    /// loop as <see cref="ForwardResident"/> but instead of the 125 KB D→H
    /// copy of <c>_logitsResident</c>, runs <see cref="CudaInvoke.ArgmaxFullyResident"/>
    /// on device and reads the chosen token id back via one 4-byte
    /// <c>cudaMemcpy</c>. Caller-side: skip <see cref="Sampling.Sampler.Sample"/>
    /// and use the returned int directly (only valid for greedy / temperature
    /// 0 sampling; stochastic sampling still goes through the host
    /// <c>Forward</c> path until top-p is ported to GPU).
    /// </summary>
    public unsafe int ForwardArgmaxGpu(int token, int position)
    {
        if (!_useResidentKv)
            throw new System.InvalidOperationException("ForwardArgmaxGpu requires the CUDA-resident path (GpuBackend.Mode == Cuda at construction).");

        ForwardResidentCore(token, position);
        return ArgmaxAndRead();
    }

    /// <summary>
    /// CUDA-resident forward pass for prompt prefill tokens where the
    /// caller doesn't need logits — only the KV cache update matters.
    /// Skips the 125 KB <c>_logitsResident</c> D→H tail of
    /// <see cref="Forward"/>. Used by <see cref="Program"/>'s GPU-greedy
    /// prefill loop for tokens 0..N−2 of the prompt; the last prompt token
    /// uses <see cref="ForwardArgmaxGpu"/> to actually sample the first
    /// generated id. Result: zero D→H anywhere on the GPU-greedy path.
    /// </summary>
    public void ForwardPrefillNoLogits(int token, int position)
    {
        if (!_useResidentKv)
            throw new System.InvalidOperationException("ForwardPrefillNoLogits requires the CUDA-resident path (GpuBackend.Mode == Cuda at construction).");

        ForwardResidentCore(token, position);
    }

    /// <summary>
    /// CUDA-resident greedy forward whose token id is read from the
    /// persistent device slot <c>_nextTokenIdResident</c> (written by the
    /// previous step's argmax). Eliminates the H→D plumbing of the chosen
    /// token id between consecutive Forwards. The 4-byte D→H of the new id
    /// at the end of the call is preserved here for the immediate-print
    /// path; Iter 3b's <c>ForwardArgmaxDeferred</c> drops it entirely.
    /// Caller must have seeded <see cref="SeedNextTokenIdFromHost"/> with
    /// the first generation token before the first call.
    /// </summary>
    public unsafe int ForwardArgmaxGpuFromDevice(int position)
    {
        if (!_useResidentKv)
            throw new System.InvalidOperationException("ForwardArgmaxGpuFromDevice requires the CUDA-resident path (GpuBackend.Mode == Cuda at construction).");

        // Token embedding lookup reads the row index from device memory.
        _weights.TokenEmbedding.CopyRowToResidentByDeviceIdx(_xResident!, _nextTokenIdResident!);
        ForwardResidentCoreFromEmbedding(position);
        return ArgmaxAndRead();
    }

    /// <summary>
    /// Seed <c>_nextTokenIdResident[0]</c> with a host-side int. Used once
    /// at the start of GPU-fed generation, after the prompt prefill has
    /// produced the first chosen token on the host side.
    /// </summary>
    public unsafe void SeedNextTokenIdFromHost(int tokenId)
    {
        if (!_useResidentKv)
            throw new System.InvalidOperationException("SeedNextTokenIdFromHost requires the CUDA-resident path (GpuBackend.Mode == Cuda at construction).");
        cuda.ERROR_CHECK(cuda.Memcpy(_nextTokenIdResident!.DevicePointer, (nint)(&tokenId), (size_t)sizeof(int), cudaMemcpyKind.cudaMemcpyHostToDevice));
    }

    /// <summary>
    /// Fully-deferred CUDA-resident greedy forward (Iter 7.B.3b). Reads the
    /// prior token from <c>_nextTokenIdResident</c>, runs the layer loop,
    /// runs argmax on device, and appends the chosen id to the device-side
    /// ring buffer via <see cref="CudaInvoke.AppendDeviceInt"/> — zero D→H
    /// per call. The host learns about generated tokens by calling
    /// <see cref="DrainPendingTokens"/> on a timer.
    ///
    /// Iter 7.A.7.a: the decode position now lives in
    /// <c>_positionResident</c> on device. Caller seeds it once via
    /// <see cref="SeedDecodePosition"/> before the first call; this method
    /// bumps it device-side at the end of each forward. No host int args —
    /// makes the body CUDA-graph-capture-friendly for iter 7.A.7.b.
    /// </summary>
    public unsafe void ForwardArgmaxDeferred()
    {
        if (!_useResidentKv)
            throw new System.InvalidOperationException("ForwardArgmaxDeferred requires the CUDA-resident path (GpuBackend.Mode == Cuda at construction).");

        // Default path is graph capture + replay (115 tok/s on Windows).
        // Fall back to per-kernel direct dispatch on the same non-default
        // stream when profiling (per-kernel cudaDeviceSynchronize aborts
        // capture) or when LLAMA_DISABLE_GRAPH=1 is set for A/B diagnostics.
        if (_disableGraph || CudaInvoke.ProfilingEnabled)
        {
            ForwardArgmaxDeferredBody();
            return;
        }

        if (!_decodeWarmupDone)
        {
            ForwardArgmaxDeferredBody();
            _decodeWarmupDone = true;
            return;
        }

        if (_decodeGraphExec == nint.Zero)
        {
            // Drain any pending work on the capture stream and wait for the
            // host side of Materialise to settle. After this we know the
            // ResidentStructCache is fully hot and the kernels' inputs are
            // initialised on device — Begin/EndCapture won't trip a legacy
            // default stream sync inside the captured region.
            GraphInvoke.SynchronizeStream(CudaInvoke.Stream);

            GraphInvoke.BeginCapture(CudaInvoke.Stream);
            ForwardArgmaxDeferredBody();
            _decodeGraph = GraphInvoke.EndCapture(CudaInvoke.Stream);
            _decodeGraphExec = GraphInvoke.Instantiate(_decodeGraph);
            GraphInvoke.Launch(_decodeGraphExec, CudaInvoke.Stream);
        }
        else
        {
            GraphInvoke.Launch(_decodeGraphExec, CudaInvoke.Stream);
        }
    }

    /// <summary>
    /// The recordable body of a deferred decode step — embedding lookup
    /// through to the device-position bump. Kept private and free of any
    /// synchronous CUDA call so it can sit inside a stream-capture region.
    /// </summary>
    private void ForwardArgmaxDeferredBody()
    {
        _weights.TokenEmbedding.CopyRowToResidentByDeviceIdx(_xResident!, _nextTokenIdResident!);
        ForwardResidentCoreFromEmbeddingDev();
        ArgmaxToDevice();
        CudaInvoke.AppendDeviceInt(_tokenIdsRingResident!, _tokenIdsCountResident!, _nextTokenIdResident!, TokenIdRingSize);
        CudaInvoke.BumpDeviceInt(_positionResident!);
    }

    /// <summary>
    /// Seed the device-side decode-position counter with the current host
    /// position. Called once before entering the deferred-decode loop, with
    /// <c>promptTokens.Count</c>. Per-token increments happen device-side
    /// inside <see cref="ForwardArgmaxDeferred"/>.
    /// </summary>
    public unsafe void SeedDecodePosition(int position)
    {
        if (!_useResidentKv)
            throw new System.InvalidOperationException("SeedDecodePosition requires the CUDA-resident path (GpuBackend.Mode == Cuda at construction).");
        cuda.ERROR_CHECK(cuda.Memcpy(_positionResident!.DevicePointer, (nint)(&position), (size_t)sizeof(int), cudaMemcpyKind.cudaMemcpyHostToDevice));
    }

    /// <summary>
    /// Drain freshly generated tokens from the device ring into
    /// <paramref name="dst"/>. Returns the number of tokens drained on this
    /// call. The first 4-byte D→H reads the producer counter; if it advanced
    /// since the last drain, a second D→H of the new ring slice follows.
    /// Drain frequency is the caller's choice (production: 500 ms timer;
    /// tests: a single final flush after the generation loop).
    /// </summary>
    public unsafe int DrainPendingTokens(int[] dst)
    {
        if (!_useResidentKv)
            throw new System.InvalidOperationException("DrainPendingTokens requires the CUDA-resident path (GpuBackend.Mode == Cuda at construction).");

        int produced = 0;
        cuda.ERROR_CHECK(cuda.Memcpy((nint)(&produced), _tokenIdsCountResident!.DevicePointer, (size_t)sizeof(int), cudaMemcpyKind.cudaMemcpyDeviceToHost));

        int newCount = produced - _lastDrainedCount;
        if (newCount <= 0)
            return 0;
        // The ring buffer is small (TokenIdRingSize); if we fell behind by more
        // than its capacity we'd be overwriting unread tokens. Detect and
        // surface so a slow drain caller can be tuned.
        if (newCount > TokenIdRingSize)
            throw new System.InvalidOperationException(
                $"Deferred token ring overflow: produced {produced} but last drained {_lastDrainedCount} ({newCount} pending > ring size {TokenIdRingSize}). Drain more frequently.");
        if (newCount > dst.Length)
            throw new System.ArgumentException(
                $"Drain destination too small: need {newCount} entries, got {dst.Length}.", nameof(dst));

        // Copy the contiguous tail; the ring may wrap.
        int start = _lastDrainedCount % TokenIdRingSize;
        int firstChunk = System.Math.Min(newCount, TokenIdRingSize - start);
        fixed (int* dstPin = dst)
        {
            cuda.ERROR_CHECK(cuda.Memcpy(
                (nint)dstPin,
                _tokenIdsRingResident!.DevicePointer + (nint)(start * sizeof(int)),
                (size_t)(firstChunk * sizeof(int)),
                cudaMemcpyKind.cudaMemcpyDeviceToHost));
            if (firstChunk < newCount)
            {
                cuda.ERROR_CHECK(cuda.Memcpy(
                    (nint)(dstPin + firstChunk),
                    _tokenIdsRingResident.DevicePointer,
                    (size_t)((newCount - firstChunk) * sizeof(int)),
                    cudaMemcpyKind.cudaMemcpyDeviceToHost));
            }
        }

        _lastDrainedCount = produced;
        return newCount;
    }

    /// <summary>
    /// Reset the device-side ring counter back to 0 and the host-side
    /// drained-count tracker. Used by tests that drive the deferred path
    /// from a fresh transformer that has already gone through the CUDA
    /// activation ctor (which writes 0 anyway, but tests reuse a transformer
    /// across multiple decode runs).
    /// </summary>
    public void ResetDeferredTokenRing()
    {
        if (!_useResidentKv)
            return;
        CudaInvoke.WriteOneInt(_tokenIdsCountResident!, 0);
        _lastDrainedCount = 0;
    }

    /// <summary>
    /// Run argmax on the logits and leave the chosen id in
    /// <c>_nextTokenIdResident[0]</c> on device. No D→H, no H→D — the
    /// argmax dispatcher's init kernel seeds the scratch buffers on device.
    /// </summary>
    private void ArgmaxToDevice()
    {
        CudaInvoke.ArgmaxFullyResident(_logitsResident!, _logits.Length, _argmaxMaxBoxDev, _nextTokenIdResident!.DevicePointer);
    }

    /// <summary>
    /// Shared argmax tail: seed boxes, run the two-phase argmax dispatcher,
    /// read the chosen token id back via 4-byte D→H. Returns the id.
    /// </summary>
    private unsafe int ArgmaxAndRead()
    {
        ArgmaxToDevice();
        int chosen = 0;
        cuda.ERROR_CHECK(cuda.Memcpy((nint)(&chosen), _nextTokenIdResident!.DevicePointer, (size_t)sizeof(int), cudaMemcpyKind.cudaMemcpyDeviceToHost));
        return chosen;
    }

    /// <summary>
    /// The shared body of the CUDA-resident forward pass — runs the full
    /// layer loop and the final RmsNorm + output matvec, leaving the result
    /// in <c>_logitsResident</c> on device. The tail (host readback vs. GPU
    /// argmax) is the caller's responsibility.
    /// </summary>
    private void ForwardResidentCore(int token, int position)
    {
        _weights.TokenEmbedding.CopyRowToResident(_xResident!, token);
        ForwardResidentCoreFromEmbedding(position);
    }

    /// <summary>
    /// Variant of <see cref="ForwardResidentCore"/> that assumes the token
    /// embedding has already been written into <c>_xResident</c> (either by
    /// <see cref="LlamaWeights.WeightMatrix.CopyRowToResident"/> with a host
    /// int or by <see cref="LlamaWeights.WeightMatrix.CopyRowToResidentByDeviceIdx"/>
    /// reading from a device int slot). Lets the GPU-greedy device-fed path
    /// share the entire layer loop with the host-fed path.
    /// </summary>
    private void ForwardResidentCoreFromEmbedding(int position)
    {
        int dim = _config.EmbeddingDim;
        int headDim = _config.HeadDim;
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int kvDim = numKvHeads * headDim;

        for (int l = 0; l < _config.NumLayers; l++)
        {
            var layer = _weights.Layers[l];

            GpuBackend.RmsNormFastResident(_xNormResident!, _xResident!, _attnNormWeightResident![l], _rmsSumBoxResident!, dim, _config.RmsNormEps);

            int ropeOffset = position * _ropePairCount;
            int seqLen = position + 1;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // QKV matvec — fully resident.
            layer.Wq.MatVecMul(_qResident!, _xNormResident!);
            layer.Wk.MatVecMul(_kResident!, _xNormResident!);
            layer.Wv.MatVecMul(_vResident!, _xNormResident!);

            GpuBackend.ApplyRopeFullyResident(
                _qResident!, _kResident!, headDim, numHeads, numKvHeads,
                _ropeCosResident!, _ropeSinResident!,
                ropeOffset, _ropePairCount);

            GpuBackend.WriteKvCacheSliceResidentSrc(_kResident!, _keyCacheResident![l], position * kvDim, kvDim);
            GpuBackend.WriteKvCacheSliceResidentSrc(_vResident!, _valueCacheResident![l], position * kvDim, kvDim);

            GpuBackend.AttentionForwardOneTokenFullyResident(
                _attnOutResident!, _qResident!,
                _keyCacheResident[l], _valueCacheResident[l],
                _attentionScoresResident!,
                seqLen, numHeads, numKvHeads, headDim, scale);

            layer.Wo.MatVecMul(_attnProjResident!, _attnOutResident!);
            GpuBackend.AccumulateFullyResident(_xResident!, _attnProjResident!, dim);

            GpuBackend.RmsNormFastResident(_xNormResident!, _xResident!, _ffnNormWeightResident![l], _rmsSumBoxResident!, dim, _config.RmsNormEps);

            // FFN block — fully resident.
            layer.W1.MatVecMul(_gateResident!, _xNormResident!);
            layer.W3.MatVecMul(_upResident!, _xNormResident!);
            GpuBackend.FusedSiluElementWiseMulFullyResident(_gateResident!, _upResident!, _config.HiddenDim);
            layer.W2.MatVecMul(_ffnOutResident!, _gateResident!);
            GpuBackend.AccumulateFullyResident(_xResident!, _ffnOutResident!, dim);
        }

        // Final RmsNorm in place; output matvec writes _logitsResident.
        GpuBackend.RmsNormFastResident(_xResident!, _xResident!, _outputNormWeightResident!, _rmsSumBoxResident!, dim, _config.RmsNormEps);
        _weights.OutputWeight.MatVecMul(_logitsResident!, _xResident!);
    }

    /// <summary>
    /// Iter 7.A.7.a device-position variant of <see cref="ForwardResidentCoreFromEmbedding"/>.
    /// Same body, but RoPE / KV-cache write / attention all read the current
    /// decode position from <c>_positionResident</c> on device. No host int
    /// parameters in the call site — required for the graph-capture path in
    /// iter 7.A.7.b. Used only by the deferred-decode loop (prompt-eval still
    /// uses the host-int variant for now).
    /// </summary>
    private void ForwardResidentCoreFromEmbeddingDev()
    {
        int dim = _config.EmbeddingDim;
        int headDim = _config.HeadDim;
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int kvDim = numKvHeads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);

        for (int l = 0; l < _config.NumLayers; l++)
        {
            var layer = _weights.Layers[l];

            GpuBackend.RmsNormFastResident(_xNormResident!, _xResident!, _attnNormWeightResident![l], _rmsSumBoxResident!, dim, _config.RmsNormEps);

            layer.Wq.MatVecMul(_qResident!, _xNormResident!);
            layer.Wk.MatVecMul(_kResident!, _xNormResident!);
            layer.Wv.MatVecMul(_vResident!, _xNormResident!);

            GpuBackend.ApplyRopeFullyResidentDev(
                _qResident!, _kResident!, headDim, numHeads, numKvHeads,
                _ropeCosResident!, _ropeSinResident!,
                _positionResident!, _ropePairCount);

            GpuBackend.WriteKvCacheSliceResidentSrcDev(_kResident!, _keyCacheResident![l], _positionResident!, kvDim);
            GpuBackend.WriteKvCacheSliceResidentSrcDev(_vResident!, _valueCacheResident![l], _positionResident!, kvDim);

            GpuBackend.AttentionForwardOneTokenFullyResidentDev(
                _attnOutResident!, _qResident!,
                _keyCacheResident[l], _valueCacheResident[l],
                _attentionScoresResident!,
                _positionResident!, numHeads, numKvHeads, headDim, scale);

            layer.Wo.MatVecMul(_attnProjResident!, _attnOutResident!);
            GpuBackend.AccumulateFullyResident(_xResident!, _attnProjResident!, dim);

            GpuBackend.RmsNormFastResident(_xNormResident!, _xResident!, _ffnNormWeightResident![l], _rmsSumBoxResident!, dim, _config.RmsNormEps);

            layer.W1.MatVecMul(_gateResident!, _xNormResident!);
            layer.W3.MatVecMul(_upResident!, _xNormResident!);
            GpuBackend.FusedSiluElementWiseMulFullyResident(_gateResident!, _upResident!, _config.HiddenDim);
            layer.W2.MatVecMul(_ffnOutResident!, _gateResident!);
            GpuBackend.AccumulateFullyResident(_xResident!, _ffnOutResident!, dim);
        }

        GpuBackend.RmsNormFastResident(_xResident!, _xResident!, _outputNormWeightResident!, _rmsSumBoxResident!, dim, _config.RmsNormEps);
        _weights.OutputWeight.MatVecMul(_logitsResident!, _xResident!);
    }

    /// <summary>
    /// Host-buffer forward pass — used for Managed and OMP backends. Same
    /// shape as the resident path but every activation lives in managed
    /// <see cref="float"/>[] arrays.
    /// </summary>
    private float[] ForwardHost(int token, int position)
    {
        int dim = _config.EmbeddingDim;
        int headDim = _config.HeadDim;
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int kvDim = numKvHeads * headDim;

        _weights.TokenEmbedding.CopyRowTo(_x, token);

        for (int l = 0; l < _config.NumLayers; l++)
        {
            var layer = _weights.Layers[l];

            GpuBackend.RmsNorm(_xNorm, _x, layer.AttnNormWeight, dim, _config.RmsNormEps);

            WeightMatrix.FusedMatVecMul3(
                _q, layer.Wq,
                _k, layer.Wk,
                _v, layer.Wv,
                _xNorm);

            int ropeOffset = position * _ropePairCount;
            GpuBackend.ApplyRope(
                _q, _k, headDim, numHeads, numKvHeads,
                _ropeCos, _ropeSin, ropeOffset, _ropePairCount);

            Array.Copy(_k, 0, _keyCache[l], position * kvDim, kvDim);
            Array.Copy(_v, 0, _valueCache[l], position * kvDim, kvDim);

            int seqLen = position + 1;
            float scale = 1.0f / MathF.Sqrt(headDim);

            GpuBackend.AttentionForwardOneToken(
                _attnOut, _q,
                _keyCache[l], _valueCache[l],
                _attentionScores,
                seqLen, numHeads, numKvHeads, headDim, scale);

            layer.Wo.MatVecMul(_attnProj, _attnOut);
            GpuBackend.Accumulate(_x, _attnProj, dim);

            GpuBackend.RmsNorm(_xNorm, _x, layer.FfnNormWeight, dim, _config.RmsNormEps);

            WeightMatrix.FusedMatVecMul2(
                _gate, layer.W1,
                _up, layer.W3,
                _xNorm);

            GpuBackend.FusedSiluElementWiseMul(_gate, _up, _config.HiddenDim);
            layer.W2.MatVecMul(_ffnOut, _gate);
            GpuBackend.Accumulate(_x, _ffnOut, dim);
        }

        GpuBackend.RmsNorm(_x, _x, _weights.OutputNormWeight, dim, _config.RmsNormEps);
        _weights.OutputWeight.MatVecMul(_logits, _x);

        return _logits;
    }

    private void BuildRopeCache()
    {
        for (int pos = 0; pos < _config.ContextLength; pos++)
        {
            int offset = pos * _ropePairCount;
            for (int pair = 0; pair < _ropePairCount; pair++)
            {
                int i = pair * 2;
                float freq = 1.0f / MathF.Pow(_config.RopeFreqBase, (float)i / _config.RopeDimCount);
                float angle = pos * freq;
                _ropeCos[offset + pair] = MathF.Cos(angle);
                _ropeSin[offset + pair] = MathF.Sin(angle);
            }
        }
    }

}
