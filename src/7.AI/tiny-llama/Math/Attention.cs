namespace LlamaCsharp.Math;

/// <summary>
/// Per-token attention forward pass (one transformer layer).
/// </summary>
/// <remarks>
/// Body shape — per query head <c>h</c>:
///   1. scores[t] = (Q_h · K_{t, kvHead(h)}) * scale,  for t in [0, seqLen)
///   2. softmax(scores, seqLen)
///   3. attnOut[h*headDim + d] = sum_t(scores[t] * V_{t, kvHead(h), d})
///
/// Lifted out of <see cref="LlamaCsharp.Model.LlamaTransformer.Forward"/> so it has a
/// single named entry point that can later carry kernel attributes when ported.
/// The scratch scores buffer is <see cref="ThreadStaticAttribute"/> — one buffer per
/// worker thread, lazily sized to <c>seqLen</c>. On GPU, this scratch will become
/// shared memory or per-thread registers, so the [ThreadStatic] is irrelevant there.
/// </remarks>
public static class Attention
{
    [ThreadStatic]
    private static float[]? s_scoresBuffer;

    private static float[] GetScoresBuffer(int needed)
    {
        var buf = s_scoresBuffer;
        if (buf == null || buf.Length < needed)
        {
            buf = new float[needed];
            s_scoresBuffer = buf;
        }
        return buf;
    }

    /// <summary>
    /// Compute one attention forward pass for a single decoding step using the
    /// per-layer key/value caches up to <paramref name="seqLen"/> (inclusive of the
    /// just-written current position). Supports Grouped Query Attention.
    /// </summary>
    /// <param name="attnOut">Output buffer [numHeads * headDim]; overwritten.</param>
    /// <param name="q">Query vector [numHeads * headDim].</param>
    /// <param name="keyCache">Key cache for the current layer [contextLen * kvDim].</param>
    /// <param name="valueCache">Value cache for the current layer [contextLen * kvDim].</param>
    /// <param name="seqLen">Number of valid time steps in the caches (position + 1).</param>
    /// <param name="numHeads">Number of query heads.</param>
    /// <param name="numKvHeads">Number of KV heads (numHeads / numKvHeads = GQA group size).</param>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="scale">Attention scale (1 / sqrt(headDim)).</param>
    public static void AttentionForwardOneToken(
        float[] attnOut,
        float[] q,
        float[] keyCache,
        float[] valueCache,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        int kvDim = numKvHeads * headDim;
        int gqaGroupSize = numHeads / numKvHeads;

        Parallel.For(0, numHeads, h =>
        {
            float[] scores = GetScoresBuffer(seqLen);
            int kvHead = h / gqaGroupSize;
            int qOffset = h * headDim;

            for (int t = 0; t < seqLen; t++)
            {
                int kCacheOffset = t * kvDim + kvHead * headDim;
                scores[t] = ParallelMath.DotProductSimd(q, qOffset, keyCache, kCacheOffset, headDim) * scale;
            }

            ParallelMath.Softmax(scores, offset: 0, size: seqLen);

            Array.Clear(attnOut, qOffset, headDim);
            for (int t = 0; t < seqLen; t++)
            {
                float score = scores[t];
                if (score < 1e-8f)
                    continue;

                int vCacheOffset = t * kvDim + kvHead * headDim;
                for (int d = 0; d < headDim; d++)
                    attnOut[qOffset + d] += score * valueCache[vCacheOffset + d];
            }
        });
    }
}
