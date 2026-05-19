using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Math;

/// <summary>
/// Embarrassingly-parallel float-tensor kernels used in the LLaMA forward pass.
/// Each <c>[EntryPoint]</c> is a single <see cref="Parallel.For"/> that transcodes
/// to both OMP (#pragma omp parallel for) and CUDA (grid-distributed) without
/// changes.
///
/// Activations are still plain <c>float[]</c> at this point — promoting them to
/// <see cref="FloatResidentArray"/> happens in a later iteration once *all* the
/// forward-pass ops have GPU kernels (otherwise managed code between two kernel
/// calls would force a D→H sync back to a host buffer anyway).
/// </summary>
public class FloatKernels
{
    /// <summary>In-place accumulation: a[i] += b[i].</summary>
    [EntryPoint]
    public static void Accumulate(float[] a, [In] float[] b, int size)
    {
        Parallel.For(0, size, i => { a[i] += b[i]; });
    }

    /// <summary>
    /// In-place fused SwiGLU: gate[i] = SiLU(gate[i]) * up[i].
    /// Uses <c>(float)Math.Exp(double)</c> rather than <c>MathF.Exp(float)</c>
    /// because the standalone Hybridizer builtins only map the <c>double</c>
    /// overload to native <c>exp</c>; the single-precision version would abort
    /// transcoding with <c>0X60AC: Cannot get IL for method Exp</c>.
    /// </summary>
    [EntryPoint]
    public static void FusedSiluElementWiseMul(float[] gate, [In] float[] up, int size)
    {
        Parallel.For(0, size, i =>
        {
            float val = gate[i];
            gate[i] = val * (1.0f / (1.0f + (float)System.Math.Exp(-val))) * up[i];
        });
    }

    /// <summary>
    /// Sum-of-squares reduction over <paramref name="input"/>[0..size) — CUDA flavor.
    /// Writes to <c>sumOut[0]</c>; caller must zero before the call.
    /// Uses <see cref="Atomics.Add"/> (mapped to native <c>atomicAdd</c>) instead of
    /// <c>AtomicExpr.apply</c> — the maintainer-flagged buggy lambda-driven path —
    /// so the cross-thread combine is the direct intrinsic call. OMP cannot use
    /// the same body (<c>atomicAdd</c> there is a non-atomic macro), hence the
    /// separate <see cref="RmsNormSumSquaresOmp"/> sibling.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void RmsNormSumSquaresCuda(int size, [In] float[] input, float[] sumOut)
    {
        Parallel.For(0, size, i =>
        {
            float v = input[i];
            Atomics.Add(ref sumOut[0], v * v);
        });
    }

    /// <summary>
    /// Sum-of-squares reduction over <paramref name="input"/>[0..size) — OMP flavor.
    /// Sequential single-thread sum: at LLaMA scale (size ≤ 2048) the per-call
    /// cost is &lt; 2 µs on a modern CPU, well below the launch+marshal overhead
    /// of any parallel reduction scheme. Will be revisited at step 8 if profiling
    /// shows it matters.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("CUDA")]
    public static void RmsNormSumSquaresOmp(int size, [In] float[] input, float[] sumOut)
    {
        float sum = 0f;
        for (int i = 0; i < size; i++)
        {
            float v = input[i];
            sum += v * v;
        }
        sumOut[0] = sum;
    }

    // ====================================================================
    // Softmax — three-phase: find max, exp + sum, normalize.
    // Used by the LM-head sampler path when temperature > 0 and (next step)
    // by per-head Attention softmax once Attention is ported. Kept as
    // standalone entry points so the dispatcher in GpuBackend can wire them
    // up where needed.
    // ====================================================================

    /// <summary>Find max value across <paramref name="input"/>[0..size) — CUDA flavor.</summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void SoftmaxFindMaxCuda(int size, [In] float[] input, float[] maxOut)
    {
        Parallel.For(0, size, i =>
        {
            Atomics.Max(ref maxOut[0], input[i]);
        });
    }

    /// <summary>Find max value across <paramref name="input"/>[0..size) — OMP sequential.</summary>
    [EntryPoint]
    [HybridizerIgnore("CUDA")]
    public static void SoftmaxFindMaxOmp(int size, [In] float[] input, float[] maxOut)
    {
        float max = input[0];
        for (int i = 1; i < size; i++)
        {
            float v = input[i];
            if (v > max) max = v;
        }
        maxOut[0] = max;
    }

    /// <summary>
    /// In-place exp(x - max) + atomic sum — CUDA flavor.
    /// Writes <c>x[i] = exp(x[i] - max)</c> and accumulates <c>sumOut[0]</c>.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void SoftmaxExpAndSumCuda(int size, float[] x, float max, float[] sumOut)
    {
        Parallel.For(0, size, i =>
        {
            float ex = (float)System.Math.Exp(x[i] - max);
            x[i] = ex;
            Atomics.Add(ref sumOut[0], ex);
        });
    }

    /// <summary>
    /// In-place exp(x - max) + sum — OMP flavor.
    /// The OMP ExternC wrapper is <c>#pragma omp parallel { method(...); }</c>
    /// so every OpenMP thread runs the kernel body in full. A pure sequential
    /// <c>for</c> that mutates <c>x[i]</c> would therefore race on the writes
    /// (multiple threads stomping on each index). Split into two phases:
    /// (1) a <c>Parallel.For</c> for the in-place exp (lowered to
    /// <c>#pragma omp parallel for</c> — iterations distributed across the
    /// threads, no overlap); then (2) a sequential sum over the now-stable
    /// <c>x[]</c> writing <c>sumOut[0]</c> — race-but-idempotent since every
    /// thread computes the same total.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("CUDA")]
    public static void SoftmaxExpAndSumOmp(int size, float[] x, float max, float[] sumOut)
    {
        Parallel.For(0, size, i =>
        {
            x[i] = (float)System.Math.Exp(x[i] - max);
        });

        float sum = 0f;
        for (int i = 0; i < size; i++)
            sum += x[i];
        sumOut[0] = sum;
    }

    /// <summary>
    /// In-place softmax normalize: x[i] *= invSum (caller passes
    /// <c>invSum = 1 / sum</c> computed on the host between phases).
    /// Embarrassingly parallel — one body for both flavors.
    /// </summary>
    [EntryPoint]
    public static void SoftmaxNormalize(int size, float[] x, float invSum)
    {
        Parallel.For(0, size, i =>
        {
            x[i] *= invSum;
        });
    }

    // ====================================================================
    // Argmax — two-phase reduction over a resident vector. Used by the GPU
    // greedy sampler. Phase 1 finds the max value via Atomics.Max; phase 2
    // selects the smallest index among entries equal to the max via
    // Atomics.Min. Two phases (rather than one max-with-index reduction)
    // because CUDA doesn't natively pack (float, int) into a single atomic;
    // a single-kernel block-reduction is a step-7.A optimization target.
    //
    // Both kernels take the float / int scratch boxes as raw arrays so the
    // CudaInvoke bypass can pass device pointers directly (same pattern as
    // RmsNorm sumBox / norm weight). Caller seeds maxBox to -inf and idxOut
    // to int.MaxValue via cuda.Memcpy(H→D) before each call.
    // ====================================================================

    /// <summary>
    /// Phase 1 of resident argmax: atomic-max over <c>x[0..n)</c> writing
    /// <paramref name="maxOut"/>[0]. Caller seeds <c>maxOut[0] = -∞</c>.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void ArgmaxFindMaxResident(int n, FloatResidentArray x, float[] maxOut)
    {
        Parallel.For(0, n, i =>
        {
            Atomics.Max(ref maxOut[0], x[i]);
        });
    }

    /// <summary>
    /// Phase 2 of resident argmax: smallest index <c>i</c> such that
    /// <c>x[i] == maxVal</c>, written into <paramref name="idxOut"/>[0] via
    /// <see cref="Atomics.Min"/>. Caller seeds <c>idxOut[0] = int.MaxValue</c>.
    /// Tie-breaking matches the host reference <c>ParallelMath.Argmax</c>
    /// which returns the first maximum in a left-to-right scan.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void ArgmaxFindFirstIndexResident(int n, FloatResidentArray x, float maxVal, int[] idxOut)
    {
        Parallel.For(0, n, i =>
        {
            if (x[i] == maxVal)
            {
                Atomics.Min(ref idxOut[0], i);
            }
        });
    }

    /// <summary>
    /// Device-only seed for the two-phase argmax: writes
    /// <c>maxBox[0] = -FLT_MAX</c> (smaller than any logit at any
    /// quantization, safe sentinel) and <c>idxOut[0] = int.MaxValue</c>.
    /// Single-thread <c>Parallel.For(0, 1, ...)</c>. Replaces the previous
    /// 2 × <c>cudaMemcpy(H→D, 4 B)</c> per argmax call.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void ArgmaxInitSeeds(float[] maxBox, int[] idxOut)
    {
        Parallel.For(0, 1, _ =>
        {
            maxBox[0] = -3.4028235e+38f;
            idxOut[0] = 2147483647;
        });
    }

    /// <summary>
    /// Phase 2 variant that reads the max value from <c>maxBox[0]</c> on
    /// the device — removes the 4-byte <c>cudaMemcpy(D→H)</c> the original
    /// <see cref="ArgmaxFindFirstIndexResident"/> needed to receive the
    /// scalar between phases.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void ArgmaxFindFirstIndexFromMaxBox(int n, FloatResidentArray x, float[] maxBox, int[] idxOut)
    {
        Parallel.For(0, n, i =>
        {
            if (x[i] == maxBox[0])
            {
                Atomics.Min(ref idxOut[0], i);
            }
        });
    }

    /// <summary>
    /// RmsNorm scale broadcast: output[i] = input[i] * scale * weight[i].
    /// Pure embarrassingly parallel — the scale factor is computed on the host
    /// between the two kernel launches (so <c>(float)Math.Sqrt</c> stays out
    /// of the kernel body).
    /// </summary>
    [EntryPoint]
    public static void RmsNormBroadcast(
        [Out] float[] output,
        [In] float[] input,
        [In] float[] weight,
        int size,
        float scale)
    {
        Parallel.For(0, size, i =>
        {
            output[i] = input[i] * scale * weight[i];
        });
    }

    // ====================================================================
    // KV cache writer — copies a per-token float[] slice (q, k, or v written
    // by the matvec) into a position-indexed slot in a device-resident cache.
    // One element per Parallel.For iteration; embarrassingly parallel.
    // The float[] src is still marshalled H→D each call (1 KB at TinyLlama
    // scale, negligible), but the resident dst stays on the device — so the
    // big read-side bandwidth for the cache is paid only on the *first* token,
    // not every token.
    // ====================================================================
    [EntryPoint]
    public static void WriteKvCacheSlice(
        [In] float[] src,
        FloatResidentArray dst,
        int dstOffset,
        int length)
    {
        Parallel.For(0, length, i =>
        {
            dst[dstOffset + i] = src[i];
        });
    }

    // ====================================================================
    // Attention — full per-token forward pass. One outer Parallel.For over
    // heads; the per-head body inlines (scores compute → sequential softmax →
    // V weighted sum). The softmax runs sequentially inside each lambda
    // iteration — safe on both flavors because each iteration executes on
    // exactly one team thread.
    //
    // scoresScratch is a caller-owned float[numHeads * seqLen]; each head
    // writes/reads its own [h*seqLen ..+seqLen) slice. Marked [Out] so the
    // initial host garbage isn't uploaded; the D→H readback at kernel end is
    // wasted bandwidth (we don't read it on the host) but only matters until
    // activations get promoted to FloatResidentArray in a later iteration.
    // ====================================================================

    /// <summary>
    /// Full per-token attention: one call replaces the managed
    /// <see cref="Attention.AttentionForwardOneToken"/> path. Supports Grouped
    /// Query Attention (numHeads / numKvHeads = group size).
    /// </summary>
    [EntryPoint]
    public static void AttentionForwardOneToken(
        float[] attnOut,
        [In] float[] q,
        [In] float[] keyCache,
        [In] float[] valueCache,
        [Out] float[] scoresScratch,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        int gqaGroupSize = numHeads / numKvHeads;
        int kvDim = numKvHeads * headDim;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / gqaGroupSize;
            int qOffset = h * headDim;
            int scoresOffset = h * seqLen;

            // Phase 1 — scores[t] = (q_h · k_{t, kvHead}) * scale, t in [0, seqLen).
            for (int t = 0; t < seqLen; t++)
            {
                int kCacheOffset = t * kvDim + kvHead * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    dot += q[qOffset + d] * keyCache[kCacheOffset + d];
                }
                scoresScratch[scoresOffset + t] = dot * scale;
            }

            // Phase 2 — in-place sequential softmax over the per-head slice.
            float max = scoresScratch[scoresOffset];
            for (int t = 1; t < seqLen; t++)
            {
                float v = scoresScratch[scoresOffset + t];
                if (v > max) max = v;
            }

            float sum = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                float ex = (float)System.Math.Exp(scoresScratch[scoresOffset + t] - max);
                scoresScratch[scoresOffset + t] = ex;
                sum += ex;
            }

            float invSum = 1.0f / sum;
            for (int t = 0; t < seqLen; t++)
            {
                scoresScratch[scoresOffset + t] *= invSum;
            }

            // Phase 3 — attnOut[h*headDim..] = Σ_t scores[t] * v_{t, kvHead}.
            for (int d = 0; d < headDim; d++)
            {
                attnOut[qOffset + d] = 0f;
            }
            for (int t = 0; t < seqLen; t++)
            {
                float score = scoresScratch[scoresOffset + t];
                int vCacheOffset = t * kvDim + kvHead * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    attnOut[qOffset + d] += score * valueCache[vCacheOffset + d];
                }
            }
        });
    }

    /// <summary>
    /// Resident-KV attention forward — identical body to
    /// <see cref="AttentionForwardOneToken"/> but <paramref name="keyCache"/>,
    /// <paramref name="valueCache"/> and <paramref name="scoresScratch"/> live
    /// device-side as <see cref="FloatResidentArray"/>, so the per-call
    /// marshalling of the multi-MB KV cache is gone. <paramref name="q"/> and
    /// <paramref name="attnOut"/> stay as <c>float[]</c> for now — they're
    /// 8 KB each, the next promotion iteration will move them too.
    /// </summary>
    [EntryPoint]
    public static void AttentionForwardOneTokenResident(
        float[] attnOut,
        [In] float[] q,
        FloatResidentArray keyCache,
        FloatResidentArray valueCache,
        FloatResidentArray scoresScratch,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        int gqaGroupSize = numHeads / numKvHeads;
        int kvDim = numKvHeads * headDim;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / gqaGroupSize;
            int qOffset = h * headDim;
            int scoresOffset = h * seqLen;

            // Phase 1 — scores[t] = (q_h · k_{t, kvHead}) * scale.
            for (int t = 0; t < seqLen; t++)
            {
                int kCacheOffset = t * kvDim + kvHead * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    dot += q[qOffset + d] * keyCache[kCacheOffset + d];
                }
                scoresScratch[scoresOffset + t] = dot * scale;
            }

            // Phase 2 — in-place sequential softmax.
            float max = scoresScratch[scoresOffset];
            for (int t = 1; t < seqLen; t++)
            {
                float v = scoresScratch[scoresOffset + t];
                if (v > max) max = v;
            }
            float sum = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                float ex = (float)System.Math.Exp(scoresScratch[scoresOffset + t] - max);
                scoresScratch[scoresOffset + t] = ex;
                sum += ex;
            }
            float invSum = 1.0f / sum;
            for (int t = 0; t < seqLen; t++)
            {
                scoresScratch[scoresOffset + t] *= invSum;
            }

            // Phase 3 — attnOut[h*headDim..] = Σ_t scores[t] * v_{t, kvHead}.
            for (int d = 0; d < headDim; d++)
            {
                attnOut[qOffset + d] = 0f;
            }
            for (int t = 0; t < seqLen; t++)
            {
                float score = scoresScratch[scoresOffset + t];
                int vCacheOffset = t * kvDim + kvHead * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    attnOut[qOffset + d] += score * valueCache[vCacheOffset + d];
                }
            }
        });
    }

    /// <summary>
    /// In-place RoPE on Q and K: each (Q-head | K-head) gets independent
    /// per-pair (cos, -sin / sin, cos) rotation using shared cos/sin tables.
    /// One <c>Parallel.For</c> iteration per head; pair loop runs sequentially
    /// inside the lambda.
    /// </summary>
    [EntryPoint]
    public static void ApplyRope(
        float[] q,
        float[] k,
        int headDim,
        int numHeads,
        int numKvHeads,
        [In] float[] cosTable,
        [In] float[] sinTable,
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
    // Fully-resident variants — CUDA path only (the OMP path doesn't
    // benefit from resident promotion since host==device on OMP), gated
    // via [HybridizerIgnore("OMP")] to keep the OMP satellite slimmer.
    //
    // sumOut on the RmsNorm reducer stays a plain float[1] so the host
    // can compute the sqrt scale between the two kernel launches.
    // weight on RmsNormBroadcast stays float[] (norm weights are 360 KB
    // total across the model — small enough to defer promoting).
    // ====================================================================

    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void RmsNormSumSquaresResidentInputCuda(int size, FloatResidentArray input, float[] sumOut)
    {
        Parallel.For(0, size, i =>
        {
            float v = input[i];
            Atomics.Add(ref sumOut[0], v * v);
        });
    }

    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void RmsNormBroadcastFullyResident(
        FloatResidentArray output,
        FloatResidentArray input,
        [In] float[] weight,
        int size,
        float scale)
    {
        Parallel.For(0, size, i =>
        {
            output[i] = input[i] * scale * weight[i];
        });
    }

    /// <summary>
    /// Fused RmsNorm broadcast that reads the sum-of-squares from a
    /// device-resident scalar <paramref name="sumBox"/> and computes the
    /// scale on device — drops the 4-byte <c>cudaMemcpy(D→H)</c> the
    /// non-fused <see cref="RmsNormBroadcastFullyResident"/> needed between
    /// phases. Every thread re-reads <c>sumBox[0]</c> and recomputes the
    /// scale (~3 ops); cheap and a broadcast read in cache. The 22 layers
    /// × 2 RmsNorms + 1 final × per-token saved D→H eliminated this way
    /// dominate the remaining sync chain on the deferred path.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void RmsNormBroadcastFromSumBox(
        FloatResidentArray output,
        FloatResidentArray input,
        [In] float[] sumBox,
        [In] float[] weight,
        int size,
        float eps)
    {
        Parallel.For(0, size, i =>
        {
            float scale = 1.0f / (float)System.Math.Sqrt(sumBox[0] / size + eps);
            output[i] = input[i] * scale * weight[i];
        });
    }

    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void AccumulateFullyResident(FloatResidentArray a, FloatResidentArray b, int size)
    {
        Parallel.For(0, size, i => { a[i] += b[i]; });
    }

    /// <summary>
    /// Bridge variant: <paramref name="a"/> still host <c>float[]</c>
    /// (e.g. <c>_x</c> not yet promoted), <paramref name="b"/> resident
    /// (e.g. <c>_attnProj</c> / <c>_ffnOut</c> after their step-3/4 promotion).
    /// Used until <c>_x</c> is promoted in step 5.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void AccumulateResidentB(float[] a, FloatResidentArray b, int size)
    {
        Parallel.For(0, size, i => { a[i] += b[i]; });
    }

    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void FusedSiluElementWiseMulFullyResident(FloatResidentArray gate, FloatResidentArray up, int size)
    {
        Parallel.For(0, size, i =>
        {
            float val = gate[i];
            gate[i] = val * (1.0f / (1.0f + (float)System.Math.Exp(-val))) * up[i];
        });
    }

    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void ApplyRopeFullyResident(
        FloatResidentArray q,
        FloatResidentArray k,
        int headDim,
        int numHeads,
        int numKvHeads,
        FloatResidentArray cosTable,
        FloatResidentArray sinTable,
        int ropeOffset,
        int ropePairCount)
    {
        int totalHeads = numHeads + numKvHeads;

        Parallel.For(0, totalHeads, idx =>
        {
            bool isQuery = idx < numHeads;
            int h = isQuery ? idx : idx - numHeads;
            FloatResidentArray target = isQuery ? q : k;
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

    /// <summary>
    /// Device-position variant of <see cref="ApplyRopeFullyResident"/> for
    /// CUDA-graph capture (iter 7.A.7.a): the decode position lives in a
    /// one-element device int slot updated by <see cref="IntKernels.BumpDeviceInt"/>
    /// instead of being passed as a host int. Each thread reads
    /// <c>position[0]</c> and derives <c>ropeOffset = position[0] * ropePairCount</c>
    /// — broadcast load served by L1/L2.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void ApplyRopeFullyResidentDev(
        FloatResidentArray q,
        FloatResidentArray k,
        int headDim,
        int numHeads,
        int numKvHeads,
        FloatResidentArray cosTable,
        FloatResidentArray sinTable,
        ResidentArrayGeneric<int> position,
        int ropePairCount)
    {
        int totalHeads = numHeads + numKvHeads;

        Parallel.For(0, totalHeads, idx =>
        {
            int ropeOffset = position[0] * ropePairCount;
            bool isQuery = idx < numHeads;
            int h = isQuery ? idx : idx - numHeads;
            FloatResidentArray target = isQuery ? q : k;
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

    /// <summary>
    /// Step-2 variant: only the cos/sin tables are resident; q and k stay as
    /// <c>float[]</c> until step 4 promotes them. Lets us land the RoPE-tables
    /// promotion as its own commit.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void ApplyRopeResidentTables(
        float[] q,
        float[] k,
        int headDim,
        int numHeads,
        int numKvHeads,
        FloatResidentArray cosTable,
        FloatResidentArray sinTable,
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

    /// <summary>
    /// Both <paramref name="src"/> and <paramref name="dst"/> resident — used
    /// once <c>_k</c> / <c>_v</c> are promoted in step 4.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void WriteKvCacheSliceResidentSrc(
        FloatResidentArray src,
        FloatResidentArray dst,
        int dstOffset,
        int length)
    {
        Parallel.For(0, length, i =>
        {
            dst[dstOffset + i] = src[i];
        });
    }

    /// <summary>
    /// Device-position variant of <see cref="WriteKvCacheSliceResidentSrc"/>
    /// for CUDA-graph capture (iter 7.A.7.a): each thread reads
    /// <c>position[0]</c> from a one-element device slot and derives
    /// <c>dstOffset = position[0] * length</c>. The host doesn't need to push
    /// the position int per token; <see cref="IntKernels.BumpDeviceInt"/>
    /// updates the slot device-side.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void WriteKvCacheSliceResidentSrcDev(
        FloatResidentArray src,
        FloatResidentArray dst,
        ResidentArrayGeneric<int> position,
        int length)
    {
        Parallel.For(0, length, i =>
        {
            int dstOffset = position[0] * length;
            dst[dstOffset + i] = src[i];
        });
    }

    /// <summary>
    /// Same body as <see cref="AttentionForwardOneTokenResident"/> but q and
    /// attnOut are also resident — used once those buffers are promoted.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void AttentionForwardOneTokenFullyResident(
        FloatResidentArray attnOut,
        FloatResidentArray q,
        FloatResidentArray keyCache,
        FloatResidentArray valueCache,
        FloatResidentArray scoresScratch,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        int gqaGroupSize = numHeads / numKvHeads;
        int kvDim = numKvHeads * headDim;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / gqaGroupSize;
            int qOffset = h * headDim;
            int scoresOffset = h * seqLen;

            for (int t = 0; t < seqLen; t++)
            {
                int kCacheOffset = t * kvDim + kvHead * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    dot += q[qOffset + d] * keyCache[kCacheOffset + d];
                }
                scoresScratch[scoresOffset + t] = dot * scale;
            }

            float max = scoresScratch[scoresOffset];
            for (int t = 1; t < seqLen; t++)
            {
                float v = scoresScratch[scoresOffset + t];
                if (v > max) max = v;
            }
            float sum = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                float ex = (float)System.Math.Exp(scoresScratch[scoresOffset + t] - max);
                scoresScratch[scoresOffset + t] = ex;
                sum += ex;
            }
            float invSum = 1.0f / sum;
            for (int t = 0; t < seqLen; t++)
            {
                scoresScratch[scoresOffset + t] *= invSum;
            }

            for (int d = 0; d < headDim; d++)
            {
                attnOut[qOffset + d] = 0f;
            }
            for (int t = 0; t < seqLen; t++)
            {
                float score = scoresScratch[scoresOffset + t];
                int vCacheOffset = t * kvDim + kvHead * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    attnOut[qOffset + d] += score * valueCache[vCacheOffset + d];
                }
            }
        });
    }

    /// <summary>
    /// Cooperative attention forward (step 7.A.2): <strong>one block per head</strong>
    /// with <c>blockDim.x</c> threads cooperating on three phases:
    /// <list type="number">
    /// <item>Per-t dot product <c>q · K[t]</c> (headDim mul-adds, reduced via shared mem).</item>
    /// <item>Online-streamed safe softmax over the seqLen scores (max + exp + sum,
    /// each pass reduced via shared mem).</item>
    /// <item>Weighted sum <c>attnOut[d] = Σ scores[t] * V[t, d]</c> distributed
    /// across threads by d (no reduction; each d written independently).</item>
    /// </list>
    /// Replaces the one-thread-per-head <see cref="AttentionForwardOneTokenFullyResident"/>
    /// which left 99% of the launch grid idle on TinyLlama-scale models
    /// (numHeads = 32). Same pattern as <see cref="LlamaCsharp.Math.Q8Kernels.MatVecMulCoopRow"/>.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void AttentionForwardOneTokenCoopHead(
        FloatResidentArray attnOut,
        FloatResidentArray q,
        FloatResidentArray keyCache,
        FloatResidentArray valueCache,
        FloatResidentArray scoresScratch,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        int h = blockIdx.x;
        if (h >= numHeads) return;
        int tid = threadIdx.x;
        int nThreads = blockDim.x;

        int gqaGroupSize = numHeads / numKvHeads;
        int kvDim = numKvHeads * headDim;
        int kvHead = h / gqaGroupSize;
        int qOffset = h * headDim;
        int scoresOffset = h * seqLen;

        var redCache = new SharedMemoryAllocator<float>().allocate(blockDim.x);

        // ---- Phase 1: scores[t] = scale * (q · K[t]) for t in [0, seqLen) ----
        for (int t = 0; t < seqLen; t++)
        {
            int kCacheOffset = t * kvDim + kvHead * headDim;
            float partial = 0f;
            for (int d = tid; d < headDim; d += nThreads)
            {
                partial = partial + q[qOffset + d] * keyCache[kCacheOffset + d];
            }
            redCache[tid] = partial;
            CUDAIntrinsics.__syncthreads();
            int stride = nThreads >> 1;
            while (stride > 0)
            {
                if (tid < stride)
                {
                    redCache[tid] = redCache[tid] + redCache[tid + stride];
                }
                CUDAIntrinsics.__syncthreads();
                stride = stride >> 1;
            }
            if (tid == 0)
            {
                scoresScratch[scoresOffset + t] = redCache[0] * scale;
            }
            CUDAIntrinsics.__syncthreads();
        }

        // ---- Phase 2a: max of scores[0..seqLen) ----
        // Each thread strides across t. -FLT_MAX seed so empty strides don't win.
        float maxLocal = -3.4028235e+38f;
        for (int t = tid; t < seqLen; t += nThreads)
        {
            float v = scoresScratch[scoresOffset + t];
            if (v > maxLocal) maxLocal = v;
        }
        redCache[tid] = maxLocal;
        CUDAIntrinsics.__syncthreads();
        int strideMax = nThreads >> 1;
        while (strideMax > 0)
        {
            if (tid < strideMax)
            {
                float a = redCache[tid];
                float b = redCache[tid + strideMax];
                redCache[tid] = a > b ? a : b;
            }
            CUDAIntrinsics.__syncthreads();
            strideMax = strideMax >> 1;
        }
        float maxVal = redCache[0];
        CUDAIntrinsics.__syncthreads();

        // ---- Phase 2b: exp(s-max) + sum ----
        // Writes the exponentiated scores back to scoresScratch in place.
        float sumLocal = 0f;
        for (int t = tid; t < seqLen; t += nThreads)
        {
            float ex = (float)System.Math.Exp(scoresScratch[scoresOffset + t] - maxVal);
            scoresScratch[scoresOffset + t] = ex;
            sumLocal = sumLocal + ex;
        }
        redCache[tid] = sumLocal;
        CUDAIntrinsics.__syncthreads();
        int strideSum = nThreads >> 1;
        while (strideSum > 0)
        {
            if (tid < strideSum)
            {
                redCache[tid] = redCache[tid] + redCache[tid + strideSum];
            }
            CUDAIntrinsics.__syncthreads();
            strideSum = strideSum >> 1;
        }
        float invSum = 1.0f / redCache[0];
        CUDAIntrinsics.__syncthreads();

        // ---- Phase 2c: normalize ----
        for (int t = tid; t < seqLen; t += nThreads)
        {
            scoresScratch[scoresOffset + t] = scoresScratch[scoresOffset + t] * invSum;
        }
        CUDAIntrinsics.__syncthreads();

        // ---- Phase 3: attnOut[h, d] = Σ_t scores[t] * V[t, kvHead, d] ----
        // Distributed across threads by d. No reduction — each d is independent.
        for (int d = tid; d < headDim; d += nThreads)
        {
            float acc = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                int vCacheOffset = t * kvDim + kvHead * headDim;
                acc = acc + scoresScratch[scoresOffset + t] * valueCache[vCacheOffset + d];
            }
            attnOut[qOffset + d] = acc;
        }
    }

    /// <summary>
    /// Warp-shuffle reduction variant of
    /// <see cref="AttentionForwardOneTokenCoopHead"/>. Same three-phase
    /// structure, but every block-wide reduction (phase 1 per-t dot
    /// product, phase 2a max, phase 2b sum) replaces its shared-mem tree
    /// with a 2-stage shuffle:
    /// <list type="number">
    /// <item>Within-warp 5× <c>shfl_down</c> → lane 0 of each warp holds
    /// the warp's partial.</item>
    /// <item>Lane 0 writes to a tiny shared-mem scratch; one
    /// <c>__syncthreads</c>; warp 0 shuffle-reduces the per-warp sums.
    /// Lane 0 of warp 0 broadcasts back via shared mem for the next
    /// phase to read.</item>
    /// </list>
    /// Drops ~5 <c>__syncthreads</c> per reduction. For TinyLlama
    /// (seqLen up to ~40 during decode, 3 reductions per attention call,
    /// 22 attention calls per token) that's ~3500 fewer syncthreads per
    /// token — at ~0.5 µs each ≈ 1.5–2 ms / token saved.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void AttentionForwardOneTokenCoopHeadShfl(
        FloatResidentArray attnOut,
        FloatResidentArray q,
        FloatResidentArray keyCache,
        FloatResidentArray valueCache,
        FloatResidentArray scoresScratch,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        int h = blockIdx.x;
        if (h >= numHeads) return;
        int tid = threadIdx.x;
        int nThreads = blockDim.x;
        int laneId = tid & 31;
        int warpId = tid >> 5;

        int gqaGroupSize = numHeads / numKvHeads;
        int kvDim = numKvHeads * headDim;
        int kvHead = h / gqaGroupSize;
        int qOffset = h * headDim;
        int scoresOffset = h * seqLen;

        // Cross-warp scratch + a single-float broadcast slot (slot 31 is
        // used for the broadcast of max/invSum so the cross-warp partials
        // [0..nWarps) don't collide).
        var cw = new SharedMemoryAllocator<float>().allocate(32);

        var tb = cooperative_groups.this_thread_block();
        var warp = cooperative_groups.tile_partition_32(tb);
        int nWarps = nThreads >> 5;

        // ---- Phase 1: scores[t] = scale * (q · K[t]) ----
        for (int t = 0; t < seqLen; t++)
        {
            int kCacheOffset = t * kvDim + kvHead * headDim;
            float partial = 0f;
            for (int d = tid; d < headDim; d += nThreads)
            {
                partial = partial + q[qOffset + d] * keyCache[kCacheOffset + d];
            }
            // Within-warp shuffle reduce.
            partial = partial + warp.shfl_down(partial, 16);
            partial = partial + warp.shfl_down(partial, 8);
            partial = partial + warp.shfl_down(partial, 4);
            partial = partial + warp.shfl_down(partial, 2);
            partial = partial + warp.shfl_down(partial, 1);
            // Lane 0 of each warp writes to cw[warpId].
            if (laneId == 0) cw[warpId] = partial;
            CUDAIntrinsics.__syncthreads();
            // Warp 0 reduces the nWarps partials.
            if (warpId == 0)
            {
                float v = laneId < nWarps ? cw[laneId] : 0f;
                v = v + warp.shfl_down(v, 16);
                v = v + warp.shfl_down(v, 8);
                v = v + warp.shfl_down(v, 4);
                v = v + warp.shfl_down(v, 2);
                v = v + warp.shfl_down(v, 1);
                if (laneId == 0) scoresScratch[scoresOffset + t] = v * scale;
            }
            CUDAIntrinsics.__syncthreads();
        }

        // ---- Phase 2a: max ----
        float maxLocal = -3.4028235e+38f;
        for (int t = tid; t < seqLen; t += nThreads)
        {
            float v = scoresScratch[scoresOffset + t];
            if (v > maxLocal) maxLocal = v;
        }
        // Warp-shuffle max reduce.
        {
            float o;
            o = warp.shfl_down(maxLocal, 16); if (o > maxLocal) maxLocal = o;
            o = warp.shfl_down(maxLocal, 8);  if (o > maxLocal) maxLocal = o;
            o = warp.shfl_down(maxLocal, 4);  if (o > maxLocal) maxLocal = o;
            o = warp.shfl_down(maxLocal, 2);  if (o > maxLocal) maxLocal = o;
            o = warp.shfl_down(maxLocal, 1);  if (o > maxLocal) maxLocal = o;
        }
        if (laneId == 0) cw[warpId] = maxLocal;
        CUDAIntrinsics.__syncthreads();
        if (warpId == 0)
        {
            float v = laneId < nWarps ? cw[laneId] : -3.4028235e+38f;
            float o;
            o = warp.shfl_down(v, 16); if (o > v) v = o;
            o = warp.shfl_down(v, 8);  if (o > v) v = o;
            o = warp.shfl_down(v, 4);  if (o > v) v = o;
            o = warp.shfl_down(v, 2);  if (o > v) v = o;
            o = warp.shfl_down(v, 1);  if (o > v) v = o;
            if (laneId == 0) cw[31] = v; // broadcast slot
        }
        CUDAIntrinsics.__syncthreads();
        float maxVal = cw[31];
        CUDAIntrinsics.__syncthreads();

        // ---- Phase 2b: exp + sum ----
        float sumLocal = 0f;
        for (int t = tid; t < seqLen; t += nThreads)
        {
            float ex = (float)System.Math.Exp(scoresScratch[scoresOffset + t] - maxVal);
            scoresScratch[scoresOffset + t] = ex;
            sumLocal = sumLocal + ex;
        }
        sumLocal = sumLocal + warp.shfl_down(sumLocal, 16);
        sumLocal = sumLocal + warp.shfl_down(sumLocal, 8);
        sumLocal = sumLocal + warp.shfl_down(sumLocal, 4);
        sumLocal = sumLocal + warp.shfl_down(sumLocal, 2);
        sumLocal = sumLocal + warp.shfl_down(sumLocal, 1);
        if (laneId == 0) cw[warpId] = sumLocal;
        CUDAIntrinsics.__syncthreads();
        if (warpId == 0)
        {
            float v = laneId < nWarps ? cw[laneId] : 0f;
            v = v + warp.shfl_down(v, 16);
            v = v + warp.shfl_down(v, 8);
            v = v + warp.shfl_down(v, 4);
            v = v + warp.shfl_down(v, 2);
            v = v + warp.shfl_down(v, 1);
            if (laneId == 0) cw[31] = 1.0f / v; // store invSum
        }
        CUDAIntrinsics.__syncthreads();
        float invSum = cw[31];
        CUDAIntrinsics.__syncthreads();

        // ---- Phase 2c: normalize ----
        for (int t = tid; t < seqLen; t += nThreads)
        {
            scoresScratch[scoresOffset + t] = scoresScratch[scoresOffset + t] * invSum;
        }
        CUDAIntrinsics.__syncthreads();

        // ---- Phase 3: V-weighted output (no reduction needed) ----
        for (int d = tid; d < headDim; d += nThreads)
        {
            float acc = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                int vCacheOffset = t * kvDim + kvHead * headDim;
                acc = acc + scoresScratch[scoresOffset + t] * valueCache[vCacheOffset + d];
            }
            attnOut[qOffset + d] = acc;
        }
    }

    /// <summary>
    /// CUB-reduction variant of <see cref="AttentionForwardOneTokenCoopHeadShfl"/>.
    /// Same three-phase structure and same launch shape (one block per head,
    /// blockDim = 64) but every block-wide reduction is now a single call into
    /// <c>cub::BlockReduce&lt;float,64&gt;</c> via <see cref="CubReduce"/>. The
    /// helper broadcasts the reduced value to every thread, so no manual
    /// cross-warp scratch or broadcast slot is needed.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void AttentionForwardOneTokenCoopHeadCub(
        FloatResidentArray attnOut,
        FloatResidentArray q,
        FloatResidentArray keyCache,
        FloatResidentArray valueCache,
        FloatResidentArray scoresScratch,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        int h = blockIdx.x;
        if (h >= numHeads) return;
        int tid = threadIdx.x;
        int nThreads = blockDim.x;

        int gqaGroupSize = numHeads / numKvHeads;
        int kvDim = numKvHeads * headDim;
        int kvHead = h / gqaGroupSize;
        int qOffset = h * headDim;
        int scoresOffset = h * seqLen;

        // ---- Phase 1: scores[t] = scale * (q · K[t]) ----
        for (int t = 0; t < seqLen; t++)
        {
            int kCacheOffset = t * kvDim + kvHead * headDim;
            float partial = 0f;
            for (int d = tid; d < headDim; d += nThreads)
            {
                partial = partial + q[qOffset + d] * keyCache[kCacheOffset + d];
            }
            float dot = CubReduce.SumBlock64(partial);
            if (tid == 0)
            {
                scoresScratch[scoresOffset + t] = dot * scale;
            }
            CUDAIntrinsics.__syncthreads();
        }

        // ---- Phase 2a: max ----
        float maxLocal = -3.4028235e+38f;
        for (int t = tid; t < seqLen; t += nThreads)
        {
            float v = scoresScratch[scoresOffset + t];
            if (v > maxLocal) maxLocal = v;
        }
        float maxVal = CubReduce.MaxBlock64(maxLocal);

        // ---- Phase 2b: exp + sum ----
        float sumLocal = 0f;
        for (int t = tid; t < seqLen; t += nThreads)
        {
            float ex = (float)System.Math.Exp(scoresScratch[scoresOffset + t] - maxVal);
            scoresScratch[scoresOffset + t] = ex;
            sumLocal = sumLocal + ex;
        }
        float sumTotal = CubReduce.SumBlock64(sumLocal);
        float invSum = 1.0f / sumTotal;

        // ---- Phase 2c: normalize ----
        for (int t = tid; t < seqLen; t += nThreads)
        {
            scoresScratch[scoresOffset + t] = scoresScratch[scoresOffset + t] * invSum;
        }
        CUDAIntrinsics.__syncthreads();

        // ---- Phase 3: V-weighted output (no reduction needed) ----
        for (int d = tid; d < headDim; d += nThreads)
        {
            float acc = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                int vCacheOffset = t * kvDim + kvHead * headDim;
                acc = acc + scoresScratch[scoresOffset + t] * valueCache[vCacheOffset + d];
            }
            attnOut[qOffset + d] = acc;
        }
    }

    /// <summary>
    /// Device-position variant of <see cref="AttentionForwardOneTokenCoopHeadCub"/>
    /// for CUDA-graph capture (iter 7.A.7.a): <c>seqLen</c> is derived from a
    /// one-element device int (<c>position[0] + 1</c>) instead of being a host
    /// int kernel arg. Each thread reads the slot once at the start; broadcast
    /// load served by L1.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void AttentionForwardOneTokenCoopHeadCubDev(
        FloatResidentArray attnOut,
        FloatResidentArray q,
        FloatResidentArray keyCache,
        FloatResidentArray valueCache,
        FloatResidentArray scoresScratch,
        ResidentArrayGeneric<int> position,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        int h = blockIdx.x;
        if (h >= numHeads) return;
        int tid = threadIdx.x;
        int nThreads = blockDim.x;
        int seqLen = position[0] + 1;

        int gqaGroupSize = numHeads / numKvHeads;
        int kvDim = numKvHeads * headDim;
        int kvHead = h / gqaGroupSize;
        int qOffset = h * headDim;
        int scoresOffset = h * seqLen;

        // ---- Phase 1: scores[t] = scale * (q · K[t]) ----
        for (int t = 0; t < seqLen; t++)
        {
            int kCacheOffset = t * kvDim + kvHead * headDim;
            float partial = 0f;
            for (int d = tid; d < headDim; d += nThreads)
            {
                partial = partial + q[qOffset + d] * keyCache[kCacheOffset + d];
            }
            float dot = CubReduce.SumBlock64(partial);
            if (tid == 0)
            {
                scoresScratch[scoresOffset + t] = dot * scale;
            }
            CUDAIntrinsics.__syncthreads();
        }

        // ---- Phase 2a: max ----
        float maxLocal = -3.4028235e+38f;
        for (int t = tid; t < seqLen; t += nThreads)
        {
            float v = scoresScratch[scoresOffset + t];
            if (v > maxLocal) maxLocal = v;
        }
        float maxVal = CubReduce.MaxBlock64(maxLocal);

        // ---- Phase 2b: exp + sum ----
        float sumLocal = 0f;
        for (int t = tid; t < seqLen; t += nThreads)
        {
            float ex = (float)System.Math.Exp(scoresScratch[scoresOffset + t] - maxVal);
            scoresScratch[scoresOffset + t] = ex;
            sumLocal = sumLocal + ex;
        }
        float sumTotal = CubReduce.SumBlock64(sumLocal);
        float invSum = 1.0f / sumTotal;

        // ---- Phase 2c: normalize ----
        for (int t = tid; t < seqLen; t += nThreads)
        {
            scoresScratch[scoresOffset + t] = scoresScratch[scoresOffset + t] * invSum;
        }
        CUDAIntrinsics.__syncthreads();

        // ---- Phase 3: V-weighted output (no reduction needed) ----
        for (int d = tid; d < headDim; d += nThreads)
        {
            float acc = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                int vCacheOffset = t * kvDim + kvHead * headDim;
                acc = acc + scoresScratch[scoresOffset + t] * valueCache[vCacheOffset + d];
            }
            attnOut[qOffset + d] = acc;
        }
    }
}
