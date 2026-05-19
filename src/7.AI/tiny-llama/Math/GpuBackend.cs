using Hybridizer.Runtime.CUDAImports;
using LlamaCsharp.Utils;

namespace LlamaCsharp.Math;

public enum ComputeBackend
{
    Managed,
    Cuda,
    Omp,
}

/// <summary>
/// Process-wide selector that routes forward-pass kernel calls to either the
/// managed dual-path or a Hybridizer satellite (CUDA / OMP).
///
/// Each kernel-bearing class (<see cref="Q8Kernels"/>, <see cref="FloatKernels"/>)
/// gets its own <see cref="HybRunner.Wrap"/>-produced <c>dynamic</c> proxy that
/// the dispatch methods route to. The proxies are constructed once at
/// <see cref="Activate"/> time and reused for every call so DLR binding caches
/// after the first invocation.
///
/// Not thread-safe — <see cref="Activate"/> is meant to be called once at startup,
/// then the forward pass runs serially.
/// </summary>
public static class GpuBackend
{
    private static ComputeBackend _mode = ComputeBackend.Managed;
    private static dynamic? _wrappedQ8;
    private static dynamic? _wrappedFloat;

    public static ComputeBackend Mode => _mode;

    public static void Activate(ComputeBackend mode)
    {
        _mode = mode;
        switch (mode)
        {
            case ComputeBackend.Managed:
                _wrappedQ8 = null;
                _wrappedFloat = null;
                break;
            case ComputeBackend.Cuda:
                (_wrappedQ8, _wrappedFloat) = WrapAllCuda();
                // Fully-resident kernels bypass HybRunner.Wrap via direct
                // NativeLibrary.GetExport + cached struct pointers; the
                // marshaller's per-call cudaMalloc+cudaMemcpy+cudaFree
                // chain is the bottleneck per nsys.
                CudaInvoke.Initialize();
                break;
            case ComputeBackend.Omp:
                (_wrappedQ8, _wrappedFloat) = WrapAllOmp();
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    // ====================================================================
    // Q8 matvec dispatch (host arrays — Managed + OMP)
    // ====================================================================

    public static void DispatchMatVecMul(
        float[] result,
        byte[] values,
        float[] scales,
        float[] vector,
        int rows,
        int blocksPerRow)
    {
        if (_mode == ComputeBackend.Managed)
        {
            Q8Kernels.MatVecMul(result, values, scales, vector, rows, blocksPerRow);
            return;
        }

        _wrappedQ8!.MatVecMul(result, values, scales, vector, rows, blocksPerRow);
        if (_mode == ComputeBackend.Cuda)
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    /// <summary>
    /// CUDA-only resident-weight Q8 matvec dispatch: weight tables already live
    /// on the device, only <paramref name="vector"/> + <paramref name="result"/>
    /// still cross the bus per call.
    /// </summary>
    public static void DispatchMatVecMulResident(
        float[] result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        float[] vector,
        int rows,
        int blocksPerRow)
    {
        _wrappedQ8!.MatVecMulResident(result, values, scales, vector, rows, blocksPerRow);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    // ====================================================================
    // Float ops dispatch — fall through to ParallelMath (which keeps its AVX
    // SIMD managed paths) for Managed; route to wrapped FloatKernels for
    // CUDA/OMP. Activations stay as float[] for now — resident-array overloads
    // can come later if/when activations get promoted.
    // ====================================================================

    public static void Accumulate(float[] a, float[] b, int size)
    {
        if (_mode == ComputeBackend.Managed)
        {
            ParallelMath.Accumulate(a, b, size);
            return;
        }
        _wrappedFloat!.Accumulate(a, b, size);
        if (_mode == ComputeBackend.Cuda)
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    public static void FusedSiluElementWiseMul(float[] gate, float[] up, int size)
    {
        if (_mode == ComputeBackend.Managed)
        {
            ParallelMath.FusedSiluElementWiseMul(gate, up, size);
            return;
        }
        _wrappedFloat!.FusedSiluElementWiseMul(gate, up, size);
        if (_mode == ComputeBackend.Cuda)
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    /// <summary>
    /// Two-pass RmsNorm: <see cref="FloatKernels.RmsNormSumSquares"/> writes
    /// the sum-of-squares into a single-cell <c>float[1]</c>, the host computes
    /// <c>scale = 1 / sqrt(sum / size + eps)</c>, then <see cref="FloatKernels.RmsNormBroadcast"/>
    /// applies the scale + weight broadcast. Two kernel launches + one D→H
    /// round-trip per call; that overhead is the price of keeping
    /// <c>Math.Sqrt</c> out of the kernel body and not needing the
    /// shape-B shared-mem reduction skeleton yet.
    /// </summary>
    public static void RmsNorm(float[] output, float[] input, float[] weight, int size, float eps)
    {
        if (_mode == ComputeBackend.Managed)
        {
            ParallelMath.RmsNorm(output, input, weight, size, eps);
            return;
        }

        float[] sumBox = new float[1];
        if (_mode == ComputeBackend.Cuda)
        {
            _wrappedFloat!.RmsNormSumSquaresCuda(size, input, sumBox);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        }
        else // Omp
        {
            _wrappedFloat!.RmsNormSumSquaresOmp(size, input, sumBox);
        }

        float scale = 1.0f / (float)System.Math.Sqrt(sumBox[0] / size + eps);

        _wrappedFloat!.RmsNormBroadcast(output, input, weight, size, scale);
        if (_mode == ComputeBackend.Cuda)
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    /// <summary>
    /// Three-phase Softmax: find max → exp(x - max) + sum → multiply by 1/sum.
    /// Two D→H syncs on CUDA (max and sum) plus three kernel launches. Kept as
    /// the parallel-atomic flavor of softmax; the sequential per-head softmax
    /// inside <see cref="Attention.AttentionForwardOneToken"/> stays on the CPU
    /// for now and will be replaced when Attention itself is ported.
    /// </summary>
    public static void Softmax(float[] x, int size)
    {
        if (_mode == ComputeBackend.Managed)
        {
            ParallelMath.Softmax(x, size);
            return;
        }

        float[] maxBox = new float[1];
        if (_mode == ComputeBackend.Cuda)
        {
            // Seed maxBox with x[0] so Atomics.Max has a sensible neutral element.
            // (float.NegativeInfinity is a fine neutral too; we just need the
            // first element if its sign happens to be the maximum.)
            maxBox[0] = float.NegativeInfinity;
            _wrappedFloat!.SoftmaxFindMaxCuda(size, x, maxBox);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        }
        else
        {
            _wrappedFloat!.SoftmaxFindMaxOmp(size, x, maxBox);
        }
        float max = maxBox[0];

        float[] sumBox = new float[1];
        if (_mode == ComputeBackend.Cuda)
        {
            _wrappedFloat!.SoftmaxExpAndSumCuda(size, x, max, sumBox);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        }
        else
        {
            _wrappedFloat!.SoftmaxExpAndSumOmp(size, x, max, sumBox);
        }

        float invSum = 1.0f / sumBox[0];
        _wrappedFloat!.SoftmaxNormalize(size, x, invSum);
        if (_mode == ComputeBackend.Cuda)
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    /// <summary>
    /// Per-token attention forward — Managed falls through to the
    /// <see cref="Attention.AttentionForwardOneToken"/> reference, CUDA/OMP
    /// route through the kernel that takes a caller-supplied scratch buffer.
    /// </summary>
    public static void AttentionForwardOneToken(
        float[] attnOut,
        float[] q,
        float[] keyCache,
        float[] valueCache,
        float[] scoresScratch,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        if (_mode == ComputeBackend.Managed)
        {
            Attention.AttentionForwardOneToken(
                attnOut, q, keyCache, valueCache,
                seqLen, numHeads, numKvHeads, headDim, scale);
            return;
        }

        _wrappedFloat!.AttentionForwardOneToken(
            attnOut, q, keyCache, valueCache, scoresScratch,
            seqLen, numHeads, numKvHeads, headDim, scale);
        if (_mode == ComputeBackend.Cuda)
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    /// <summary>
    /// CUDA-only resident-KV attention dispatch.
    /// <paramref name="keyCache"/> / <paramref name="valueCache"/> /
    /// <paramref name="scoresScratch"/> live device-side so no per-call
    /// marshalling of the multi-MB caches.
    /// </summary>
    public static void AttentionForwardOneTokenResident(
        float[] attnOut,
        float[] q,
        FloatResidentArray keyCache,
        FloatResidentArray valueCache,
        FloatResidentArray scoresScratch,
        int seqLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        float scale)
    {
        _wrappedFloat!.AttentionForwardOneTokenResident(
            attnOut, q, keyCache, valueCache, scoresScratch,
            seqLen, numHeads, numKvHeads, headDim, scale);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    /// <summary>
    /// CUDA-only: copy a <c>float[]</c> source into a slice of a
    /// device-resident <see cref="FloatResidentArray"/>. Replaces
    /// <see cref="Array.Copy(Array, int, Array, int, int)"/> for the
    /// per-token KV cache update when the cache is promoted.
    /// </summary>
    public static void WriteKvCacheSlice(
        float[] src,
        FloatResidentArray dst,
        int dstOffset,
        int length)
    {
        _wrappedFloat!.WriteKvCacheSlice(src, dst, dstOffset, length);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    public static void ApplyRope(
        float[] q, float[] k, int headDim, int numHeads, int numKvHeads,
        float[] cosTable, float[] sinTable, int ropeOffset, int ropePairCount)
    {
        if (_mode == ComputeBackend.Managed)
        {
            ParallelMath.ApplyRope(q, k, headDim, numHeads, numKvHeads, cosTable, sinTable, ropeOffset, ropePairCount);
            return;
        }
        _wrappedFloat!.ApplyRope(q, k, headDim, numHeads, numKvHeads, cosTable, sinTable, ropeOffset, ropePairCount);
        if (_mode == ComputeBackend.Cuda)
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    // ====================================================================
    // CUDA-only "fully resident" dispatchers — activations and weights both
    // live on the device, only kernel-launch overhead crosses the host
    // boundary per call. These exist to support the step-by-step activation
    // promotion in LlamaTransformer; OMP doesn't use them.
    // ====================================================================

    public static void DispatchMatVecMulFullyResident(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        FloatResidentArray vector,
        int rows,
        int blocksPerRow)
    {
        // Step 7.A.8 — vectorized Q8 byte load. Same launch shape as 7.A.6's
        // MatVecMulCoopRowCub; the inner loop now reads 4 contiguous Q8 bytes
        // per thread per iteration via the q8_load_uchar4 intrinsic, cutting
        // LSU instruction count 4× on the matvec hot path.
        //
        // Rollback to 7.A.6 if broken: change to MatVecMulCoopRowCub.
        // Rollback to 7.A.4: MatVecMulCoopRowShfl. All prior kernels stay wired.
        CudaInvoke.MatVecMulCoopRowCubVec4(result, values, scales, vector, rows, blocksPerRow);
    }

    /// <summary>
    /// Bridge variant for the FFN-block promotion (step 3): float[] vector,
    /// resident result. Becomes obsolete once <c>_xNorm</c> is also promoted
    /// in step 5 and callers switch to <see cref="DispatchMatVecMulFullyResident"/>.
    /// </summary>
    public static void DispatchMatVecMulResidentOut(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        float[] vector,
        int rows,
        int blocksPerRow)
    {
        _wrappedQ8!.MatVecMulResidentOut(result, values, scales, vector, rows, blocksPerRow);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    /// <summary>Bridge: float[] a + resident b.</summary>
    public static void AccumulateResidentB(float[] a, FloatResidentArray b, int size)
    {
        _wrappedFloat!.AccumulateResidentB(a, b, size);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    public static void DispatchDequantizeRowFullyResident(
        FloatResidentArray dst,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        int rowIndex,
        int cols,
        int blocksPerRow)
    {
        CudaInvoke.DequantizeRowFullyResident(dst, values, scales, rowIndex, cols, blocksPerRow);
    }

    public static void DispatchDequantizeRowByDeviceIdxFullyResident(
        FloatResidentArray dst,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        ResidentArrayGeneric<int> rowIdx,
        int cols,
        int blocksPerRow)
    {
        CudaInvoke.DequantizeRowByDeviceIdxFullyResident(dst, values, scales, rowIdx, cols, blocksPerRow);
    }

    public static void RmsNormResident(FloatResidentArray output, FloatResidentArray input, float[] weight, int size, float eps)
    {
        float[] sumBox = new float[1];
        _wrappedFloat!.RmsNormSumSquaresResidentInputCuda(size, input, sumBox);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());

        float scale = 1.0f / (float)System.Math.Sqrt(sumBox[0] / size + eps);

        _wrappedFloat!.RmsNormBroadcastFullyResident(output, input, weight, size, scale);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    /// <summary>
    /// Fast RmsNorm dispatch — both kernels go through CudaInvoke (no
    /// marshaller). The norm weight is a pre-uploaded device buffer; sumBox
    /// is a persistent 1-element device buffer the caller cudaMemsetAsyncs to
    /// zero before this call. The fused broadcast reads sumBox on device and
    /// computes the scale inline, so per-call D→H cost is **zero** — the
    /// previous variant's 4-byte sumBox readback (× 45 RmsNorms per token
    /// at TinyLlama scale) is fully eliminated.
    /// </summary>
    public static void RmsNormFastResident(
        FloatResidentArray output,
        FloatResidentArray input,
        FloatResidentArray weightResident,
        FloatResidentArray sumBoxResident,
        int size,
        float eps)
    {
        // Zero the persistent sumBox on device. Submitted on CudaInvoke.Stream
        // (the captured non-default stream) so it gets recorded into the
        // CUDA graph alongside the reducer kernel — passing the legacy
        // default stream here would abort graph capture (iter 7.A.7.b).
        cuda.ERROR_CHECK(cuda.MemsetAsync(sumBoxResident.DevicePointer, 0, (size_t)sizeof(float), new cudaStream_t(CudaInvoke.Stream)));

        CudaInvoke.RmsNormSumSquaresResidentInput(size, input, sumBoxResident.DevicePointer);

        // Fused broadcast: reads sumBox[0] on device, computes
        // scale = 1 / sqrt(sumBox/size + eps) inline, applies the broadcast.
        CudaInvoke.RmsNormBroadcastFromSumBox(output, input, sumBoxResident.DevicePointer, weightResident.DevicePointer, size, eps);
    }

    public static void AccumulateFullyResident(FloatResidentArray a, FloatResidentArray b, int size)
    {
        CudaInvoke.AccumulateFullyResident(a, b, size);
    }

    public static void FusedSiluElementWiseMulFullyResident(FloatResidentArray gate, FloatResidentArray up, int size)
    {
        CudaInvoke.FusedSiluElementWiseMulFullyResident(gate, up, size);
    }

    /// <summary>
    /// Step-2 variant: q and k stay <c>float[]</c>; only the cos/sin tables are
    /// resident. Lets us promote the tables as a self-contained commit.
    /// </summary>
    public static void ApplyRopeResidentTables(
        float[] q, float[] k, int headDim, int numHeads, int numKvHeads,
        FloatResidentArray cosTable, FloatResidentArray sinTable, int ropeOffset, int ropePairCount)
    {
        _wrappedFloat!.ApplyRopeResidentTables(q, k, headDim, numHeads, numKvHeads, cosTable, sinTable, ropeOffset, ropePairCount);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
    }

    public static void ApplyRopeFullyResident(
        FloatResidentArray q, FloatResidentArray k, int headDim, int numHeads, int numKvHeads,
        FloatResidentArray cosTable, FloatResidentArray sinTable, int ropeOffset, int ropePairCount)
    {
        CudaInvoke.ApplyRopeFullyResident(q, k, headDim, numHeads, numKvHeads, cosTable, sinTable, ropeOffset, ropePairCount);
    }

    /// <summary>
    /// Device-position RoPE (iter 7.A.7.a) — reads the decode position from a
    /// one-element device int slot. Used by the deferred-decode path so the
    /// per-token forward can be CUDA-graph-captured (iter 7.A.7.b).
    /// </summary>
    public static void ApplyRopeFullyResidentDev(
        FloatResidentArray q, FloatResidentArray k, int headDim, int numHeads, int numKvHeads,
        FloatResidentArray cosTable, FloatResidentArray sinTable,
        ResidentArrayGeneric<int> position, int ropePairCount)
    {
        CudaInvoke.ApplyRopeFullyResidentDev(q, k, headDim, numHeads, numKvHeads, cosTable, sinTable, position, ropePairCount);
    }

    public static void WriteKvCacheSliceResidentSrc(FloatResidentArray src, FloatResidentArray dst, int dstOffset, int length)
    {
        CudaInvoke.WriteKvCacheSliceResidentSrc(src, dst, dstOffset, length);
    }

    /// <summary>
    /// Device-position KV-cache slice write (iter 7.A.7.a) — derives
    /// <c>dstOffset = position[0] * length</c> inside the kernel.
    /// </summary>
    public static void WriteKvCacheSliceResidentSrcDev(
        FloatResidentArray src, FloatResidentArray dst,
        ResidentArrayGeneric<int> position, int length)
    {
        CudaInvoke.WriteKvCacheSliceResidentSrcDev(src, dst, position, length);
    }

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
        // Step 7.A.6 — CUB BlockReduce attention. Same shape as 7.A.5's
        // shuffle variant (one block per head, blockDim = 64); the three
        // block-wide reductions (per-t dot product, max, sum) now go through
        // cub::BlockReduce<float,64>. CUB's helpers broadcast to every thread
        // so no manual cross-warp scratch or broadcast slot is needed.
        //
        // Rollback to 7.A.5 if broken: change to AttentionForwardOneTokenCoopHeadShfl.
        // Rollback to 7.A.2: AttentionForwardOneTokenCoopHead. All stay wired.
        CudaInvoke.AttentionForwardOneTokenCoopHeadCub(
            attnOut, q, keyCache, valueCache, scoresScratch,
            seqLen, numHeads, numKvHeads, headDim, scale);
    }

    /// <summary>
    /// Device-position attention (iter 7.A.7.a) — reads <c>seqLen = position[0] + 1</c>
    /// from a one-element device int slot. Same launch shape as the host-int
    /// variant.
    /// </summary>
    public static void AttentionForwardOneTokenFullyResidentDev(
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
        CudaInvoke.AttentionForwardOneTokenCoopHeadCubDev(
            attnOut, q, keyCache, valueCache, scoresScratch,
            position, numHeads, numKvHeads, headDim, scale);
    }

    // ====================================================================
    // Satellite wrapping
    // ====================================================================

    private static (dynamic q8, dynamic floatOps) WrapAllCuda()
    {
        cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);
        HybRunner runner = SatelliteLoader.LoadCuda()
            .SetDistrib(prop.multiProcessorCount * 16, 128);
        return (runner.Wrap(new Q8Kernels()), runner.Wrap(new FloatKernels()));
    }

    private static (dynamic q8, dynamic floatOps) WrapAllOmp()
    {
        HybRunner runner = SatelliteLoader.LoadOmp();
        return (runner.Wrap(new Q8Kernels()), runner.Wrap(new FloatKernels()));
    }
}
