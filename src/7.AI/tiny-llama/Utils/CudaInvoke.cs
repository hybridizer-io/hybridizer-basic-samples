using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Utils;

/// <summary>
/// Direct invocation of the generated CUDA <c>_ExternCWrapperStream_CUDA</c>
/// kernel wrappers, bypassing HybRunner's per-call marshaller.
///
/// HybRunner.Wrap'ed calls go through <c>NativeSerializer.InitialVisit</c>
/// for every class/interface parameter — including FloatResidentArray —
/// which does cudaMalloc + cudaMemcpy + cudaDeviceSynchronize + cudaFree
/// per param per call. For a 32-token run that's ~92K cudaMemcpy / ~47K
/// cudaMalloc/Free, which dominates wall-clock per the nsys profile.
///
/// This bypass pre-allocates the 32-byte native struct mirror of each
/// FloatResidentArray / ResidentArrayGeneric&lt;byte&gt; ONCE via
/// <see cref="ResidentStructCache"/>, then passes the cached device-side
/// struct pointer to the kernel — zero marshalling at call time. Plus the
/// <c>_ExternCWrapperStream_CUDA</c> variant doesn't do the cudaDeviceSync
/// the non-Stream wrapper does, so we save that too.
///
/// Only the fully-resident kernels are exposed here. The wrapped path stays
/// alive for the kernels that still take <c>float[]</c> params (RmsNorm
/// sumBox/weight, the Managed/OMP-path kernels) — those need a separate
/// promotion to be worth bypassing.
/// </summary>
public static class CudaInvoke
{
    private static nint _lib;
    private static nint _stream;

    // Block size used by every element-wise / strided-Parallel.For kernel.
    // Cooperative-block kernels (matvec, attention) override their own block
    // size locally.
    private const int BlockDim = 128;

    /// <summary>
    /// Grid size that covers <paramref name="n"/> iterations with no waste:
    /// every thread gets at most one Parallel.For iteration (any extra
    /// threads exit on the bounds check). Replaces the historical
    /// "<c>_gridDimX = SMs × 16</c>" default that launched 73K threads for
    /// even tiny kernels like RoPE (36 heads) — see iter 7.A.10 in the
    /// porting log.
    /// </summary>
    private static int GridForSize(int n) => System.Math.Max(1, (n + BlockDim - 1) / BlockDim);

    public static bool Initialized { get; private set; }

    /// <summary>Satellite library handle, shared with <see cref="GraphInvoke"/>.</summary>
    internal static nint LibraryHandle => _lib;

    public static nint Stream => _stream;

    public static void Initialize()
    {
        if (Initialized) return;

        string? path = SatelliteLoader.CudaSatellitePath()
            ?? throw new System.IO.FileNotFoundException("CUDA satellite not found; build with <CompileCUDA>enable</CompileCUDA>.");
        _lib = NativeLibrary.Load(path);

        ResolveAll();

        // Iter 7.A.7.b — kernel dispatch runs on a dedicated non-default
        // stream so the decode body can be captured by cudaStreamBeginCapture
        // (rejects the legacy default stream). Cross-stream coherence with
        // the HybRunner.Wrap path (prompt eval) and host cuda.Memcpy calls
        // (drain, argmax read) is preserved: wrapped dispatchers end with
        // cuda.DeviceSynchronize and synchronous memcpy implicitly syncs all
        // streams. LLAMA_DISABLE_GRAPH=1 keeps this stream but disables the
        // capture/replay path itself (handled in LlamaTransformer).
        _stream = GraphInvoke.CreateStream();

        Initialized = true;
    }

    public static void Synchronize() => cuda.ERROR_CHECK(cuda.DeviceSynchronize());

    // ==== In-process kernel profiler =====================================
    //
    // Goal: rank kernels by wall time without depending on nsys (the WSL2
    // CUDA driver wedge has been blocking external profilers all session).
    //
    // When enabled, each dispatcher's Check(...) is wrapped with a Stopwatch
    // + cuda.DeviceSynchronize after the launch. The host wall time captures
    // launch overhead + queueing + actual GPU kernel execution — i.e. the
    // total time each kernel costs us. Numbers are NOT the pure-GPU times
    // nsys would report (we lose async pipelining during profiling) but
    // they're the right ranking for "what should we optimize next".
    //
    // Set the LLAMA_KERNEL_PROFILE=1 env var (read in Program.cs) to enable.
    // Overhead per call: ~3-5 µs of cudaDeviceSynchronize + Stopwatch +
    // dictionary update. With profiling on, expect 2-3× slower inference
    // (each kernel call forces a sync). Profiling OFF is free: a single
    // bool check skips the wrapper.

    private static bool _profEnabled;
    private static readonly Dictionary<string, (long Count, double TotalMs)> _profStats = new();

    public static bool ProfilingEnabled => _profEnabled;

    public static void EnableProfiling()
    {
        _profEnabled = true;
        _profStats.Clear();
    }

    /// <summary>
    /// Print the per-kernel timing table to stdout, sorted by total time.
    /// </summary>
    public static void PrintProfile()
    {
        if (_profStats.Count == 0)
        {
            System.Console.WriteLine("(no kernel timing samples — was profiling enabled?)");
            return;
        }
        double total = 0.0;
        foreach (var kv in _profStats) total += kv.Value.TotalMs;

        System.Console.WriteLine();
        System.Console.WriteLine("── Kernel timing (host wall-clock incl. launch + sync) ────");
        System.Console.WriteLine($"  {"Kernel",-50} {"calls",8} {"total ms",10} {"avg µs",10} {"%",6}");
        var entries = new List<KeyValuePair<string, (long Count, double TotalMs)>>(_profStats);
        entries.Sort((a, b) => b.Value.TotalMs.CompareTo(a.Value.TotalMs));
        foreach (var kv in entries)
        {
            double pct = total > 0 ? kv.Value.TotalMs * 100.0 / total : 0;
            double avgUs = kv.Value.Count > 0 ? kv.Value.TotalMs * 1000.0 / kv.Value.Count : 0;
            System.Console.WriteLine($"  {kv.Key,-50} {kv.Value.Count,8} {kv.Value.TotalMs,10:F2} {avgUs,10:F1} {pct,6:F1}");
        }
        System.Console.WriteLine($"  {"TOTAL",-50} {"",8} {total,10:F2}");
        System.Console.WriteLine();
    }

    /// <summary>
    /// Wraps a kernel-launch error check with optional timing. When
    /// profiling is off this is just `Check(err)`. When on, it also runs
    /// <c>cudaDeviceSynchronize</c> + accumulates the host wall time.
    /// Returning <paramref name="err"/> lets callers use this inline:
    ///   <c>Profile("Name", _kernelDel(...));</c>
    /// </summary>
    private static void Profile(string name, int err)
    {
        if (!_profEnabled)
        {
            Check(err);
            return;
        }
        var sw = Stopwatch.StartNew();
        Check(err);
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds;
        if (_profStats.TryGetValue(name, out var s))
            _profStats[name] = (s.Count + 1, s.TotalMs + ms);
        else
            _profStats[name] = (1, ms);
    }

    // ==== Q8 kernels ====================================================

    private delegate int MatVecMulFullyResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint result, nint values, nint scales, nint vector, int rows, int blocksPerRow);
    private static MatVecMulFullyResidentDel _matVecMulFullyResident = null!;

    private delegate int DequantizeRowFullyResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint dst, nint values, nint scales, int rowIndex, int cols, int blocksPerRow);
    private static DequantizeRowFullyResidentDel _dequantizeRowFullyResident = null!;

    private delegate int DequantizeRowByDeviceIdxFullyResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint dst, nint values, nint scales, nint rowIdx, int cols, int blocksPerRow);
    private static DequantizeRowByDeviceIdxFullyResidentDel _dequantizeRowByDeviceIdxFullyResident = null!;

    // Cooperative-row Q8 matvec — one block per row, blockDim.x threads stride
    // the column dot product, log₂(blockDim.x) shared-mem tree reduction.
    private delegate int MatVecMulCoopRowDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint result, nint values, nint scales, nint vector, int rows, int blocksPerRow);
    private static MatVecMulCoopRowDel _matVecMulCoopRow = null!;

    // 4-rows-per-block cooperative Q8 matvec — same signature, different
    // launch grid (gridDim.x = ceil(rows/4)) and 4× shared-mem requirement.
    private delegate int MatVecMulCoopRow4Del(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint result, nint values, nint scales, nint vector, int rows, int blocksPerRow);
    private static MatVecMulCoopRow4Del _matVecMulCoopRow4 = null!;

    // Warp-shuffle reduction variant — same launch shape as MatVecMulCoopRow
    // but reduction is 5 shfl_downs per warp + 1 cross-warp sync + 5 more
    // shfl_downs in warp 0. Drops 5 of the 7 __syncthreads.
    private delegate int MatVecMulCoopRowShflDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint result, nint values, nint scales, nint vector, int rows, int blocksPerRow);
    private static MatVecMulCoopRowShflDel _matVecMulCoopRowShfl = null!;

    // CUB BlockReduce variant — single cub::BlockReduce<float,128>::Sum call
    // replaces the manual 2-stage shuffle. Same launch shape, but no dynamic
    // shared mem needed (CUB's TempStorage is static __shared__ inside the
    // intrinsic helper).
    private delegate int MatVecMulCoopRowCubDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint result, nint values, nint scales, nint vector, int rows, int blocksPerRow);
    private static MatVecMulCoopRowCubDel _matVecMulCoopRowCub = null!;

    // Vectorized-load variant of MatVecMulCoopRowCub (iter 7.A.8). Same
    // signature as the parent; the C# kernel issues an [IntrinsicFunction]
    // q8_load_uchar4 call inside the inner loop and Hybridizer lowers it to
    // ld.global.v4.u8 (one LSU instruction per 4 bytes).
    private delegate int MatVecMulCoopRowCubVec4Del(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint result, nint values, nint scales, nint vector, int rows, int blocksPerRow);
    private static MatVecMulCoopRowCubVec4Del _matVecMulCoopRowCubVec4 = null!;

    // ==== FloatKernels — pure-resident ==================================

    private delegate int AccumulateFullyResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint a, nint b, int size);
    private static AccumulateFullyResidentDel _accumulateFullyResident = null!;

    private delegate int AccumulateResidentBDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint a, nint b, int size);  // a is float* (still managed array passes a marshalled pointer)
    private static AccumulateResidentBDel _accumulateResidentB = null!;

    private delegate int FusedSiluElementWiseMulFullyResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint gate, nint up, int size);
    private static FusedSiluElementWiseMulFullyResidentDel _fusedSiluElementWiseMulFullyResident = null!;

    private delegate int ApplyRopeFullyResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint q, nint k, int headDim, int numHeads, int numKvHeads,
        nint cosTable, nint sinTable, int ropeOffset, int ropePairCount);
    private static ApplyRopeFullyResidentDel _applyRopeFullyResident = null!;

    private delegate int WriteKvCacheSliceResidentSrcDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint src, nint dst, int dstOffset, int length);
    private static WriteKvCacheSliceResidentSrcDel _writeKvCacheSliceResidentSrc = null!;

    private delegate int AttentionForwardOneTokenFullyResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint attnOut, nint q, nint keyCache, nint valueCache, nint scoresScratch,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale);
    private static AttentionForwardOneTokenFullyResidentDel _attentionForwardOneTokenFullyResident = null!;

    // Cooperative-head attention — one block per head, threads cooperate on
    // dot products / softmax / weighted sum via shared-mem reductions.
    private delegate int AttentionForwardOneTokenCoopHeadDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint attnOut, nint q, nint keyCache, nint valueCache, nint scoresScratch,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale);
    private static AttentionForwardOneTokenCoopHeadDel _attentionForwardOneTokenCoopHead = null!;

    // Warp-shuffle reduction variant of the cooperative-head attention.
    private delegate int AttentionForwardOneTokenCoopHeadShflDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint attnOut, nint q, nint keyCache, nint valueCache, nint scoresScratch,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale);
    private static AttentionForwardOneTokenCoopHeadShflDel _attentionForwardOneTokenCoopHeadShfl = null!;

    // CUB BlockReduce variant of the cooperative-head attention. Same launch
    // shape as the Shfl variant; reductions go through cub::BlockReduce<float,64>.
    private delegate int AttentionForwardOneTokenCoopHeadCubDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint attnOut, nint q, nint keyCache, nint valueCache, nint scoresScratch,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale);
    private static AttentionForwardOneTokenCoopHeadCubDel _attentionForwardOneTokenCoopHeadCub = null!;

    // RmsNorm reducer: input resident, sumOut is a raw device float* (caller
    // owns a persistent 4-byte device buffer + cudaMemsetAsync before call).
    private delegate int RmsNormSumSquaresResidentInputCudaDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        int size, nint input, nint sumOut);
    private static RmsNormSumSquaresResidentInputCudaDel _rmsNormSumSquaresResidentInputCuda = null!;

    // RmsNorm broadcast: output + input resident; weight is a raw device float*
    // (caller owns a persistent device buffer copied from the host norm weight
    // at ctor time).
    private delegate int RmsNormBroadcastFullyResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint output, nint input, nint weight, int size, float scale);
    private static RmsNormBroadcastFullyResidentDel _rmsNormBroadcastFullyResident = null!;

    // Fused RmsNorm broadcast: same shape but the scale comes from a device-
    // side sumBox (no D→H needed between phases).
    private delegate int RmsNormBroadcastFromSumBoxDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint output, nint input, nint sumBox, nint weight, int size, float eps);
    private static RmsNormBroadcastFromSumBoxDel _rmsNormBroadcastFromSumBox = null!;

    // ==== IntKernels — int-resident plumbing ============================

    private delegate int WriteOneIntDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint dst, int value);
    private static WriteOneIntDel _writeOneInt = null!;

    private delegate int AppendDeviceIntDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint ring, nint counter, nint src, int ringSize);
    private static AppendDeviceIntDel _appendDeviceInt = null!;

    // Iter 7.A.7.a — bump a 1-element device int (++slot[0]) without a host
    // roundtrip. Used to advance the decode-position counter under graph
    // capture, where every kernel arg is locked at capture time.
    private delegate int BumpDeviceIntDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint slot);
    private static BumpDeviceIntDel _bumpDeviceInt = null!;

    // Iter 7.A.7.a — device-position variants of the three position-dependent
    // kernels. Each takes a ResidentArrayGeneric<int> position pointer instead
    // of a host int; the kernels read position[0] at the top and derive their
    // offsets (ropeOffset = position * ropePairCount; dstOffset = position *
    // length; seqLen = position + 1).
    private delegate int ApplyRopeFullyResidentDevDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint q, nint k, int headDim, int numHeads, int numKvHeads,
        nint cosTable, nint sinTable, nint position, int ropePairCount);
    private static ApplyRopeFullyResidentDevDel _applyRopeFullyResidentDev = null!;

    private delegate int WriteKvCacheSliceResidentSrcDevDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint src, nint dst, nint position, int length);
    private static WriteKvCacheSliceResidentSrcDevDel _writeKvCacheSliceResidentSrcDev = null!;

    private delegate int AttentionForwardOneTokenCoopHeadCubDevDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint attnOut, nint q, nint keyCache, nint valueCache, nint scoresScratch,
        nint position, int numHeads, int numKvHeads, int headDim, float scale);
    private static AttentionForwardOneTokenCoopHeadCubDevDel _attentionForwardOneTokenCoopHeadCubDev = null!;

    // ==== Argmax — two-phase resident reduction =========================
    // maxOut / idxOut are float* and int* respectively (caller owns persistent
    // 4-byte device buffers seeded with -inf / int.MaxValue before each call).

    private delegate int ArgmaxFindMaxResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        int n, nint x, nint maxOut);
    private static ArgmaxFindMaxResidentDel _argmaxFindMaxResident = null!;

    private delegate int ArgmaxFindFirstIndexResidentDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        int n, nint x, float maxVal, nint idxOut);
    private static ArgmaxFindFirstIndexResidentDel _argmaxFindFirstIndexResident = null!;

    private delegate int ArgmaxInitSeedsDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        nint maxBox, nint idxOut);
    private static ArgmaxInitSeedsDel _argmaxInitSeeds = null!;

    private delegate int ArgmaxFindFirstIndexFromMaxBoxDel(
        int gx, int gy, int gz, int bx, int by, int bz, int shared, nint stream,
        int n, nint x, nint maxBox, nint idxOut);
    private static ArgmaxFindFirstIndexFromMaxBoxDel _argmaxFindFirstIndexFromMaxBox = null!;

    private static void ResolveAll()
    {
        Resolve(out _matVecMulFullyResident, "LlamaCsharpx46Mathx46Q8Kernelsx46MatVecMulFullyResident_ExternCWrapperStream_CUDA");
        Resolve(out _dequantizeRowFullyResident, "LlamaCsharpx46Mathx46Q8Kernelsx46DequantizeRowFullyResident_ExternCWrapperStream_CUDA");
        Resolve(out _dequantizeRowByDeviceIdxFullyResident, "LlamaCsharpx46Mathx46Q8Kernelsx46DequantizeRowByDeviceIdxFullyResident_ExternCWrapperStream_CUDA");
        Resolve(out _matVecMulCoopRow, "LlamaCsharpx46Mathx46Q8Kernelsx46MatVecMulCoopRow_ExternCWrapperStream_CUDA");
        Resolve(out _matVecMulCoopRow4, "LlamaCsharpx46Mathx46Q8Kernelsx46MatVecMulCoopRow4_ExternCWrapperStream_CUDA");
        Resolve(out _matVecMulCoopRowShfl, "LlamaCsharpx46Mathx46Q8Kernelsx46MatVecMulCoopRowShfl_ExternCWrapperStream_CUDA");
        Resolve(out _matVecMulCoopRowCub, "LlamaCsharpx46Mathx46Q8Kernelsx46MatVecMulCoopRowCub_ExternCWrapperStream_CUDA");
        Resolve(out _matVecMulCoopRowCubVec4, "LlamaCsharpx46Mathx46Q8Kernelsx46MatVecMulCoopRowCubVec4_ExternCWrapperStream_CUDA");
        Resolve(out _accumulateFullyResident, "LlamaCsharpx46Mathx46FloatKernelsx46AccumulateFullyResident_ExternCWrapperStream_CUDA");
        Resolve(out _accumulateResidentB, "LlamaCsharpx46Mathx46FloatKernelsx46AccumulateResidentB_ExternCWrapperStream_CUDA");
        Resolve(out _fusedSiluElementWiseMulFullyResident, "LlamaCsharpx46Mathx46FloatKernelsx46FusedSiluElementWiseMulFullyResident_ExternCWrapperStream_CUDA");
        Resolve(out _applyRopeFullyResident, "LlamaCsharpx46Mathx46FloatKernelsx46ApplyRopeFullyResident_ExternCWrapperStream_CUDA");
        Resolve(out _writeKvCacheSliceResidentSrc, "LlamaCsharpx46Mathx46FloatKernelsx46WriteKvCacheSliceResidentSrc_ExternCWrapperStream_CUDA");
        Resolve(out _attentionForwardOneTokenFullyResident, "LlamaCsharpx46Mathx46FloatKernelsx46AttentionForwardOneTokenFullyResident_ExternCWrapperStream_CUDA");
        Resolve(out _attentionForwardOneTokenCoopHead, "LlamaCsharpx46Mathx46FloatKernelsx46AttentionForwardOneTokenCoopHead_ExternCWrapperStream_CUDA");
        Resolve(out _attentionForwardOneTokenCoopHeadShfl, "LlamaCsharpx46Mathx46FloatKernelsx46AttentionForwardOneTokenCoopHeadShfl_ExternCWrapperStream_CUDA");
        Resolve(out _attentionForwardOneTokenCoopHeadCub, "LlamaCsharpx46Mathx46FloatKernelsx46AttentionForwardOneTokenCoopHeadCub_ExternCWrapperStream_CUDA");
        Resolve(out _rmsNormSumSquaresResidentInputCuda, "LlamaCsharpx46Mathx46FloatKernelsx46RmsNormSumSquaresResidentInputCuda_ExternCWrapperStream_CUDA");
        Resolve(out _rmsNormBroadcastFullyResident, "LlamaCsharpx46Mathx46FloatKernelsx46RmsNormBroadcastFullyResident_ExternCWrapperStream_CUDA");
        Resolve(out _rmsNormBroadcastFromSumBox, "LlamaCsharpx46Mathx46FloatKernelsx46RmsNormBroadcastFromSumBox_ExternCWrapperStream_CUDA");
        Resolve(out _writeOneInt, "LlamaCsharpx46Mathx46IntKernelsx46WriteOneInt_ExternCWrapperStream_CUDA");
        Resolve(out _appendDeviceInt, "LlamaCsharpx46Mathx46IntKernelsx46AppendDeviceInt_ExternCWrapperStream_CUDA");
        Resolve(out _bumpDeviceInt, "LlamaCsharpx46Mathx46IntKernelsx46BumpDeviceInt_ExternCWrapperStream_CUDA");
        Resolve(out _applyRopeFullyResidentDev, "LlamaCsharpx46Mathx46FloatKernelsx46ApplyRopeFullyResidentDev_ExternCWrapperStream_CUDA");
        Resolve(out _writeKvCacheSliceResidentSrcDev, "LlamaCsharpx46Mathx46FloatKernelsx46WriteKvCacheSliceResidentSrcDev_ExternCWrapperStream_CUDA");
        Resolve(out _attentionForwardOneTokenCoopHeadCubDev, "LlamaCsharpx46Mathx46FloatKernelsx46AttentionForwardOneTokenCoopHeadCubDev_ExternCWrapperStream_CUDA");
        Resolve(out _argmaxFindMaxResident, "LlamaCsharpx46Mathx46FloatKernelsx46ArgmaxFindMaxResident_ExternCWrapperStream_CUDA");
        Resolve(out _argmaxFindFirstIndexResident, "LlamaCsharpx46Mathx46FloatKernelsx46ArgmaxFindFirstIndexResident_ExternCWrapperStream_CUDA");
        Resolve(out _argmaxInitSeeds, "LlamaCsharpx46Mathx46FloatKernelsx46ArgmaxInitSeeds_ExternCWrapperStream_CUDA");
        Resolve(out _argmaxFindFirstIndexFromMaxBox, "LlamaCsharpx46Mathx46FloatKernelsx46ArgmaxFindFirstIndexFromMaxBox_ExternCWrapperStream_CUDA");
    }

    private static void Resolve<TDelegate>(out TDelegate del, string symbol) where TDelegate : System.Delegate
    {
        nint fp = NativeLibrary.GetExport(_lib, symbol);
        del = Marshal.GetDelegateForFunctionPointer<TDelegate>(fp);
    }

    // ==== Dispatchers ===================================================

    public static void MatVecMulFullyResident(FloatResidentArray result, ResidentArrayGeneric<byte> values, FloatResidentArray scales, FloatResidentArray vector, int rows, int blocksPerRow)
    {
        Profile("MatVecMulFullyResident", _matVecMulFullyResident(
            GridForSize(rows), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(result),
            ResidentStructCache.Get(values),
            ResidentStructCache.Get(scales),
            ResidentStructCache.Get(vector),
            rows, blocksPerRow));
    }

    public static void DequantizeRowFullyResident(FloatResidentArray dst, ResidentArrayGeneric<byte> values, FloatResidentArray scales, int rowIndex, int cols, int blocksPerRow)
    {
        Profile("DequantizeRowFullyResident", _dequantizeRowFullyResident(
            GridForSize(cols), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(dst),
            ResidentStructCache.Get(values),
            ResidentStructCache.Get(scales),
            rowIndex, cols, blocksPerRow));
    }

    /// <summary>
    /// Q8 row-dequantize that reads the row index from a device-resident int
    /// slot — see <see cref="LlamaCsharp.Math.Q8Kernels.DequantizeRowByDeviceIdxFullyResident"/>.
    /// Used by the GPU greedy sampler to feed the next-token id from the
    /// previous step's argmax straight into the embedding lookup with no
    /// host roundtrip.
    /// </summary>
    public static void DequantizeRowByDeviceIdxFullyResident(FloatResidentArray dst, ResidentArrayGeneric<byte> values, FloatResidentArray scales, ResidentArrayGeneric<int> rowIdx, int cols, int blocksPerRow)
    {
        Profile("DequantizeRowByDeviceIdxFullyResident", _dequantizeRowByDeviceIdxFullyResident(
            GridForSize(cols), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(dst),
            ResidentStructCache.Get(values),
            ResidentStructCache.Get(scales),
            ResidentStructCache.Get(rowIdx),
            cols, blocksPerRow));
    }

    // Block size for the cooperative-row matvec. 128 threads × 4 bytes = 512 B
    // of shared memory per block; well under the per-block limit. Block size
    // is hard-coded here so the kernel can sync over a fixed power-of-two
    // count without runtime branching.
    private const int CoopRowBlockDim = 128;

    /// <summary>
    /// Cooperative-row Q8 matvec — see
    /// <see cref="LlamaCsharp.Math.Q8Kernels.MatVecMulCoopRow"/>. One block
    /// per output row; <c>CoopRowBlockDim</c> threads cooperate on the
    /// column dot product. Drops the per-call thread under-utilization the
    /// one-thread-per-row layout suffered on small-row matrices.
    /// </summary>
    public static void MatVecMulCoopRow(FloatResidentArray result, ResidentArrayGeneric<byte> values, FloatResidentArray scales, FloatResidentArray vector, int rows, int blocksPerRow)
    {
        int sharedBytes = CoopRowBlockDim * sizeof(float);
        Profile("MatVecMulCoopRow", _matVecMulCoopRow(
            rows, 1, 1, CoopRowBlockDim, 1, 1, sharedBytes, _stream,
            ResidentStructCache.Get(result),
            ResidentStructCache.Get(values),
            ResidentStructCache.Get(scales),
            ResidentStructCache.Get(vector),
            rows, blocksPerRow));
    }

    /// <summary>
    /// 4-rows-per-block variant — see
    /// <see cref="LlamaCsharp.Math.Q8Kernels.MatVecMulCoopRow4"/>. Halves the
    /// vector-load instruction count per kernel and 4×s shared-memory usage
    /// (4 reduction slots × <see cref="CoopRowBlockDim"/> threads × 4 B
    /// = 2 KB per block; well under per-block limits).
    /// </summary>
    public static void MatVecMulCoopRow4(FloatResidentArray result, ResidentArrayGeneric<byte> values, FloatResidentArray scales, FloatResidentArray vector, int rows, int blocksPerRow)
    {
        int gridX = (rows + 3) / 4;
        int sharedBytes = 4 * CoopRowBlockDim * sizeof(float);
        Profile("MatVecMulCoopRow4", _matVecMulCoopRow4(
            gridX, 1, 1, CoopRowBlockDim, 1, 1, sharedBytes, _stream,
            ResidentStructCache.Get(result),
            ResidentStructCache.Get(values),
            ResidentStructCache.Get(scales),
            ResidentStructCache.Get(vector),
            rows, blocksPerRow));
    }

    /// <summary>
    /// Warp-shuffle reduction variant — see
    /// <see cref="LlamaCsharp.Math.Q8Kernels.MatVecMulCoopRowShfl"/>.
    /// Same launch shape as <see cref="MatVecMulCoopRow"/> but the kernel
    /// uses cooperative-groups shfl_down to reduce within each warp without
    /// __syncthreads; cross-warp via a 32-float shared mem array.
    /// </summary>
    public static void MatVecMulCoopRowShfl(FloatResidentArray result, ResidentArrayGeneric<byte> values, FloatResidentArray scales, FloatResidentArray vector, int rows, int blocksPerRow)
    {
        // Shared mem holds one float per warp. Allocator reserves 32 floats
        // = 128 B (matches kernel's allocate(32) call), well under any limit.
        int sharedBytes = 32 * sizeof(float);
        Profile("MatVecMulCoopRowShfl", _matVecMulCoopRowShfl(
            rows, 1, 1, CoopRowBlockDim, 1, 1, sharedBytes, _stream,
            ResidentStructCache.Get(result),
            ResidentStructCache.Get(values),
            ResidentStructCache.Get(scales),
            ResidentStructCache.Get(vector),
            rows, blocksPerRow));
    }

    /// <summary>
    /// CUB BlockReduce matvec — see
    /// <see cref="LlamaCsharp.Math.Q8Kernels.MatVecMulCoopRowCub"/>. Same launch
    /// shape as <see cref="MatVecMulCoopRow"/> (one block per row, blockDim =
    /// 128). No dynamic shared memory: CUB's TempStorage and the broadcast slot
    /// are static <c>__shared__</c> declarations inside the intrinsic helper.
    /// </summary>
    public static void MatVecMulCoopRowCub(FloatResidentArray result, ResidentArrayGeneric<byte> values, FloatResidentArray scales, FloatResidentArray vector, int rows, int blocksPerRow)
    {
        Profile("MatVecMulCoopRowCub", _matVecMulCoopRowCub(
            rows, 1, 1, CoopRowBlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(result),
            ResidentStructCache.Get(values),
            ResidentStructCache.Get(scales),
            ResidentStructCache.Get(vector),
            rows, blocksPerRow));
    }

    /// <summary>
    /// Vectorized-load variant of <see cref="MatVecMulCoopRowCub"/> (iter
    /// 7.A.8). Same launch shape and kernel arguments; the kernel issues
    /// <c>ld.global.v4.u8</c> per thread per iteration via the
    /// <c>q8_load_uchar4</c> intrinsic. See
    /// <see cref="LlamaCsharp.Math.Q8Kernels.MatVecMulCoopRowCubVec4"/>.
    /// </summary>
    public static void MatVecMulCoopRowCubVec4(FloatResidentArray result, ResidentArrayGeneric<byte> values, FloatResidentArray scales, FloatResidentArray vector, int rows, int blocksPerRow)
    {
        Profile("MatVecMulCoopRowCubVec4", _matVecMulCoopRowCubVec4(
            rows, 1, 1, CoopRowBlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(result),
            ResidentStructCache.Get(values),
            ResidentStructCache.Get(scales),
            ResidentStructCache.Get(vector),
            rows, blocksPerRow));
    }

    public static void AccumulateFullyResident(FloatResidentArray a, FloatResidentArray b, int size)
    {
        Profile("AccumulateFullyResident", _accumulateFullyResident(
            GridForSize(size), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(a),
            ResidentStructCache.Get(b),
            size));
    }

    public static void FusedSiluElementWiseMulFullyResident(FloatResidentArray gate, FloatResidentArray up, int size)
    {
        Profile("FusedSiluElementWiseMulFullyResident", _fusedSiluElementWiseMulFullyResident(
            GridForSize(size), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(gate),
            ResidentStructCache.Get(up),
            size));
    }

    public static void ApplyRopeFullyResident(
        FloatResidentArray q, FloatResidentArray k, int headDim, int numHeads, int numKvHeads,
        FloatResidentArray cosTable, FloatResidentArray sinTable, int ropeOffset, int ropePairCount)
    {
        Profile("ApplyRopeFullyResident", _applyRopeFullyResident(
            GridForSize(numHeads + numKvHeads), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(q),
            ResidentStructCache.Get(k),
            headDim, numHeads, numKvHeads,
            ResidentStructCache.Get(cosTable),
            ResidentStructCache.Get(sinTable),
            ropeOffset, ropePairCount));
    }

    public static void WriteKvCacheSliceResidentSrc(FloatResidentArray src, FloatResidentArray dst, int dstOffset, int length)
    {
        Profile("WriteKvCacheSliceResidentSrc", _writeKvCacheSliceResidentSrc(
            GridForSize(length), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(src),
            ResidentStructCache.Get(dst),
            dstOffset, length));
    }

    public static void AttentionForwardOneTokenFullyResident(
        FloatResidentArray attnOut, FloatResidentArray q,
        FloatResidentArray keyCache, FloatResidentArray valueCache, FloatResidentArray scoresScratch,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale)
    {
        Profile("AttentionForwardOneTokenFullyResident", _attentionForwardOneTokenFullyResident(
            GridForSize(numHeads), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(attnOut),
            ResidentStructCache.Get(q),
            ResidentStructCache.Get(keyCache),
            ResidentStructCache.Get(valueCache),
            ResidentStructCache.Get(scoresScratch),
            seqLen, numHeads, numKvHeads, headDim, scale));
    }

    // Block size for the cooperative-head attention. 64 threads per block;
    // matches typical headDim (TinyLlama / Llama-3.2-3B), 256 B of shared
    // memory per block. If headDim grows beyond 64, the kernel's strided
    // loops still work — just slightly less efficient on the per-d weighted
    // sum.
    private const int CoopHeadBlockDim = 64;

    /// <summary>
    /// Cooperative-head attention forward — see
    /// <see cref="LlamaCsharp.Math.FloatKernels.AttentionForwardOneTokenCoopHead"/>.
    /// One block per head; threads cooperate on the per-head dot products,
    /// softmax reductions, and weighted-sum write. Replaces the
    /// one-thread-per-head launch which under-utilized the GPU at TinyLlama
    /// numHeads = 32.
    /// </summary>
    public static void AttentionForwardOneTokenCoopHead(
        FloatResidentArray attnOut, FloatResidentArray q,
        FloatResidentArray keyCache, FloatResidentArray valueCache, FloatResidentArray scoresScratch,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale)
    {
        int sharedBytes = CoopHeadBlockDim * sizeof(float);
        Profile("AttentionForwardOneTokenCoopHead", _attentionForwardOneTokenCoopHead(
            numHeads, 1, 1, CoopHeadBlockDim, 1, 1, sharedBytes, _stream,
            ResidentStructCache.Get(attnOut),
            ResidentStructCache.Get(q),
            ResidentStructCache.Get(keyCache),
            ResidentStructCache.Get(valueCache),
            ResidentStructCache.Get(scoresScratch),
            seqLen, numHeads, numKvHeads, headDim, scale));
    }

    /// <summary>
    /// Warp-shuffle attention — see
    /// <see cref="LlamaCsharp.Math.FloatKernels.AttentionForwardOneTokenCoopHeadShfl"/>.
    /// Same launch shape (one block per head, blockDim = CoopHeadBlockDim
    /// = 64), but the kernel uses shfl_down for all three block-wide
    /// reductions (per-t dot product, max, sum). Shared mem holds 32
    /// floats (cross-warp scratch + a broadcast slot at index 31).
    /// </summary>
    public static void AttentionForwardOneTokenCoopHeadShfl(
        FloatResidentArray attnOut, FloatResidentArray q,
        FloatResidentArray keyCache, FloatResidentArray valueCache, FloatResidentArray scoresScratch,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale)
    {
        int sharedBytes = 32 * sizeof(float);
        Profile("AttentionForwardOneTokenCoopHeadShfl", _attentionForwardOneTokenCoopHeadShfl(
            numHeads, 1, 1, CoopHeadBlockDim, 1, 1, sharedBytes, _stream,
            ResidentStructCache.Get(attnOut),
            ResidentStructCache.Get(q),
            ResidentStructCache.Get(keyCache),
            ResidentStructCache.Get(valueCache),
            ResidentStructCache.Get(scoresScratch),
            seqLen, numHeads, numKvHeads, headDim, scale));
    }

    /// <summary>
    /// CUB BlockReduce attention — see
    /// <see cref="LlamaCsharp.Math.FloatKernels.AttentionForwardOneTokenCoopHeadCub"/>.
    /// Same launch shape (one block per head, blockDim = CoopHeadBlockDim = 64);
    /// the three block-wide reductions use <c>cub::BlockReduce&lt;float,64&gt;</c>.
    /// No dynamic shared memory — CUB's TempStorage and the broadcast slot live
    /// in static <c>__shared__</c> inside the intrinsic helper.
    /// </summary>
    public static void AttentionForwardOneTokenCoopHeadCub(
        FloatResidentArray attnOut, FloatResidentArray q,
        FloatResidentArray keyCache, FloatResidentArray valueCache, FloatResidentArray scoresScratch,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale)
    {
        Profile("AttentionForwardOneTokenCoopHeadCub", _attentionForwardOneTokenCoopHeadCub(
            numHeads, 1, 1, CoopHeadBlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(attnOut),
            ResidentStructCache.Get(q),
            ResidentStructCache.Get(keyCache),
            ResidentStructCache.Get(valueCache),
            ResidentStructCache.Get(scoresScratch),
            seqLen, numHeads, numKvHeads, headDim, scale));
    }

    /// <summary>
    /// RmsNorm sum-of-squares reducer with both input resident and sumOut as
    /// a raw device pointer (caller has already zeroed it via cudaMemsetAsync).
    /// </summary>
    public static void RmsNormSumSquaresResidentInput(int size, FloatResidentArray input, nint sumOutDev)
    {
        Profile("RmsNormSumSquaresResidentInputCuda", _rmsNormSumSquaresResidentInputCuda(
            GridForSize(size), 1, 1, BlockDim, 1, 1, 0, _stream,
            size, ResidentStructCache.Get(input), sumOutDev));
    }

    /// <summary>
    /// RmsNorm broadcast with both output + input resident and weight as a raw
    /// device pointer (caller owns a persistent buffer with the norm weight
    /// pre-uploaded). Scale is computed on the host between the two calls.
    /// </summary>
    public static void RmsNormBroadcastFullyResident(FloatResidentArray output, FloatResidentArray input, nint weightDev, int size, float scale)
    {
        Profile("RmsNormBroadcastFullyResident", _rmsNormBroadcastFullyResident(
            GridForSize(size), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(output), ResidentStructCache.Get(input), weightDev,
            size, scale));
    }

    /// <summary>
    /// Fused RmsNorm broadcast: reads sum-of-squares from
    /// <paramref name="sumBoxDev"/> on device, computes the scale inline,
    /// applies <c>output = input * scale * weight</c>. Eliminates the 4-byte
    /// D→H of sumBox that the non-fused variant needs between phases.
    /// </summary>
    public static void RmsNormBroadcastFromSumBox(FloatResidentArray output, FloatResidentArray input, nint sumBoxDev, nint weightDev, int size, float eps)
    {
        Profile("RmsNormBroadcastFromSumBox", _rmsNormBroadcastFromSumBox(
            GridForSize(size), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(output), ResidentStructCache.Get(input), sumBoxDev, weightDev,
            size, eps));
    }

    /// <summary>
    /// Smoke single-int write: stores <paramref name="value"/> into
    /// <paramref name="dst"/>[0]. Used by the int-resident bring-up test and
    /// to seed the device-side next-token slot before the GPU-sampling
    /// generation loop starts.
    /// </summary>
    public static void WriteOneInt(ResidentArrayGeneric<int> dst, int value)
    {
        Profile("WriteOneInt", _writeOneInt(
            1, 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(dst), value));
    }

    /// <summary>
    /// Append <paramref name="src"/>[0] to <paramref name="ring"/> at slot
    /// <paramref name="counter"/>[0] mod <paramref name="ringSize"/>, then
    /// increment the counter. All three buffers stay on device; the host
    /// learns about new tokens via a periodic drain that reads the counter.
    /// </summary>
    public static void AppendDeviceInt(
        ResidentArrayGeneric<int> ring,
        ResidentArrayGeneric<int> counter,
        ResidentArrayGeneric<int> src,
        int ringSize)
    {
        Profile("AppendDeviceInt", _appendDeviceInt(
            1, 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(ring),
            ResidentStructCache.Get(counter),
            ResidentStructCache.Get(src),
            ringSize));
    }

    /// <summary>
    /// Bump a one-element device int slot: <c>slot[0]++</c>. Used to advance
    /// the decode-position counter device-side under graph capture (iter
    /// 7.A.7.a).
    /// </summary>
    public static void BumpDeviceInt(ResidentArrayGeneric<int> slot)
    {
        Profile("BumpDeviceInt", _bumpDeviceInt(
            1, 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(slot)));
    }

    /// <summary>
    /// Device-position RoPE — see
    /// <see cref="LlamaCsharp.Math.FloatKernels.ApplyRopeFullyResidentDev"/>.
    /// Reads the current decode position from a one-element device int slot
    /// (<paramref name="position"/>) instead of a host int.
    /// </summary>
    public static void ApplyRopeFullyResidentDev(
        FloatResidentArray q, FloatResidentArray k, int headDim, int numHeads, int numKvHeads,
        FloatResidentArray cosTable, FloatResidentArray sinTable,
        ResidentArrayGeneric<int> position, int ropePairCount)
    {
        Profile("ApplyRopeFullyResidentDev", _applyRopeFullyResidentDev(
            GridForSize(numHeads + numKvHeads), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(q),
            ResidentStructCache.Get(k),
            headDim, numHeads, numKvHeads,
            ResidentStructCache.Get(cosTable),
            ResidentStructCache.Get(sinTable),
            ResidentStructCache.Get(position),
            ropePairCount));
    }

    /// <summary>
    /// Device-position KV-cache slice write — see
    /// <see cref="LlamaCsharp.Math.FloatKernels.WriteKvCacheSliceResidentSrcDev"/>.
    /// Computes <c>dstOffset = position[0] * length</c> inside the kernel.
    /// </summary>
    public static void WriteKvCacheSliceResidentSrcDev(
        FloatResidentArray src, FloatResidentArray dst,
        ResidentArrayGeneric<int> position, int length)
    {
        Profile("WriteKvCacheSliceResidentSrcDev", _writeKvCacheSliceResidentSrcDev(
            GridForSize(length), 1, 1, BlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(src),
            ResidentStructCache.Get(dst),
            ResidentStructCache.Get(position),
            length));
    }

    /// <summary>
    /// Device-position CUB attention — see
    /// <see cref="LlamaCsharp.Math.FloatKernels.AttentionForwardOneTokenCoopHeadCubDev"/>.
    /// Same launch shape as the host-int variant; reads
    /// <c>seqLen = position[0] + 1</c> from a one-element device int slot.
    /// </summary>
    public static void AttentionForwardOneTokenCoopHeadCubDev(
        FloatResidentArray attnOut, FloatResidentArray q,
        FloatResidentArray keyCache, FloatResidentArray valueCache, FloatResidentArray scoresScratch,
        ResidentArrayGeneric<int> position,
        int numHeads, int numKvHeads, int headDim, float scale)
    {
        Profile("AttentionForwardOneTokenCoopHeadCubDev", _attentionForwardOneTokenCoopHeadCubDev(
            numHeads, 1, 1, CoopHeadBlockDim, 1, 1, 0, _stream,
            ResidentStructCache.Get(attnOut),
            ResidentStructCache.Get(q),
            ResidentStructCache.Get(keyCache),
            ResidentStructCache.Get(valueCache),
            ResidentStructCache.Get(scoresScratch),
            ResidentStructCache.Get(position),
            numHeads, numKvHeads, headDim, scale));
    }

    /// <summary>
    /// Two-phase resident argmax: writes the smallest-index argmax of
    /// <paramref name="logits"/>[0..size) into <paramref name="idxOutDev"/>[0].
    /// <paramref name="maxBoxDev"/> is a caller-owned 4-byte device float
    /// scratch. <paramref name="idxOutDev"/> is a caller-owned 4-byte device
    /// int. Both are seeded on device by an init kernel — caller does NOT
    /// need to pre-write -∞ / int.MaxValue (the previous variant did).
    /// Per call: <strong>zero D→H</strong>. Implementation is 3 kernels:
    /// (1) init seeds, (2) find max value, (3) atomicMin index across
    /// entries equal to the max — phase (3) reads <paramref name="maxBoxDev"/>
    /// from device.
    /// </summary>
    public static unsafe void ArgmaxFullyResident(FloatResidentArray logits, int size, nint maxBoxDev, nint idxOutDev)
    {
        Profile("ArgmaxInitSeeds", _argmaxInitSeeds(
            1, 1, 1, BlockDim, 1, 1, 0, _stream,
            maxBoxDev, idxOutDev));

        Profile("ArgmaxFindMaxResident", _argmaxFindMaxResident(
            GridForSize(size), 1, 1, BlockDim, 1, 1, 0, _stream,
            size, ResidentStructCache.Get(logits), maxBoxDev));

        Profile("ArgmaxFindFirstIndexFromMaxBox", _argmaxFindFirstIndexFromMaxBox(
            GridForSize(size), 1, 1, BlockDim, 1, 1, 0, _stream,
            size, ResidentStructCache.Get(logits), maxBoxDev, idxOutDev));
    }

    private static void Check(int err)
    {
        if (err != 0)
            throw new System.ApplicationException($"CUDA kernel returned error {err}: {cuda.GetErrorString((cudaError_t)err)}");
    }
}
