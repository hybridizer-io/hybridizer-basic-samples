using System;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;
using Wmma = Hybridizer.Runtime.CUDAImports.wmma;
using Pipeline = Hybridizer.Runtime.CUDAImports.pipeline;
using Ldm = Hybridizer.Runtime.CUDAImports.ldmatrix;
using Mma = Hybridizer.Runtime.CUDAImports.mma;

namespace WmmaGemm;

// Bind to the C++ helpers in wmma_helpers.cuh. We use IntPtr-typed base
// pointers so the WMMA kernel can be fed device addresses we allocate
// ourselves (cuda.Malloc + cuda.Memcpy) — Hybridizer's array marshaller
// re-copies on every call and dominates the kernel timing otherwise.
[IntrinsicIncludeCUDA("wmma_helpers.cuh")]
internal static class WmmaArr
{
    [IntrinsicFunction("matmul::wmma_load_a_16x16x16_half_row")]
    public static void LoadA(ref Wmma.frag_a_16x16x16_half_row f, IntPtr mptr, int off, uint ldm) { }

    [IntrinsicFunction("matmul::wmma_load_b_16x16x16_half_row")]
    public static void LoadB(ref Wmma.frag_b_16x16x16_half_row f, IntPtr mptr, int off, uint ldm) { }

    [IntrinsicFunction("matmul::wmma_store_c_16x16x16_float")]
    public static void StoreC(IntPtr mptr, int off, Wmma.frag_acc_16x16x16_float f, uint ldm, Wmma.wmma_layout layout) { }

    // Shmem variants: the C# shmem allocator returns `half[]`, which Hybridizer
    // decays to `half*` on device — distinct from the IntPtr/void* path used
    // for global memory above.
    [IntrinsicFunction("matmul::wmma_load_a_shmem_16x16x16_half_row")]
    public static void LoadAShmem(ref Wmma.frag_a_16x16x16_half_row f, half[] mptr, int off, uint ldm) { }

    [IntrinsicFunction("matmul::wmma_load_b_shmem_16x16x16_half_row")]
    public static void LoadBShmem(ref Wmma.frag_b_16x16x16_half_row f, half[] mptr, int off, uint ldm) { }

    [IntrinsicFunction("matmul::global_load_half")]
    public static half GlobalLoadHalf(IntPtr base_, int off) => default;

    // Project-local sugar over the upstream pipeline::op::memcpy_async — folds
    // the (sDst, sOff, gSrc, gOff) shape used by the staging loop into a
    // single 16-byte cp.async at base + offset. When CUDAImports grows
    // half[]-typed overloads, this collapses into one of them.
    [IntrinsicFunction("matmul::cp_async_16B_half")]
    public static void CpAsync16(half[] sDst, int sOff, IntPtr gSrc, int gOff) { }

    // PTX-MMA loaders: warp-collective, write to raw uint32 register storage.
    // r0..r3 are the layout `mma.sync.m16n8k16` expects per its PTX spec.
    [IntrinsicFunction("matmul::ptx_load_a_m16k16")]
    public static void LoadAm16k16(out uint r0, out uint r1, out uint r2, out uint r3, half[] sBase, int rowBase, int colBase, int ldm) { r0 = r1 = r2 = r3 = 0; }

    [IntrinsicFunction("matmul::ptx_load_b_k16n16")]
    public static void LoadBk16n16(out uint r0, out uint r1, out uint r2, out uint r3, half[] sBase, int rowBase, int colBase, int ldm) { r0 = r1 = r2 = r3 = 0; }

    [IntrinsicFunction("matmul::ptx_store_c_m16n8")]
    public static void StoreCm16n8(IntPtr cBase, int rowBase, int colBase, int n, float c0, float c1, float c2, float c3) { }
}

public class Program
{
    private const int Tile = 16;
    private const int RegBlockM = 64;
    private const int RegBlockN = 64;
    private const int RegBlockK = 16;
    private const int RegThreadM = 4;
    private const int RegThreadN = 4;
    private const int BigBlockM = 128;
    private const int BigBlockN = 128;
    private const int BigBlockK = 16;
    private const int BigThreadM = 8;
    private const int BigThreadN = 8;

    private static void Main(string[] args)
    {
        int n = 512;
        if (args.Length > 0 && int.TryParse(args[0], out var parsed) && parsed > 0)
        {
            n = parsed;
        }
        if (n % BigBlockM != 0)
        {
            Console.Error.WriteLine($"Matrix size {n} must be a multiple of {BigBlockM} (big kernel constraint).");
            Environment.Exit(2);
        }

        int profileIters = 1;
        int profileWarmup = 0;
        if (Environment.GetEnvironmentVariable("HYB_PROFILE") == "1")
        {
            profileIters = 10;
            profileWarmup = 2;
        }

        var rng = new Random(42);
        var a = CreateRandomMatrix(n, n, rng);
        var b = CreateRandomMatrix(n, n, rng);
        var cCpu = new float[n * n];

        var swCpu = Stopwatch.StartNew();
        MultiplyCpu(a, b, cCpu, n);
        swCpu.Stop();

        var runner = SatelliteLoader.Load();
        dynamic wrapper = runner.Wrap(new Program());

        var aDev = new FloatResidentArray(n * n);
        var bDev = new FloatResidentArray(n * n);
        Marshal.Copy(a, 0, aDev.HostPointer, n * n);
        Marshal.Copy(b, 0, bDev.HostPointer, n * n);
        aDev.RefreshDevice();
        bDev.RefreshDevice();

        var cNaiveDev = MakeOutputResident(n * n);
        var cTiledDev = MakeOutputResident(n * n);
        var cRegDev   = MakeOutputResident(n * n);
        var cBigDev   = MakeOutputResident(n * n);

        // WMMA tile sizes are 16, so n must be a multiple of 16. We already
        // enforce n % BigBlockM == 0 (BigBlockM = 128), so this is implicit.
        //
        // Hybridizer's marshaller copies managed arrays on every kernel call,
        // which dominates the WMMA timing at small n. We bypass it by holding
        // the half inputs and float output in raw device buffers and passing
        // IntPtr device addresses to the kernel.
        var (aHalfDev, bHalfDev, cWmmaDev) = AllocWmmaDeviceBuffers(n, a, b);
        int wmmaGrid = n / 16;
        var cWmma = new float[n * n];

        // Shmem-tiled WMMA: 64x64 block, 4 warps (128 threads) cooperating.
        IntPtr cWmmaShmemDev;
        cuda.ERROR_CHECK(cuda.Malloc(out cWmmaShmemDev, (long)n * n * sizeof(float)));
        int wmmaShmemGrid = n / 64;
        // Padded shmem footprint: sA = 64 * (16+8) = 1536 halves,
        //                         sB = 16 * (64+8) = 1152 halves.
        int wmmaShmemShmemBytes = (64 * 24 + 16 * 72) * sizeof(ushort);
        var cWmmaShmem = new float[n * n];

        // cp.async double-buffered shmem WMMA: same block geometry, 2x the
        // shmem (one buffer per pipeline stage).
        IntPtr cWmmaAsyncDev;
        cuda.ERROR_CHECK(cuda.Malloc(out cWmmaAsyncDev, (long)n * n * sizeof(float)));
        int wmmaAsyncShmemBytes = 2 * wmmaShmemShmemBytes;
        var cWmmaAsync = new float[n * n];

        // Diagnostic PTX kernel: same 64x64/4-warp/2-stage geometry as
        // wmma-async, but ldmatrix + mma.sync.m16n8k16. Validates the PTX
        // path before scaling to the 128x128/3-stage version.
        IntPtr cWmmaPtxDev;
        cuda.ERROR_CHECK(cuda.Malloc(out cWmmaPtxDev, (long)n * n * sizeof(float)));
        var cWmmaPtx = new float[n * n];

        // Big-tile 3-stage WMMA + ldmatrix. 128x128 block, 8 warps, 3 pipeline
        // stages. Shmem footprint: STAGES * (BLOCK_M*LDA + BLOCK_K*LDB) halves
        // = 3 * (128*24 + 16*136) halves = 3 * 5248 = 15744 halves = 31488 B.
        IntPtr cWmmaBigDev;
        cuda.ERROR_CHECK(cuda.Malloc(out cWmmaBigDev, (long)n * n * sizeof(float)));
        int wmmaBigGrid = n / 128;
        // wmma-big uses 3 stages; wmma-bigPtx uses 4. Compute both.
        int wmmaBigShmemBytes    = 3 * (128 * 24 + 16 * 136) * sizeof(ushort);
        int wmmaBigPtxShmemBytes = 4 * (128 * 24 + 16 * 136) * sizeof(ushort);
        var cWmmaBig = new float[n * n];

        // Full PTX-MMA kernel at the big tile / 3-stage configuration. Same
        // launch params as wmma-big; different shmem only changes occupancy
        // marginally.
        IntPtr cWmmaBigPtxDev;
        cuda.ERROR_CHECK(cuda.Malloc(out cWmmaBigPtxDev, (long)n * n * sizeof(float)));
        var cWmmaBigPtx = new float[n * n];

        // cuBLAS reference: same inputs (half) and output (float) as our WMMA
        // kernels, so the comparison is apples-to-apples for accuracy and a
        // fair upper-bound for perf.
        IntPtr cublasHandle = Cublas.Create();
        IntPtr cBlasDev;
        cuda.ERROR_CHECK(cuda.Malloc(out cBlasDev, (long)n * n * sizeof(float)));
        var cBlas = new float[n * n];

        int naiveGrid = 16;
        int tiledGrid = n / Tile;
        int tiledShmemBytes = Tile * Tile * 2 * sizeof(float);
        int regGrid = n / RegBlockM;
        int regThreadsX = RegBlockN / RegThreadN;
        int regThreadsY = RegBlockM / RegThreadM;
        int regShmemBytes = (RegBlockM * RegBlockK + RegBlockK * RegBlockN) * sizeof(float);
        int bigGrid = n / BigBlockM;
        int bigThreadsX = BigBlockN / BigThreadN;
        int bigThreadsY = BigBlockM / BigThreadM;
        int bigShmemBytes = (BigBlockM * BigBlockK + BigBlockK * BigBlockN) * sizeof(float);

        var naive = BenchmarkKernel("naive", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(naiveGrid, naiveGrid, 32, 32, 1, 0).MultiplyKernel(aDev, bDev, cNaiveDev, n));

        var tiled = BenchmarkKernel("tiled", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(tiledGrid, tiledGrid, Tile, Tile, 1, tiledShmemBytes).MultiplyKernelTiled(aDev, bDev, cTiledDev, n));

        var regTiled = BenchmarkKernel("regtiled", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(regGrid, regGrid, regThreadsX, regThreadsY, 1, regShmemBytes).MultiplyKernelRegTiled(aDev, bDev, cRegDev, n));

        var big = BenchmarkKernel("big", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(bigGrid, bigGrid, bigThreadsX, bigThreadsY, 1, bigShmemBytes).MultiplyKernelBig(aDev, bDev, cBigDev, n));

        var wmma = BenchmarkKernel("wmma", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(wmmaGrid, wmmaGrid, 32, 1, 1, 0).MultiplyKernelWmma(aHalfDev, bHalfDev, cWmmaDev, n));

        var wmmaShmem = BenchmarkKernel("wmma-shmem", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(wmmaShmemGrid, wmmaShmemGrid, 128, 1, 1, wmmaShmemShmemBytes).MultiplyKernelWmmaShmem(aHalfDev, bHalfDev, cWmmaShmemDev, n));

        var wmmaAsync = BenchmarkKernel("wmma-async", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(wmmaShmemGrid, wmmaShmemGrid, 128, 1, 1, wmmaAsyncShmemBytes).MultiplyKernelWmmaShmemAsync(aHalfDev, bHalfDev, cWmmaAsyncDev, n));

        var wmmaPtx = BenchmarkKernel("wmma-ptx", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(wmmaShmemGrid, wmmaShmemGrid, 128, 1, 1, wmmaAsyncShmemBytes).MultiplyKernelWmmaShmemAsyncPtx(aHalfDev, bHalfDev, cWmmaPtxDev, n));

        var wmmaBig = BenchmarkKernel("wmma-big", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(wmmaBigGrid, wmmaBigGrid, 256, 1, 1, wmmaBigShmemBytes).MultiplyKernelWmmaBig3Stage(aHalfDev, bHalfDev, cWmmaBigDev, n));

        var wmmaBigPtx = BenchmarkKernel("wmma-bigptx", profileIters, profileWarmup, () =>
            wrapper.SetDistrib(wmmaBigGrid, wmmaBigGrid, 256, 1, 1, wmmaBigPtxShmemBytes).MultiplyKernelWmmaBigPtx(aHalfDev, bHalfDev, cWmmaBigPtxDev, n));

        var cublas = BenchmarkKernel("cublas", profileIters, profileWarmup, () =>
            Cublas.Hgemm_RowMajor(cublasHandle, n, aHalfDev, bHalfDev, cBlasDev));

        CopyDeviceFloatsToHost(cWmmaDev, cWmma);
        CopyDeviceFloatsToHost(cWmmaShmemDev, cWmmaShmem);
        CopyDeviceFloatsToHost(cWmmaAsyncDev, cWmmaAsync);
        CopyDeviceFloatsToHost(cWmmaPtxDev,   cWmmaPtx);
        CopyDeviceFloatsToHost(cWmmaBigDev, cWmmaBig);
        CopyDeviceFloatsToHost(cWmmaBigPtxDev, cWmmaBigPtx);
        CopyDeviceFloatsToHost(cBlasDev, cBlas);

        var cNaive = HostCopy(cNaiveDev, n * n);
        var cTiled = HostCopy(cTiledDev, n * n);
        var cReg   = HostCopy(cRegDev,   n * n);
        var cBig   = HostCopy(cBigDev,   n * n);

        var (errNaive, _) = Compare(cCpu, cNaive);
        var (errTiled, _) = Compare(cCpu, cTiled);
        var (errReg, _) = Compare(cCpu, cReg);
        var (errBig, _) = Compare(cCpu, cBig);
        var (errWmma, _)      = Compare(cCpu, cWmma);
        var (errWmmaShmem, _) = Compare(cCpu, cWmmaShmem);
        var (errWmmaAsync, _) = Compare(cCpu, cWmmaAsync);
        var (errWmmaPtx, _)   = Compare(cCpu, cWmmaPtx);
        var (errWmmaBig, _)   = Compare(cCpu, cWmmaBig);
        var (errWmmaBigPtx,_) = Compare(cCpu, cWmmaBigPtx);
        var (errBlas, _)      = Compare(cCpu, cBlas);
        var sumCpu = Checksum(cCpu);

        Console.WriteLine($"Matrix size:    {n}x{n}");
        Console.WriteLine($"CPU elapsed:    {swCpu.Elapsed.TotalMilliseconds:F2} ms ({Gflops(n, swCpu.Elapsed.TotalSeconds):F3} GFLOPS)");
        PrintGpu("naive",    naive,    n, profileIters, profileWarmup);
        PrintGpu("tiled",    tiled,    n, profileIters, profileWarmup);
        PrintGpu("regtiled", regTiled, n, profileIters, profileWarmup);
        PrintGpu("big",      big,      n, profileIters, profileWarmup);
        PrintGpu("wmma",     wmma,     n, profileIters, profileWarmup);
        PrintGpu("wmma-sh",  wmmaShmem, n, profileIters, profileWarmup);
        PrintGpu("wmma-as",  wmmaAsync, n, profileIters, profileWarmup);
        PrintGpu("wmma-ptx", wmmaPtx,   n, profileIters, profileWarmup);
        PrintGpu("wmma-big", wmmaBig,   n, profileIters, profileWarmup);
        PrintGpu("wmma-bigp",wmmaBigPtx,n, profileIters, profileWarmup);
        PrintGpu("cublas",   cublas,    n, profileIters, profileWarmup);
        Console.WriteLine($"Speedup tiled/naive:     {naive.bestMs / tiled.bestMs:F2}x");
        Console.WriteLine($"Speedup regtiled/naive:  {naive.bestMs / regTiled.bestMs:F2}x");
        Console.WriteLine($"Speedup big/naive:       {naive.bestMs / big.bestMs:F2}x");
        Console.WriteLine($"Speedup wmma/naive:      {naive.bestMs / wmma.bestMs:F2}x");
        Console.WriteLine($"Speedup wmma-sh/naive:   {naive.bestMs / wmmaShmem.bestMs:F2}x");
        Console.WriteLine($"Speedup wmma-sh/wmma:    {wmma.bestMs / wmmaShmem.bestMs:F2}x");
        Console.WriteLine($"Speedup wmma-as/naive:   {naive.bestMs / wmmaAsync.bestMs:F2}x");
        Console.WriteLine($"Speedup wmma-as/wmma-sh: {wmmaShmem.bestMs / wmmaAsync.bestMs:F2}x");
        Console.WriteLine($"Speedup wmma-big/naive:  {naive.bestMs / wmmaBig.bestMs:F2}x");
        Console.WriteLine($"Speedup wmma-big/wmma-as:{wmmaAsync.bestMs / wmmaBig.bestMs:F2}x");
        Console.WriteLine($"Speedup cublas/naive:    {naive.bestMs / cublas.bestMs:F2}x");
        Console.WriteLine($"Speedup wmma-as/cublas:  {cublas.bestMs / wmmaAsync.bestMs:F2}x (>1 = we beat cuBLAS)");
        Console.WriteLine($"Speedup wmma-big/cublas: {cublas.bestMs / wmmaBig.bestMs:F2}x (>1 = we beat cuBLAS)");
        Console.WriteLine($"Speedup wmma-bigp/cublas:{cublas.bestMs / wmmaBigPtx.bestMs:F2}x (>1 = we beat cuBLAS)");
        Console.WriteLine($"CPU checksum:           {sumCpu:E6}");
        Console.WriteLine($"naive    max abs error: {errNaive:E3}");
        Console.WriteLine($"tiled    max abs error: {errTiled:E3}");
        Console.WriteLine($"regtiled max abs error: {errReg:E3}");
        Console.WriteLine($"big      max abs error: {errBig:E3}");
        // WMMA is half-precision input -> float accumulator, so tolerance is
        // looser than the other kernels (which are float-in float-out).
        Console.WriteLine($"wmma     max abs error: {errWmma:E3}");
        Console.WriteLine($"wmma-sh  max abs error: {errWmmaShmem:E3}");
        Console.WriteLine($"wmma-as  max abs error: {errWmmaAsync:E3}");
        Console.WriteLine($"wmma-ptx max abs error: {errWmmaPtx:E3}");
        Console.WriteLine($"wmma-big max abs error: {errWmmaBig:E3}");
        Console.WriteLine($"wmma-bigp max abs error:{errWmmaBigPtx:E3}");
        Console.WriteLine($"cublas   max abs error: {errBlas:E3}");

        if (errNaive > 1e-2f || errTiled > 1e-2f || errReg > 1e-2f || errBig > 1e-2f)
        {
            Console.Error.WriteLine("FAIL: at least one float-precision kernel exceeded tolerance.");
            Environment.Exit(1);
        }
        // Empirical tolerance for half->float WMMA at n up to a few thousand:
        // accumulated drift is dominated by the half rounding of A and B.
        float wmmaTol = 1.0f * n / 256.0f;
        if (errWmma > wmmaTol)
        {
            Console.Error.WriteLine($"FAIL: WMMA error {errWmma:E3} exceeded tolerance for n={n}.");
            Environment.Exit(1);
        }
        if (errWmmaShmem > wmmaTol)
        {
            Console.Error.WriteLine($"FAIL: WMMA-shmem error {errWmmaShmem:E3} exceeded tolerance for n={n}.");
            Environment.Exit(1);
        }
        if (errWmmaAsync > wmmaTol)
        {
            Console.Error.WriteLine($"FAIL: WMMA-async error {errWmmaAsync:E3} exceeded tolerance for n={n}.");
            Environment.Exit(1);
        }
        if (errWmmaPtx > wmmaTol)
        {
            Console.Error.WriteLine($"FAIL: WMMA-ptx error {errWmmaPtx:E3} exceeded tolerance for n={n}.");
            Environment.Exit(1);
        }
        if (errWmmaBig > wmmaTol)
        {
            Console.Error.WriteLine($"FAIL: WMMA-big error {errWmmaBig:E3} exceeded tolerance for n={n}.");
            Environment.Exit(1);
        }
        if (errWmmaBigPtx > wmmaTol)
        {
            Console.Error.WriteLine($"FAIL: WMMA-bigPtx error {errWmmaBigPtx:E3} exceeded tolerance for n={n}.");
            Environment.Exit(1);
        }
        if (errBlas > wmmaTol)
        {
            Console.Error.WriteLine($"FAIL: cuBLAS error {errBlas:E3} exceeded tolerance for n={n}.");
            Environment.Exit(1);
        }
        Cublas.Destroy(cublasHandle);
        Console.WriteLine("OK");
    }

    [IntrinsicFunction("__syncthreads")]
    private static void SyncThreads() { }

    [EntryPoint]
    public static void MultiplyKernel(FloatResidentArray a, FloatResidentArray b, FloatResidentArray c, int n)
    {
        for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < n; i += blockDim.y * gridDim.y)
        {
            for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < n; j += blockDim.x * gridDim.x)
            {
                float acc = 0.0f;
                for (int k = 0; k < n; ++k)
                {
                    acc += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = acc;
            }
        }
    }

    [EntryPoint]
    public static void MultiplyKernelTiled(FloatResidentArray a, FloatResidentArray b, FloatResidentArray c, int n)
    {
        SharedMemoryAllocator<float> allocator = new SharedMemoryAllocator<float>();
        float[] cacheA = allocator.allocate(blockDim.y * blockDim.x);
        float[] cacheB = allocator.allocate(blockDim.y * blockDim.x);

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        for (int by = blockIdx.y; by < n / blockDim.y; by += gridDim.y)
        {
            for (int bx = blockIdx.x; bx < n / blockDim.x; bx += gridDim.x)
            {
                int i = by * blockDim.y + ty;
                int j = bx * blockDim.x + tx;

                float acc = 0.0f;
                int numTiles = n / blockDim.x;
                for (int t = 0; t < numTiles; ++t)
                {
                    cacheA[ty * blockDim.x + tx] = a[i * n + (t * blockDim.x + tx)];
                    cacheB[ty * blockDim.x + tx] = b[(t * blockDim.y + ty) * n + j];

                    SyncThreads();

                    for (int k = 0; k < blockDim.x; ++k)
                    {
                        acc += cacheA[ty * blockDim.x + k] * cacheB[k * blockDim.x + tx];
                    }

                    SyncThreads();
                }

                c[i * n + j] = acc;
            }
        }
    }

    [EntryPoint]
    public static void MultiplyKernelRegTiled(FloatResidentArray a, FloatResidentArray b, FloatResidentArray c, int n)
    {
        const int BLOCK_M = 64;
        const int BLOCK_N = 64;
        const int BLOCK_K = 16;
        const int THREAD_M = 4;
        const int THREAD_N = 4;
        const int TY_STRIDE = BLOCK_M / THREAD_M;
        const int TX_STRIDE = BLOCK_N / THREAD_N;
        const int LOADS_PER_THREAD = (BLOCK_M * BLOCK_K) / 256;

        SharedMemoryAllocator<float> allocator = new SharedMemoryAllocator<float>();
        float[] sA = allocator.allocate(BLOCK_M * BLOCK_K);
        float[] sB = allocator.allocate(BLOCK_K * BLOCK_N);

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tid = ty * TX_STRIDE + tx;

        int blockRow = blockIdx.y * BLOCK_M;
        int blockCol = blockIdx.x * BLOCK_N;

        float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
        float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
        float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
        float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

        for (int kTile = 0; kTile < n; kTile += BLOCK_K)
        {
            for (int li = 0; li < LOADS_PER_THREAD; ++li)
            {
                int flat = tid + li * 256;
                int row = flat / BLOCK_K;
                int col = flat % BLOCK_K;
                sA[flat] = a[(blockRow + row) * n + (kTile + col)];
            }

            for (int li = 0; li < LOADS_PER_THREAD; ++li)
            {
                int flat = tid + li * 256;
                int row = flat / BLOCK_N;
                int col = flat % BLOCK_N;
                sB[flat] = b[(kTile + row) * n + (blockCol + col)];
            }

            SyncThreads();

            for (int kk = 0; kk < BLOCK_K; ++kk)
            {
                float a0 = sA[(ty + 0 * TY_STRIDE) * BLOCK_K + kk];
                float a1 = sA[(ty + 1 * TY_STRIDE) * BLOCK_K + kk];
                float a2 = sA[(ty + 2 * TY_STRIDE) * BLOCK_K + kk];
                float a3 = sA[(ty + 3 * TY_STRIDE) * BLOCK_K + kk];

                float b0 = sB[kk * BLOCK_N + (tx + 0 * TX_STRIDE)];
                float b1 = sB[kk * BLOCK_N + (tx + 1 * TX_STRIDE)];
                float b2 = sB[kk * BLOCK_N + (tx + 2 * TX_STRIDE)];
                float b3 = sB[kk * BLOCK_N + (tx + 3 * TX_STRIDE)];

                acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
                acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
                acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
                acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
            }

            SyncThreads();
        }

        int outRow = blockRow + ty;
        int outCol = blockCol + tx;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = acc00;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = acc01;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = acc02;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = acc03;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = acc10;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = acc11;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = acc12;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = acc13;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = acc20;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = acc21;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = acc22;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = acc23;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = acc30;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = acc31;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = acc32;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = acc33;
    }

    [EntryPoint]
    public static void MultiplyKernelBig(FloatResidentArray a, FloatResidentArray b, FloatResidentArray c, int n)
    {
        const int BLOCK_M = 128;
        const int BLOCK_N = 128;
        const int BLOCK_K = 16;
        const int THREAD_M = 8;
        const int THREAD_N = 8;
        const int TY_STRIDE = BLOCK_M / THREAD_M;
        const int TX_STRIDE = BLOCK_N / THREAD_N;
        const int LOADS_PER_THREAD_A = (BLOCK_M * BLOCK_K) / 256;
        const int LOADS_PER_THREAD_B = (BLOCK_K * BLOCK_N) / 256;

        SharedMemoryAllocator<float> allocator = new SharedMemoryAllocator<float>();
        float[] sA = allocator.allocate(BLOCK_M * BLOCK_K);
        float[] sB = allocator.allocate(BLOCK_K * BLOCK_N);

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tid = ty * TX_STRIDE + tx;

        int blockRow = blockIdx.y * BLOCK_M;
        int blockCol = blockIdx.x * BLOCK_N;

        float ac00 = 0, ac01 = 0, ac02 = 0, ac03 = 0, ac04 = 0, ac05 = 0, ac06 = 0, ac07 = 0;
        float ac10 = 0, ac11 = 0, ac12 = 0, ac13 = 0, ac14 = 0, ac15 = 0, ac16 = 0, ac17 = 0;
        float ac20 = 0, ac21 = 0, ac22 = 0, ac23 = 0, ac24 = 0, ac25 = 0, ac26 = 0, ac27 = 0;
        float ac30 = 0, ac31 = 0, ac32 = 0, ac33 = 0, ac34 = 0, ac35 = 0, ac36 = 0, ac37 = 0;
        float ac40 = 0, ac41 = 0, ac42 = 0, ac43 = 0, ac44 = 0, ac45 = 0, ac46 = 0, ac47 = 0;
        float ac50 = 0, ac51 = 0, ac52 = 0, ac53 = 0, ac54 = 0, ac55 = 0, ac56 = 0, ac57 = 0;
        float ac60 = 0, ac61 = 0, ac62 = 0, ac63 = 0, ac64 = 0, ac65 = 0, ac66 = 0, ac67 = 0;
        float ac70 = 0, ac71 = 0, ac72 = 0, ac73 = 0, ac74 = 0, ac75 = 0, ac76 = 0, ac77 = 0;

        for (int kTile = 0; kTile < n; kTile += BLOCK_K)
        {
            for (int li = 0; li < LOADS_PER_THREAD_A; ++li)
            {
                int flat = tid + li * 256;
                int row = flat / BLOCK_K;
                int col = flat % BLOCK_K;
                sA[flat] = a[(blockRow + row) * n + (kTile + col)];
            }

            for (int li = 0; li < LOADS_PER_THREAD_B; ++li)
            {
                int flat = tid + li * 256;
                int row = flat / BLOCK_N;
                int col = flat % BLOCK_N;
                sB[flat] = b[(kTile + row) * n + (blockCol + col)];
            }

            SyncThreads();

            for (int kk = 0; kk < BLOCK_K; ++kk)
            {
                float a0 = sA[(ty + 0 * TY_STRIDE) * BLOCK_K + kk];
                float a1 = sA[(ty + 1 * TY_STRIDE) * BLOCK_K + kk];
                float a2 = sA[(ty + 2 * TY_STRIDE) * BLOCK_K + kk];
                float a3 = sA[(ty + 3 * TY_STRIDE) * BLOCK_K + kk];
                float a4 = sA[(ty + 4 * TY_STRIDE) * BLOCK_K + kk];
                float a5 = sA[(ty + 5 * TY_STRIDE) * BLOCK_K + kk];
                float a6 = sA[(ty + 6 * TY_STRIDE) * BLOCK_K + kk];
                float a7 = sA[(ty + 7 * TY_STRIDE) * BLOCK_K + kk];

                float b0 = sB[kk * BLOCK_N + (tx + 0 * TX_STRIDE)];
                float b1 = sB[kk * BLOCK_N + (tx + 1 * TX_STRIDE)];
                float b2 = sB[kk * BLOCK_N + (tx + 2 * TX_STRIDE)];
                float b3 = sB[kk * BLOCK_N + (tx + 3 * TX_STRIDE)];
                float b4 = sB[kk * BLOCK_N + (tx + 4 * TX_STRIDE)];
                float b5 = sB[kk * BLOCK_N + (tx + 5 * TX_STRIDE)];
                float b6 = sB[kk * BLOCK_N + (tx + 6 * TX_STRIDE)];
                float b7 = sB[kk * BLOCK_N + (tx + 7 * TX_STRIDE)];

                ac00 += a0 * b0; ac01 += a0 * b1; ac02 += a0 * b2; ac03 += a0 * b3; ac04 += a0 * b4; ac05 += a0 * b5; ac06 += a0 * b6; ac07 += a0 * b7;
                ac10 += a1 * b0; ac11 += a1 * b1; ac12 += a1 * b2; ac13 += a1 * b3; ac14 += a1 * b4; ac15 += a1 * b5; ac16 += a1 * b6; ac17 += a1 * b7;
                ac20 += a2 * b0; ac21 += a2 * b1; ac22 += a2 * b2; ac23 += a2 * b3; ac24 += a2 * b4; ac25 += a2 * b5; ac26 += a2 * b6; ac27 += a2 * b7;
                ac30 += a3 * b0; ac31 += a3 * b1; ac32 += a3 * b2; ac33 += a3 * b3; ac34 += a3 * b4; ac35 += a3 * b5; ac36 += a3 * b6; ac37 += a3 * b7;
                ac40 += a4 * b0; ac41 += a4 * b1; ac42 += a4 * b2; ac43 += a4 * b3; ac44 += a4 * b4; ac45 += a4 * b5; ac46 += a4 * b6; ac47 += a4 * b7;
                ac50 += a5 * b0; ac51 += a5 * b1; ac52 += a5 * b2; ac53 += a5 * b3; ac54 += a5 * b4; ac55 += a5 * b5; ac56 += a5 * b6; ac57 += a5 * b7;
                ac60 += a6 * b0; ac61 += a6 * b1; ac62 += a6 * b2; ac63 += a6 * b3; ac64 += a6 * b4; ac65 += a6 * b5; ac66 += a6 * b6; ac67 += a6 * b7;
                ac70 += a7 * b0; ac71 += a7 * b1; ac72 += a7 * b2; ac73 += a7 * b3; ac74 += a7 * b4; ac75 += a7 * b5; ac76 += a7 * b6; ac77 += a7 * b7;
            }

            SyncThreads();
        }

        int outRow = blockRow + ty;
        int outCol = blockCol + tx;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = ac00;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = ac01;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = ac02;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = ac03;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 4 * TX_STRIDE] = ac04;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 5 * TX_STRIDE] = ac05;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 6 * TX_STRIDE] = ac06;
        c[(outRow + 0 * TY_STRIDE) * n + outCol + 7 * TX_STRIDE] = ac07;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = ac10;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = ac11;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = ac12;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = ac13;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 4 * TX_STRIDE] = ac14;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 5 * TX_STRIDE] = ac15;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 6 * TX_STRIDE] = ac16;
        c[(outRow + 1 * TY_STRIDE) * n + outCol + 7 * TX_STRIDE] = ac17;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = ac20;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = ac21;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = ac22;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = ac23;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 4 * TX_STRIDE] = ac24;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 5 * TX_STRIDE] = ac25;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 6 * TX_STRIDE] = ac26;
        c[(outRow + 2 * TY_STRIDE) * n + outCol + 7 * TX_STRIDE] = ac27;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = ac30;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = ac31;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = ac32;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = ac33;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 4 * TX_STRIDE] = ac34;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 5 * TX_STRIDE] = ac35;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 6 * TX_STRIDE] = ac36;
        c[(outRow + 3 * TY_STRIDE) * n + outCol + 7 * TX_STRIDE] = ac37;
        c[(outRow + 4 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = ac40;
        c[(outRow + 4 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = ac41;
        c[(outRow + 4 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = ac42;
        c[(outRow + 4 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = ac43;
        c[(outRow + 4 * TY_STRIDE) * n + outCol + 4 * TX_STRIDE] = ac44;
        c[(outRow + 4 * TY_STRIDE) * n + outCol + 5 * TX_STRIDE] = ac45;
        c[(outRow + 4 * TY_STRIDE) * n + outCol + 6 * TX_STRIDE] = ac46;
        c[(outRow + 4 * TY_STRIDE) * n + outCol + 7 * TX_STRIDE] = ac47;
        c[(outRow + 5 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = ac50;
        c[(outRow + 5 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = ac51;
        c[(outRow + 5 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = ac52;
        c[(outRow + 5 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = ac53;
        c[(outRow + 5 * TY_STRIDE) * n + outCol + 4 * TX_STRIDE] = ac54;
        c[(outRow + 5 * TY_STRIDE) * n + outCol + 5 * TX_STRIDE] = ac55;
        c[(outRow + 5 * TY_STRIDE) * n + outCol + 6 * TX_STRIDE] = ac56;
        c[(outRow + 5 * TY_STRIDE) * n + outCol + 7 * TX_STRIDE] = ac57;
        c[(outRow + 6 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = ac60;
        c[(outRow + 6 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = ac61;
        c[(outRow + 6 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = ac62;
        c[(outRow + 6 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = ac63;
        c[(outRow + 6 * TY_STRIDE) * n + outCol + 4 * TX_STRIDE] = ac64;
        c[(outRow + 6 * TY_STRIDE) * n + outCol + 5 * TX_STRIDE] = ac65;
        c[(outRow + 6 * TY_STRIDE) * n + outCol + 6 * TX_STRIDE] = ac66;
        c[(outRow + 6 * TY_STRIDE) * n + outCol + 7 * TX_STRIDE] = ac67;
        c[(outRow + 7 * TY_STRIDE) * n + outCol + 0 * TX_STRIDE] = ac70;
        c[(outRow + 7 * TY_STRIDE) * n + outCol + 1 * TX_STRIDE] = ac71;
        c[(outRow + 7 * TY_STRIDE) * n + outCol + 2 * TX_STRIDE] = ac72;
        c[(outRow + 7 * TY_STRIDE) * n + outCol + 3 * TX_STRIDE] = ac73;
        c[(outRow + 7 * TY_STRIDE) * n + outCol + 4 * TX_STRIDE] = ac74;
        c[(outRow + 7 * TY_STRIDE) * n + outCol + 5 * TX_STRIDE] = ac75;
        c[(outRow + 7 * TY_STRIDE) * n + outCol + 6 * TX_STRIDE] = ac76;
        c[(outRow + 7 * TY_STRIDE) * n + outCol + 7 * TX_STRIDE] = ac77;
    }

    // 16x16x16 half->float WMMA GEMM. Grid = (n/16, n/16), block = 32 threads
    // (one warp). Each warp computes one 16x16 output tile by sweeping K in
    // 16-wide chunks of A (row-major) and B (row-major). Pointers come in as
    // IntPtr device addresses; the WmmaArr helpers cast them to half*/float*.
    [EntryPoint]
    public static void MultiplyKernelWmma(IntPtr a, IntPtr b, IntPtr c, int n)
    {
        var fragA = new Wmma.frag_a_16x16x16_half_row();
        var fragB = new Wmma.frag_b_16x16x16_half_row();
        var acc   = new Wmma.frag_acc_16x16x16_float();
        Wmma.op.fill_fragment(ref acc, 0.0f);

        int row = blockIdx.y * 16;
        int col = blockIdx.x * 16;

        for (int k = 0; k < n; k += 16)
        {
            // Row-major: A[row:row+16, k:k+16] starts at row*n + k; ldm = n.
            //            B[k:k+16, col:col+16] starts at k*n + col; ldm = n.
            WmmaArr.LoadA(ref fragA, a, row * n + k, (uint)n);
            WmmaArr.LoadB(ref fragB, b, k * n + col, (uint)n);
            Wmma.op.mma_sync(ref acc, fragA, fragB, acc);
        }

        WmmaArr.StoreC(c, row * n + col, acc, (uint)n, Wmma.wmma_layout.mem_row_major);
    }

    // Shared-memory-tiled WMMA GEMM. Each block computes a 64x64 output tile
    // with 4 warps (128 threads) cooperating; each warp owns a 32x32 quadrant
    // = 4 of the 16 WMMA fragments per K-iter. A single 64x16/16x64 chunk of A
    // and B is loaded into shmem per K-iter and reused by all 4 warps.
    //
    // Each row in shmem is padded by SKEW halves on the inner dimension so
    // that the row stride mod 128 bytes (the bank-cycle period for fp16) is
    // non-zero. Without this, ncu reports an average 20-way bank conflict on
    // every `load_matrix_sync` from sA (80% of shmem load wavefronts are
    // conflicted). With SKEW=8 we go from 80% conflict → ~0%.
    [EntryPoint]
    public static void MultiplyKernelWmmaShmem(IntPtr a, IntPtr b, IntPtr c, int n)
    {
        const int BLOCK_M = 64;
        const int BLOCK_N = 64;
        const int BLOCK_K = 16;
        const int SKEW   = 8;             // halves of padding on the inner dim
        const int LDA    = BLOCK_K + SKEW; // 24 halves = 48 bytes per row of sA
        const int LDB    = BLOCK_N + SKEW; // 72 halves = 144 bytes per row of sB
        const int WARPS_N = 2;            // 2x2 grid of warps over the block tile
        const int THREADS = 128;          // 4 warps * 32 lanes
        const int LOADS_PER_THREAD = 8;   // (BLOCK_M*BLOCK_K)/THREADS == (BLOCK_K*BLOCK_N)/THREADS == 8

        SharedMemoryAllocator<half> allocator = new SharedMemoryAllocator<half>();
        half[] sA = allocator.allocate(BLOCK_M * LDA);
        half[] sB = allocator.allocate(BLOCK_K * LDB);

        int tid = threadIdx.x;
        int warpId = tid / 32;
        int warpRow = warpId / WARPS_N;   // 0 or 1
        int warpCol = warpId % WARPS_N;   // 0 or 1

        var a0 = new Wmma.frag_a_16x16x16_half_row();
        var a1 = new Wmma.frag_a_16x16x16_half_row();
        var b0 = new Wmma.frag_b_16x16x16_half_row();
        var b1 = new Wmma.frag_b_16x16x16_half_row();
        var c00 = new Wmma.frag_acc_16x16x16_float();
        var c01 = new Wmma.frag_acc_16x16x16_float();
        var c10 = new Wmma.frag_acc_16x16x16_float();
        var c11 = new Wmma.frag_acc_16x16x16_float();
        Wmma.op.fill_fragment(ref c00, 0.0f);
        Wmma.op.fill_fragment(ref c01, 0.0f);
        Wmma.op.fill_fragment(ref c10, 0.0f);
        Wmma.op.fill_fragment(ref c11, 0.0f);

        int blockRow = blockIdx.y * BLOCK_M;
        int blockCol = blockIdx.x * BLOCK_N;

        for (int kTile = 0; kTile < n; kTile += BLOCK_K)
        {
            // Cooperatively stage sA[BLOCK_M, BLOCK_K] and sB[BLOCK_K, BLOCK_N]
            // — each thread loads 8 elements per buffer (1024 elements / 128 threads).
            // The destination shmem row stride is LDA/LDB (padded); the source
            // global row stride is n.
            for (int li = 0; li < LOADS_PER_THREAD; li++)
            {
                int flat = tid + li * THREADS;
                int row = flat / BLOCK_K;
                int col = flat % BLOCK_K;
                sA[row * LDA + col] = WmmaArr.GlobalLoadHalf(a, (blockRow + row) * n + (kTile + col));
            }
            for (int li = 0; li < LOADS_PER_THREAD; li++)
            {
                int flat = tid + li * THREADS;
                int row = flat / BLOCK_N;
                int col = flat % BLOCK_N;
                sB[row * LDB + col] = WmmaArr.GlobalLoadHalf(b, (kTile + row) * n + (blockCol + col));
            }

            SyncThreads();

            // 2 A frags (rows warpRow*32 + {0,16}) and 2 B frags (cols warpCol*32 + {0,16}).
            // ldm is the padded shmem row stride, not BLOCK_K / BLOCK_N.
            WmmaArr.LoadAShmem(ref a0, sA, (warpRow * 32 + 0)  * LDA, (uint)LDA);
            WmmaArr.LoadAShmem(ref a1, sA, (warpRow * 32 + 16) * LDA, (uint)LDA);
            WmmaArr.LoadBShmem(ref b0, sB, warpCol * 32 + 0,  (uint)LDB);
            WmmaArr.LoadBShmem(ref b1, sB, warpCol * 32 + 16, (uint)LDB);

            // 2x2 outer product into the accumulators.
            Wmma.op.mma_sync(ref c00, a0, b0, c00);
            Wmma.op.mma_sync(ref c01, a0, b1, c01);
            Wmma.op.mma_sync(ref c10, a1, b0, c10);
            Wmma.op.mma_sync(ref c11, a1, b1, c11);

            SyncThreads();
        }

        int rowBase = blockRow + warpRow * 32;
        int colBase = blockCol + warpCol * 32;
        WmmaArr.StoreC(c, (rowBase + 0)  * n + (colBase + 0),  c00, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 0)  * n + (colBase + 16), c01, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 16) * n + (colBase + 0),  c10, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 16) * n + (colBase + 16), c11, (uint)n, Wmma.wmma_layout.mem_row_major);
    }

    // cp.async double-buffered WMMA. Same 64x64 block / 4-warp layout as
    // MultiplyKernelWmmaShmem, but with two shmem buffers (sA[2*64*24],
    // sB[2*16*72]) and a producer/consumer pipeline:
    //
    //   prologue:   issue cp.async into buffer 0 for K=0;            commit
    //   for kTile in [0..n):
    //       if kTile+BLOCK_K < n: issue cp.async into the OTHER buf; commit
    //       wait_prior(1)  iff a next-tile commit is in flight, else wait_prior(0)
    //       __syncthreads
    //       run mma_sync chain against the *current* buffer
    //       __syncthreads
    //       swap curBuf
    //
    // The fundamental shape: stage K+1 while computing K. The L1TEX scoreboard
    // stall ncu flagged on the synchronous shmem kernel turns into a wait_prior
    // that completes immediately by the time the mma_syncs finish, so the
    // global-load latency hides under tensor-core work.
    //
    // Each thread issues exactly one cp.async per buffer per K-iter (8 halves
    // = 16 bytes per copy, 128 threads × 8 = 1024 halves per buffer = exactly
    // BLOCK_M*BLOCK_K and BLOCK_K*BLOCK_N). Pad cells in sA / sB are never
    // written; load_matrix_sync skips them via ldm.
    [EntryPoint]
    public static void MultiplyKernelWmmaShmemAsync(IntPtr a, IntPtr b, IntPtr c, int n)
    {
        const int BLOCK_M = 64;
        const int BLOCK_N = 64;
        const int BLOCK_K = 16;
        const int SKEW    = 8;
        const int LDA     = BLOCK_K + SKEW;     // 24 halves per padded sA row
        const int LDB     = BLOCK_N + SKEW;     // 72 halves per padded sB row
        const int SA_BUF  = BLOCK_M * LDA;      // 1536 halves per A buffer
        const int SB_BUF  = BLOCK_K * LDB;      // 1152 halves per B buffer
        const int WARPS_N = 2;

        SharedMemoryAllocator<half> allocator = new SharedMemoryAllocator<half>();
        half[] sA = allocator.allocate(2 * SA_BUF);
        half[] sB = allocator.allocate(2 * SB_BUF);

        int tid = threadIdx.x;
        int warpId = tid / 32;
        int warpRow = warpId / WARPS_N;
        int warpCol = warpId % WARPS_N;

        // cp.async slot assignment — each thread copies 8 contiguous halves.
        // sA has BLOCK_M=64 rows × BLOCK_K=16 cols → 2 slots/row × 64 rows = 128 slots.
        // sB has BLOCK_K=16 rows × BLOCK_N=64 cols → 8 slots/row × 16 rows = 128 slots.
        int aRow = tid / 2;
        int aCol = (tid % 2) * 8;
        int bRow = tid / 8;
        int bCol = (tid % 8) * 8;
        int aPadOff = aRow * LDA + aCol;
        int bPadOff = bRow * LDB + bCol;

        int blockRow = blockIdx.y * BLOCK_M;
        int blockCol = blockIdx.x * BLOCK_N;

        // Prologue: stage K=0 into buffer 0.
        WmmaArr.CpAsync16(sA, 0 * SA_BUF + aPadOff, a, (blockRow + aRow) * n + (0 + aCol));
        WmmaArr.CpAsync16(sB, 0 * SB_BUF + bPadOff, b, (0 + bRow) * n + (blockCol + bCol));
        Pipeline.op.commit();

        var a0 = new Wmma.frag_a_16x16x16_half_row();
        var a1 = new Wmma.frag_a_16x16x16_half_row();
        var b0 = new Wmma.frag_b_16x16x16_half_row();
        var b1 = new Wmma.frag_b_16x16x16_half_row();
        var c00 = new Wmma.frag_acc_16x16x16_float();
        var c01 = new Wmma.frag_acc_16x16x16_float();
        var c10 = new Wmma.frag_acc_16x16x16_float();
        var c11 = new Wmma.frag_acc_16x16x16_float();
        Wmma.op.fill_fragment(ref c00, 0.0f);
        Wmma.op.fill_fragment(ref c01, 0.0f);
        Wmma.op.fill_fragment(ref c10, 0.0f);
        Wmma.op.fill_fragment(ref c11, 0.0f);

        int curBuf = 0;

        for (int kTile = 0; kTile < n; kTile += BLOCK_K)
        {
            int nextKTile = kTile + BLOCK_K;
            int nextBuf = 1 - curBuf;

            if (nextKTile < n)
            {
                // Stage next K-tile into the other buffer. After this commit,
                // 2 groups are in flight; wait_prior(1) drains the current.
                WmmaArr.CpAsync16(sA, nextBuf * SA_BUF + aPadOff, a, (blockRow + aRow) * n + (nextKTile + aCol));
                WmmaArr.CpAsync16(sB, nextBuf * SB_BUF + bPadOff, b, (nextKTile + bRow) * n + (blockCol + bCol));
                Pipeline.op.commit();
                Pipeline.op.wait_prior(1UL);
            }
            else
            {
                // Last K-iter — just drain the one in-flight group.
                Pipeline.op.wait_prior(0UL);
            }

            SyncThreads();

            int saBase = curBuf * SA_BUF;
            int sbBase = curBuf * SB_BUF;
            WmmaArr.LoadAShmem(ref a0, sA, saBase + (warpRow * 32 + 0)  * LDA, (uint)LDA);
            WmmaArr.LoadAShmem(ref a1, sA, saBase + (warpRow * 32 + 16) * LDA, (uint)LDA);
            WmmaArr.LoadBShmem(ref b0, sB, sbBase + warpCol * 32 + 0,  (uint)LDB);
            WmmaArr.LoadBShmem(ref b1, sB, sbBase + warpCol * 32 + 16, (uint)LDB);

            Wmma.op.mma_sync(ref c00, a0, b0, c00);
            Wmma.op.mma_sync(ref c01, a0, b1, c01);
            Wmma.op.mma_sync(ref c10, a1, b0, c10);
            Wmma.op.mma_sync(ref c11, a1, b1, c11);

            SyncThreads();

            curBuf = nextBuf;
        }

        int rowBase = blockRow + warpRow * 32;
        int colBase = blockCol + warpCol * 32;
        WmmaArr.StoreC(c, (rowBase + 0)  * n + (colBase + 0),  c00, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 0)  * n + (colBase + 16), c01, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 16) * n + (colBase + 0),  c10, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 16) * n + (colBase + 16), c11, (uint)n, Wmma.wmma_layout.mem_row_major);
    }

    // 64x64 / 4-warp / 2-stage PTX kernel — same block geometry as
    // MultiplyKernelWmmaShmemAsync but the wmma::fragment + wmma::mma_sync
    // path is replaced with raw uint32/float register storage + ldmatrix +
    // mma.sync.m16n8k16. This is the CUTLASS shape: holds A as 8 uint32, B as
    // 8 uint32, the 32-element fp32 accumulator across explicit locals, and
    // calls 8 `mma.m16n8k16` per K-iter per warp (2 m16 × 4 n8). Validation
    // step before we scale to 128x128 / 8 warps / 3 stages.
    [EntryPoint]
    public static void MultiplyKernelWmmaShmemAsyncPtx(IntPtr a, IntPtr b, IntPtr c, int n)
    {
        const int BLOCK_M = 64;
        const int BLOCK_N = 64;
        const int BLOCK_K = 16;
        const int SKEW    = 8;
        const int LDA     = BLOCK_K + SKEW;     // 24
        const int LDB     = BLOCK_N + SKEW;     // 72
        const int SA_BUF  = BLOCK_M * LDA;      // 1536
        const int SB_BUF  = BLOCK_K * LDB;      // 1152
        const int WARPS_N = 2;

        SharedMemoryAllocator<half> allocator = new SharedMemoryAllocator<half>();
        half[] sA = allocator.allocate(2 * SA_BUF);
        half[] sB = allocator.allocate(2 * SB_BUF);

        int tid = threadIdx.x;
        int warpId = tid / 32;
        int warpRow = warpId / WARPS_N;         // 0 or 1
        int warpCol = warpId % WARPS_N;         // 0 or 1

        // cp.async slot assignment — same as wmma-shmem-async.
        int aRow = tid / 2;
        int aCol = (tid % 2) * 8;
        int bRow = tid / 8;
        int bCol = (tid % 8) * 8;
        int aPadOff = aRow * LDA + aCol;
        int bPadOff = bRow * LDB + bCol;

        int blockRow = blockIdx.y * BLOCK_M;
        int blockCol = blockIdx.x * BLOCK_N;

        // Per-warp area = 32M × 32N = 2 m16 × 2 n16 = 2 m16 × 4 n8.
        // A: 2 m16k16 ldmatrix.x4 → 2 × 4 = 8 uint regs per lane
        // B: 2 k16n16 ldmatrix.x4.trans → 2 × 4 = 8 uint regs per lane
        // C: 8 m16n8 mmas × 4 fp32 = 32 fp32 per lane
        uint a0_0 = 0, a0_1 = 0, a0_2 = 0, a0_3 = 0;
        uint a1_0 = 0, a1_1 = 0, a1_2 = 0, a1_3 = 0;
        uint b0_0 = 0, b0_1 = 0, b0_2 = 0, b0_3 = 0;
        uint b1_0 = 0, b1_1 = 0, b1_2 = 0, b1_3 = 0;
        // Accumulators: c{m}{n}_{r} for m∈{0,1} (m16 slice), n∈{0,1,2,3} (n8 slice), r∈{0..3}
        float c00_0 = 0, c00_1 = 0, c00_2 = 0, c00_3 = 0;
        float c01_0 = 0, c01_1 = 0, c01_2 = 0, c01_3 = 0;
        float c02_0 = 0, c02_1 = 0, c02_2 = 0, c02_3 = 0;
        float c03_0 = 0, c03_1 = 0, c03_2 = 0, c03_3 = 0;
        float c10_0 = 0, c10_1 = 0, c10_2 = 0, c10_3 = 0;
        float c11_0 = 0, c11_1 = 0, c11_2 = 0, c11_3 = 0;
        float c12_0 = 0, c12_1 = 0, c12_2 = 0, c12_3 = 0;
        float c13_0 = 0, c13_1 = 0, c13_2 = 0, c13_3 = 0;

        // Prologue: stage K=0 into buffer 0.
        WmmaArr.CpAsync16(sA, 0 * SA_BUF + aPadOff, a, (blockRow + aRow) * n + aCol);
        WmmaArr.CpAsync16(sB, 0 * SB_BUF + bPadOff, b, bRow * n + (blockCol + bCol));
        Pipeline.op.commit();

        int curBuf = 0;

        for (int kTile = 0; kTile < n; kTile += BLOCK_K)
        {
            int nextKTile = kTile + BLOCK_K;
            int nextBuf = 1 - curBuf;

            if (nextKTile < n)
            {
                WmmaArr.CpAsync16(sA, nextBuf * SA_BUF + aPadOff, a, (blockRow + aRow) * n + (nextKTile + aCol));
                WmmaArr.CpAsync16(sB, nextBuf * SB_BUF + bPadOff, b, (nextKTile + bRow) * n + (blockCol + bCol));
                Pipeline.op.commit();
                Pipeline.op.wait_prior(1UL);
            }
            else
            {
                Pipeline.op.wait_prior(0UL);
            }

            SyncThreads();

            int aStageRow = curBuf * BLOCK_M;
            int bStageRow = curBuf * BLOCK_K;
            int warpRowBase = warpRow * 32;
            int warpColBase = warpCol * 32;

            // 2 A tiles (m16 slices at row warpRowBase + {0,16}).
            WmmaArr.LoadAm16k16(out a0_0, out a0_1, out a0_2, out a0_3, sA, aStageRow + warpRowBase + 0 , 0, LDA);
            WmmaArr.LoadAm16k16(out a1_0, out a1_1, out a1_2, out a1_3, sA, aStageRow + warpRowBase + 16, 0, LDA);
            // 2 B tiles (n16 slices at col warpColBase + {0,16}).
            WmmaArr.LoadBk16n16(out b0_0, out b0_1, out b0_2, out b0_3, sB, bStageRow, warpColBase + 0 , LDB);
            WmmaArr.LoadBk16n16(out b1_0, out b1_1, out b1_2, out b1_3, sB, bStageRow, warpColBase + 16, LDB);

            // 8 mma.m16n8k16. With column-major submatrix arrangement, each B
            // k16n16 tile feeds 2 n8 mmas: cols 0..7 use (r0, r1) (top-left +
            // bottom-left = full k for n=0..7), cols 8..15 use (r2, r3).
            // m16=0
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c00_0, ref c00_1, ref c00_2, ref c00_3, a0_0, a0_1, a0_2, a0_3, b0_0, b0_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c01_0, ref c01_1, ref c01_2, ref c01_3, a0_0, a0_1, a0_2, a0_3, b0_2, b0_3);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c02_0, ref c02_1, ref c02_2, ref c02_3, a0_0, a0_1, a0_2, a0_3, b1_0, b1_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c03_0, ref c03_1, ref c03_2, ref c03_3, a0_0, a0_1, a0_2, a0_3, b1_2, b1_3);
            // m16=1
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c10_0, ref c10_1, ref c10_2, ref c10_3, a1_0, a1_1, a1_2, a1_3, b0_0, b0_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c11_0, ref c11_1, ref c11_2, ref c11_3, a1_0, a1_1, a1_2, a1_3, b0_2, b0_3);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c12_0, ref c12_1, ref c12_2, ref c12_3, a1_0, a1_1, a1_2, a1_3, b1_0, b1_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c13_0, ref c13_1, ref c13_2, ref c13_3, a1_0, a1_1, a1_2, a1_3, b1_2, b1_3);

            SyncThreads();
            curBuf = nextBuf;
        }

        // Store: each m16n8 output region = 16 rows × 8 cols.
        int outRowM0 = blockRow + warpRow * 32 + 0;
        int outRowM1 = blockRow + warpRow * 32 + 16;
        int outCol0  = blockCol + warpCol * 32 + 0;
        int outCol1  = blockCol + warpCol * 32 + 8;
        int outCol2  = blockCol + warpCol * 32 + 16;
        int outCol3  = blockCol + warpCol * 32 + 24;
        WmmaArr.StoreCm16n8(c, outRowM0, outCol0, n, c00_0, c00_1, c00_2, c00_3);
        WmmaArr.StoreCm16n8(c, outRowM0, outCol1, n, c01_0, c01_1, c01_2, c01_3);
        WmmaArr.StoreCm16n8(c, outRowM0, outCol2, n, c02_0, c02_1, c02_2, c02_3);
        WmmaArr.StoreCm16n8(c, outRowM0, outCol3, n, c03_0, c03_1, c03_2, c03_3);
        WmmaArr.StoreCm16n8(c, outRowM1, outCol0, n, c10_0, c10_1, c10_2, c10_3);
        WmmaArr.StoreCm16n8(c, outRowM1, outCol1, n, c11_0, c11_1, c11_2, c11_3);
        WmmaArr.StoreCm16n8(c, outRowM1, outCol2, n, c12_0, c12_1, c12_2, c12_3);
        WmmaArr.StoreCm16n8(c, outRowM1, outCol3, n, c13_0, c13_1, c13_2, c13_3);
    }

    // Full PTX-MMA + ldmatrix + 3-stage pipeline. 128x128 block, 8 warps
    // (2M × 4N), per-warp 64M × 32N = 4 m16 × 4 n8 = 16 mma.m16n8k16 per K-iter.
    // Holds A as 16 uint, B as 8 uint, C as 64 fp32 in explicit C# locals.
    //
    // Block-swizzle rasterization (CUTLASS-style "supergroup"): the default
    // CUDA launch order has blockIdx.x increment fastest, which means
    // consecutive blocks share an A-row stripe. ncu showed L2 at 95% hit but
    // only 52% bandwidth utilized — the working set of B columns across a
    // single wave exceeds L2 and gets evicted before reuse. Swizzling into
    // groups of GROUP_M consecutive M-blocks (iterated column-first inside a
    // group) makes each wave cover a roughly square region of the output,
    // shrinking the L2 working set from ~thin-strip-of-B to sqrt(blocks)
    // worth of both A and B.
    [EntryPoint]
    public static void MultiplyKernelWmmaBigPtx(IntPtr a, IntPtr b, IntPtr c, int n)
    {
        const int BLOCK_M = 128;
        const int BLOCK_N = 128;
        const int BLOCK_K = 16;
        const int SKEW    = 8;
        const int LDA     = BLOCK_K + SKEW;     // 24
        const int LDB     = BLOCK_N + SKEW;     // 136
        const int STAGES  = 4;                  // bumped from 3 — more L2-latency hiding
        const int SA_BUF  = BLOCK_M * LDA;      // 3072
        const int SB_BUF  = BLOCK_K * LDB;      // 2176
        const int WARPS_N = 4;
        const int THREADS = 256;
        // GROUP_M=8 picked empirically: matches the ~72-block launch wave
        // (36 SMs × 2 blocks/SM register-limited) into a roughly square 8×9
        // region. GROUP_M=4 (90% cublas at n=4096) and GROUP_M=16 (also 93%)
        // both regressed vs 8 (93% at n=4096, 87% at n=2048).
        const int GROUP_M = 8;

        SharedMemoryAllocator<half> allocator = new SharedMemoryAllocator<half>();
        half[] sA = allocator.allocate(STAGES * SA_BUF);
        half[] sB = allocator.allocate(STAGES * SB_BUF);

        int tid = threadIdx.x;
        int warpId = tid / 32;
        int warpRow = warpId / WARPS_N;         // 0..1
        int warpCol = warpId % WARPS_N;         // 0..3

        // cp.async slot assignment — same as wmma-big-3stage.
        int aRow = tid / 2;
        int aCol = (tid % 2) * 8;
        int bRow = tid / 16;
        int bCol = (tid % 16) * 8;
        int aPadOff = aRow * LDA + aCol;
        int bPadOff = bRow * LDB + bCol;

        // Supergroup swizzle: remap (blockIdx.x, blockIdx.y) → (blockRowIdx,
        // blockColIdx) so a launch wave of ~72 blocks covers an 8 × 9 region
        // instead of a 1 × 72 strip.
        int gridM = gridDim.x;
        int gridN = gridDim.y;
        int blockLinear = blockIdx.x + blockIdx.y * gridM;
        int blocksPerGroup = GROUP_M * gridN;
        int groupId = blockLinear / blocksPerGroup;
        int firstM = groupId * GROUP_M;
        int sizeM = (gridM - firstM) > GROUP_M ? GROUP_M : (gridM - firstM);
        int idxInGroup = blockLinear - groupId * blocksPerGroup;
        int blockRowIdx = firstM + (idxInGroup % sizeM);
        int blockColIdx = idxInGroup / sizeM;

        int blockRow = blockRowIdx * BLOCK_M;
        int blockCol = blockColIdx * BLOCK_N;

        // 4 A tiles (m16k16) × 4 uint = 16 uint for A
        uint a0_0 = 0, a0_1 = 0, a0_2 = 0, a0_3 = 0;
        uint a1_0 = 0, a1_1 = 0, a1_2 = 0, a1_3 = 0;
        uint a2_0 = 0, a2_1 = 0, a2_2 = 0, a2_3 = 0;
        uint a3_0 = 0, a3_1 = 0, a3_2 = 0, a3_3 = 0;
        // 2 B tiles (k16n16) × 4 uint = 8 uint for B
        uint b0_0 = 0, b0_1 = 0, b0_2 = 0, b0_3 = 0;
        uint b1_0 = 0, b1_1 = 0, b1_2 = 0, b1_3 = 0;
        // 16 m16n8 accumulators × 4 fp32 = 64 fp32 locals.
        // Naming: c{m16}{n8}_{reg}, m16 ∈ {0..3}, n8 ∈ {0..3}.
        float c00_0 = 0, c00_1 = 0, c00_2 = 0, c00_3 = 0;
        float c01_0 = 0, c01_1 = 0, c01_2 = 0, c01_3 = 0;
        float c02_0 = 0, c02_1 = 0, c02_2 = 0, c02_3 = 0;
        float c03_0 = 0, c03_1 = 0, c03_2 = 0, c03_3 = 0;
        float c10_0 = 0, c10_1 = 0, c10_2 = 0, c10_3 = 0;
        float c11_0 = 0, c11_1 = 0, c11_2 = 0, c11_3 = 0;
        float c12_0 = 0, c12_1 = 0, c12_2 = 0, c12_3 = 0;
        float c13_0 = 0, c13_1 = 0, c13_2 = 0, c13_3 = 0;
        float c20_0 = 0, c20_1 = 0, c20_2 = 0, c20_3 = 0;
        float c21_0 = 0, c21_1 = 0, c21_2 = 0, c21_3 = 0;
        float c22_0 = 0, c22_1 = 0, c22_2 = 0, c22_3 = 0;
        float c23_0 = 0, c23_1 = 0, c23_2 = 0, c23_3 = 0;
        float c30_0 = 0, c30_1 = 0, c30_2 = 0, c30_3 = 0;
        float c31_0 = 0, c31_1 = 0, c31_2 = 0, c31_3 = 0;
        float c32_0 = 0, c32_1 = 0, c32_2 = 0, c32_3 = 0;
        float c33_0 = 0, c33_1 = 0, c33_2 = 0, c33_3 = 0;

        // Prologue: 2 stages.
        for (int s = 0; s < STAGES - 1; s++)
        {
            int kPre = s * BLOCK_K;
            if (kPre < n)
            {
                WmmaArr.CpAsync16(sA, s * SA_BUF + aPadOff, a, (blockRow + aRow) * n + (kPre + aCol));
                WmmaArr.CpAsync16(sB, s * SB_BUF + bPadOff, b, (kPre + bRow) * n + (blockCol + bCol));
                Pipeline.op.commit();
            }
        }

        int curBuf = 0;

        for (int kTile = 0; kTile < n; kTile += BLOCK_K)
        {
            int prefetchK = kTile + (STAGES - 1) * BLOCK_K;
            int prefetchBuf = (curBuf + STAGES - 1) % STAGES;

            if (prefetchK < n)
            {
                WmmaArr.CpAsync16(sA, prefetchBuf * SA_BUF + aPadOff, a, (blockRow + aRow) * n + (prefetchK + aCol));
                WmmaArr.CpAsync16(sB, prefetchBuf * SB_BUF + bPadOff, b, (prefetchK + bRow) * n + (blockCol + bCol));
                Pipeline.op.commit();
                Pipeline.op.wait_prior((ulong)(STAGES - 1));
            }
            else
            {
                Pipeline.op.wait_prior(0UL);
            }

            SyncThreads();

            int aStageRow = curBuf * BLOCK_M;
            int bStageRow = curBuf * BLOCK_K;
            int warpRowBase = warpRow * 64;
            int warpColBase = warpCol * 32;

            // 4 A tiles (m16 slices) — same column 0 of sA.
            WmmaArr.LoadAm16k16(out a0_0, out a0_1, out a0_2, out a0_3, sA, aStageRow + warpRowBase + 0 , 0, LDA);
            WmmaArr.LoadAm16k16(out a1_0, out a1_1, out a1_2, out a1_3, sA, aStageRow + warpRowBase + 16, 0, LDA);
            WmmaArr.LoadAm16k16(out a2_0, out a2_1, out a2_2, out a2_3, sA, aStageRow + warpRowBase + 32, 0, LDA);
            WmmaArr.LoadAm16k16(out a3_0, out a3_1, out a3_2, out a3_3, sA, aStageRow + warpRowBase + 48, 0, LDA);
            // 2 B tiles (n16 slices) — same row 0 of sB.
            WmmaArr.LoadBk16n16(out b0_0, out b0_1, out b0_2, out b0_3, sB, bStageRow, warpColBase + 0 , LDB);
            WmmaArr.LoadBk16n16(out b1_0, out b1_1, out b1_2, out b1_3, sB, bStageRow, warpColBase + 16, LDB);

            // 16 mma.m16n8k16. Pairing for B: first n8 in a tile = (r0, r1),
            // second n8 = (r2, r3) — column-major submatrix arrangement.
            // m16=0
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c00_0, ref c00_1, ref c00_2, ref c00_3, a0_0, a0_1, a0_2, a0_3, b0_0, b0_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c01_0, ref c01_1, ref c01_2, ref c01_3, a0_0, a0_1, a0_2, a0_3, b0_2, b0_3);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c02_0, ref c02_1, ref c02_2, ref c02_3, a0_0, a0_1, a0_2, a0_3, b1_0, b1_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c03_0, ref c03_1, ref c03_2, ref c03_3, a0_0, a0_1, a0_2, a0_3, b1_2, b1_3);
            // m16=1
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c10_0, ref c10_1, ref c10_2, ref c10_3, a1_0, a1_1, a1_2, a1_3, b0_0, b0_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c11_0, ref c11_1, ref c11_2, ref c11_3, a1_0, a1_1, a1_2, a1_3, b0_2, b0_3);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c12_0, ref c12_1, ref c12_2, ref c12_3, a1_0, a1_1, a1_2, a1_3, b1_0, b1_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c13_0, ref c13_1, ref c13_2, ref c13_3, a1_0, a1_1, a1_2, a1_3, b1_2, b1_3);
            // m16=2
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c20_0, ref c20_1, ref c20_2, ref c20_3, a2_0, a2_1, a2_2, a2_3, b0_0, b0_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c21_0, ref c21_1, ref c21_2, ref c21_3, a2_0, a2_1, a2_2, a2_3, b0_2, b0_3);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c22_0, ref c22_1, ref c22_2, ref c22_3, a2_0, a2_1, a2_2, a2_3, b1_0, b1_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c23_0, ref c23_1, ref c23_2, ref c23_3, a2_0, a2_1, a2_2, a2_3, b1_2, b1_3);
            // m16=3
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c30_0, ref c30_1, ref c30_2, ref c30_3, a3_0, a3_1, a3_2, a3_3, b0_0, b0_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c31_0, ref c31_1, ref c31_2, ref c31_3, a3_0, a3_1, a3_2, a3_3, b0_2, b0_3);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c32_0, ref c32_1, ref c32_2, ref c32_3, a3_0, a3_1, a3_2, a3_3, b1_0, b1_1);
            Mma.op.m16n8k16_f32_f16_f16_f32(ref c33_0, ref c33_1, ref c33_2, ref c33_3, a3_0, a3_1, a3_2, a3_3, b1_2, b1_3);

            SyncThreads();
            curBuf = (curBuf + 1) % STAGES;
        }

        // 16 m16n8 stores — each is 16 rows × 8 cols of fp32.
        int rM0 = blockRow + warpRow * 64 + 0;
        int rM1 = blockRow + warpRow * 64 + 16;
        int rM2 = blockRow + warpRow * 64 + 32;
        int rM3 = blockRow + warpRow * 64 + 48;
        int cN0 = blockCol + warpCol * 32 + 0;
        int cN1 = blockCol + warpCol * 32 + 8;
        int cN2 = blockCol + warpCol * 32 + 16;
        int cN3 = blockCol + warpCol * 32 + 24;
        WmmaArr.StoreCm16n8(c, rM0, cN0, n, c00_0, c00_1, c00_2, c00_3);
        WmmaArr.StoreCm16n8(c, rM0, cN1, n, c01_0, c01_1, c01_2, c01_3);
        WmmaArr.StoreCm16n8(c, rM0, cN2, n, c02_0, c02_1, c02_2, c02_3);
        WmmaArr.StoreCm16n8(c, rM0, cN3, n, c03_0, c03_1, c03_2, c03_3);
        WmmaArr.StoreCm16n8(c, rM1, cN0, n, c10_0, c10_1, c10_2, c10_3);
        WmmaArr.StoreCm16n8(c, rM1, cN1, n, c11_0, c11_1, c11_2, c11_3);
        WmmaArr.StoreCm16n8(c, rM1, cN2, n, c12_0, c12_1, c12_2, c12_3);
        WmmaArr.StoreCm16n8(c, rM1, cN3, n, c13_0, c13_1, c13_2, c13_3);
        WmmaArr.StoreCm16n8(c, rM2, cN0, n, c20_0, c20_1, c20_2, c20_3);
        WmmaArr.StoreCm16n8(c, rM2, cN1, n, c21_0, c21_1, c21_2, c21_3);
        WmmaArr.StoreCm16n8(c, rM2, cN2, n, c22_0, c22_1, c22_2, c22_3);
        WmmaArr.StoreCm16n8(c, rM2, cN3, n, c23_0, c23_1, c23_2, c23_3);
        WmmaArr.StoreCm16n8(c, rM3, cN0, n, c30_0, c30_1, c30_2, c30_3);
        WmmaArr.StoreCm16n8(c, rM3, cN1, n, c31_0, c31_1, c31_2, c31_3);
        WmmaArr.StoreCm16n8(c, rM3, cN2, n, c32_0, c32_1, c32_2, c32_3);
        WmmaArr.StoreCm16n8(c, rM3, cN3, n, c33_0, c33_1, c33_2, c33_3);
    }

    // Big-tile 3-stage WMMA + ldmatrix. Bigger block (128x128), more warps
    // (8 in a 2M x 4N grid), deeper pipeline (3 stages — issue K+2 while
    // computing K), and the shmem->fragment load uses our ldmatrix bindings
    // directly instead of nvcuda::wmma::load_matrix_sync. The aim is to
    // close on cuBLAS at large n: it gives more mma_syncs per global-load
    // byte (better A/B reuse) and overlaps two K-tiles of loads under the
    // tensor-core work.
    //
    // Per-warp area: 64 (M) x 32 (N) = 4 m16 frags x 2 n16 frags = 8 acc
    // fragments. Per K-iter per warp: 4 LdmatrixA + 2 LdmatrixB + 8 mma_sync.
    //
    // n must be a multiple of BLOCK_M=128. Existing Main check guarantees
    // this (n % 128 == 0).
    [EntryPoint]
    public static void MultiplyKernelWmmaBig3Stage(IntPtr a, IntPtr b, IntPtr c, int n)
    {
        const int BLOCK_M = 128;
        const int BLOCK_N = 128;
        const int BLOCK_K = 16;
        const int SKEW    = 8;
        const int LDA     = BLOCK_K + SKEW;     // 24 halves per padded sA row
        const int LDB     = BLOCK_N + SKEW;     // 136 halves per padded sB row
        const int STAGES  = 3;
        const int SA_BUF  = BLOCK_M * LDA;      // 3072 halves per A stage
        const int SB_BUF  = BLOCK_K * LDB;      // 2176 halves per B stage
        const int WARPS_N = 4;                  // 2x4 warp grid over the 128x128 block
        const int THREADS = 256;                // 8 warps * 32

        SharedMemoryAllocator<half> allocator = new SharedMemoryAllocator<half>();
        half[] sA = allocator.allocate(STAGES * SA_BUF);
        half[] sB = allocator.allocate(STAGES * SB_BUF);

        int tid = threadIdx.x;
        int warpId = tid / 32;
        int warpRow = warpId / WARPS_N;         // 0..1
        int warpCol = warpId % WARPS_N;         // 0..3

        // cp.async slot assignment — each thread copies 8 halves per buffer
        // per K-iter. THREADS=256, BLOCK_M*BLOCK_K = BLOCK_K*BLOCK_N = 2048
        // halves per stage; 2048/256 = 8 halves per thread.
        //   sA: 128 rows x 16 cols → 2 slots/row, 128 rows = 256 slots.
        //   sB:  16 rows x 128 cols → 16 slots/row, 16 rows = 256 slots.
        int aRow = tid / 2;
        int aCol = (tid % 2) * 8;
        int bRow = tid / 16;
        int bCol = (tid % 16) * 8;
        int aPadOff = aRow * LDA + aCol;
        int bPadOff = bRow * LDB + bCol;

        int blockRow = blockIdx.y * BLOCK_M;
        int blockCol = blockIdx.x * BLOCK_N;

        // Eight fragments of A (one per M slice per K-iter), but we only need
        // 4 live at a time within one K-iter (warp covers 64M = 4 m16). Same
        // for B (2 live n16 frags).
        var a0 = new Wmma.frag_a_16x16x16_half_row();
        var a1 = new Wmma.frag_a_16x16x16_half_row();
        var a2 = new Wmma.frag_a_16x16x16_half_row();
        var a3 = new Wmma.frag_a_16x16x16_half_row();
        var b0 = new Wmma.frag_b_16x16x16_half_row();
        var b1 = new Wmma.frag_b_16x16x16_half_row();

        // 8 accumulator fragments (4 m16 x 2 n16).
        var c00 = new Wmma.frag_acc_16x16x16_float();
        var c01 = new Wmma.frag_acc_16x16x16_float();
        var c10 = new Wmma.frag_acc_16x16x16_float();
        var c11 = new Wmma.frag_acc_16x16x16_float();
        var c20 = new Wmma.frag_acc_16x16x16_float();
        var c21 = new Wmma.frag_acc_16x16x16_float();
        var c30 = new Wmma.frag_acc_16x16x16_float();
        var c31 = new Wmma.frag_acc_16x16x16_float();
        Wmma.op.fill_fragment(ref c00, 0.0f);
        Wmma.op.fill_fragment(ref c01, 0.0f);
        Wmma.op.fill_fragment(ref c10, 0.0f);
        Wmma.op.fill_fragment(ref c11, 0.0f);
        Wmma.op.fill_fragment(ref c20, 0.0f);
        Wmma.op.fill_fragment(ref c21, 0.0f);
        Wmma.op.fill_fragment(ref c30, 0.0f);
        Wmma.op.fill_fragment(ref c31, 0.0f);

        // Prologue: stage K=0..(STAGES-2)*BLOCK_K. Each iteration's loads form
        // one commit group, so by the time the main loop starts we have
        // STAGES-1 = 2 groups in flight.
        for (int s = 0; s < STAGES - 1; s++)
        {
            int kPre = s * BLOCK_K;
            if (kPre < n)
            {
                WmmaArr.CpAsync16(sA, s * SA_BUF + aPadOff, a, (blockRow + aRow) * n + (kPre + aCol));
                WmmaArr.CpAsync16(sB, s * SB_BUF + bPadOff, b, (kPre + bRow) * n + (blockCol + bCol));
                Pipeline.op.commit();
            }
        }

        int curBuf = 0;

        for (int kTile = 0; kTile < n; kTile += BLOCK_K)
        {
            int prefetchK = kTile + (STAGES - 1) * BLOCK_K;
            int prefetchBuf = (curBuf + STAGES - 1) % STAGES;

            if (prefetchK < n)
            {
                WmmaArr.CpAsync16(sA, prefetchBuf * SA_BUF + aPadOff, a, (blockRow + aRow) * n + (prefetchK + aCol));
                WmmaArr.CpAsync16(sB, prefetchBuf * SB_BUF + bPadOff, b, (prefetchK + bRow) * n + (blockCol + bCol));
                Pipeline.op.commit();
                // STAGES commits in flight now; drain just the oldest (the
                // one we're about to consume), keeping STAGES-1 outstanding.
                // Earlier this was hardcoded to wait_prior(1UL), which only
                // gives true 2-stage pipelining no matter how many shmem
                // buffers we allocated — the over-drain was a real bug.
                Pipeline.op.wait_prior((ulong)(STAGES - 1));
            }
            else
            {
                // No more prefetches — drain everything still in flight.
                Pipeline.op.wait_prior(0UL);
            }

            SyncThreads();

            // Stage offsets fold cleanly into row offsets because
            // SA_BUF / LDA == BLOCK_M and SB_BUF / LDB == BLOCK_K.
            int aStageRow = curBuf * BLOCK_M;       // 0 / 128 / 256 within sA
            int bStageRow = curBuf * BLOCK_K;       // 0 /  16 /  32 within sB
            int warpRowBase = warpRow * 64;         // 0 or 64 within the 128M block
            int warpColBase = warpCol * 32;         // 0/32/64/96 within the 128N block

            // This kernel keeps wmma::load_matrix_sync (which itself uses ldmatrix
            // internally on sm_80+). The PTX-level variant lives in
            // MultiplyKernelWmmaBigPtx below — it ditches wmma::fragment entirely
            // because the per-lane register layout produced by ldmatrix doesn't
            // match wmma::fragment::x[] portably (officially unspecified).
            int saBase = curBuf * SA_BUF;
            int sbBase = curBuf * SB_BUF;
            WmmaArr.LoadAShmem(ref a0, sA, saBase + (warpRowBase + 0 ) * LDA, (uint)LDA);
            WmmaArr.LoadAShmem(ref a1, sA, saBase + (warpRowBase + 16) * LDA, (uint)LDA);
            WmmaArr.LoadAShmem(ref a2, sA, saBase + (warpRowBase + 32) * LDA, (uint)LDA);
            WmmaArr.LoadAShmem(ref a3, sA, saBase + (warpRowBase + 48) * LDA, (uint)LDA);
            WmmaArr.LoadBShmem(ref b0, sB, sbBase + warpColBase + 0 , (uint)LDB);
            WmmaArr.LoadBShmem(ref b1, sB, sbBase + warpColBase + 16, (uint)LDB);

            // 4x2 outer product.
            Wmma.op.mma_sync(ref c00, a0, b0, c00);
            Wmma.op.mma_sync(ref c01, a0, b1, c01);
            Wmma.op.mma_sync(ref c10, a1, b0, c10);
            Wmma.op.mma_sync(ref c11, a1, b1, c11);
            Wmma.op.mma_sync(ref c20, a2, b0, c20);
            Wmma.op.mma_sync(ref c21, a2, b1, c21);
            Wmma.op.mma_sync(ref c30, a3, b0, c30);
            Wmma.op.mma_sync(ref c31, a3, b1, c31);

            SyncThreads();

            curBuf = (curBuf + 1) % STAGES;
        }

        int rowBase = blockRow + warpRow * 64;
        int colBase = blockCol + warpCol * 32;
        WmmaArr.StoreC(c, (rowBase + 0 ) * n + (colBase + 0 ), c00, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 0 ) * n + (colBase + 16), c01, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 16) * n + (colBase + 0 ), c10, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 16) * n + (colBase + 16), c11, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 32) * n + (colBase + 0 ), c20, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 32) * n + (colBase + 16), c21, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 48) * n + (colBase + 0 ), c30, (uint)n, Wmma.wmma_layout.mem_row_major);
        WmmaArr.StoreC(c, (rowBase + 48) * n + (colBase + 16), c31, (uint)n, Wmma.wmma_layout.mem_row_major);
    }

    // Allocate raw device buffers for the WMMA inputs/output. Returning IntPtrs
    // means we can pass these directly to the WMMA kernel without involving
    // Hybridizer's per-call array marshaller.
    private static (IntPtr aHalfDev, IntPtr bHalfDev, IntPtr cWmmaDev) AllocWmmaDeviceBuffers(
        int n, float[] aFloat, float[] bFloat)
    {
        int elems = n * n;
        IntPtr aHalfDev, bHalfDev, cWmmaDev;
        cuda.ERROR_CHECK(cuda.Malloc(out aHalfDev, (long)elems * sizeof(ushort)));
        cuda.ERROR_CHECK(cuda.Malloc(out bHalfDev, (long)elems * sizeof(ushort)));
        cuda.ERROR_CHECK(cuda.Malloc(out cWmmaDev, (long)elems * sizeof(float)));

        // Convert float -> binary16 bits on the host once, then push to device.
        var halfBits = new ushort[elems];
        for (int i = 0; i < elems; i++) halfBits[i] = BitConverter.HalfToUInt16Bits((Half)aFloat[i]);
        var pinned = GCHandle.Alloc(halfBits, GCHandleType.Pinned);
        try
        {
            cuda.ERROR_CHECK(cuda.Memcpy(aHalfDev, pinned.AddrOfPinnedObject(),
                (long)elems * sizeof(ushort), cudaMemcpyKind.cudaMemcpyHostToDevice));
        }
        finally { pinned.Free(); }
        for (int i = 0; i < elems; i++) halfBits[i] = BitConverter.HalfToUInt16Bits((Half)bFloat[i]);
        pinned = GCHandle.Alloc(halfBits, GCHandleType.Pinned);
        try
        {
            cuda.ERROR_CHECK(cuda.Memcpy(bHalfDev, pinned.AddrOfPinnedObject(),
                (long)elems * sizeof(ushort), cudaMemcpyKind.cudaMemcpyHostToDevice));
        }
        finally { pinned.Free(); }
        return (aHalfDev, bHalfDev, cWmmaDev);
    }

    private static void CopyDeviceFloatsToHost(IntPtr devPtr, float[] dst)
    {
        var pinned = GCHandle.Alloc(dst, GCHandleType.Pinned);
        try
        {
            cuda.ERROR_CHECK(cuda.Memcpy(pinned.AddrOfPinnedObject(), devPtr,
                (long)dst.Length * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost));
        }
        finally { pinned.Free(); }
    }

    private static FloatResidentArray MakeOutputResident(int count)
    {
        var arr = new FloatResidentArray(count);
        _ = arr.DevicePointer;
        arr.Status = ResidentArrayStatus.HostNeedsRefresh;
        return arr;
    }

    private static float[] HostCopy(FloatResidentArray arr, int count)
    {
        arr.RefreshHost();
        var host = new float[count];
        Marshal.Copy(arr.HostPointer, host, 0, count);
        return host;
    }

    private static (double bestMs, double avgMs, double worstMs) BenchmarkKernel(
        string name, int iters, int warmup, Action launch)
    {
        launch();
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());

        for (int w = 0; w < warmup; w++)
        {
            launch();
        }
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());

        double min = double.MaxValue, max = 0.0, sum = 0.0;
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            launch();
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
            sw.Stop();
            var ms = sw.Elapsed.TotalMilliseconds;
            if (ms < min) min = ms;
            if (ms > max) max = ms;
            sum += ms;
        }
        return (min, sum / iters, max);
    }

    private static void PrintGpu(string name, (double bestMs, double avgMs, double worstMs) t, int n, int iters, int warmup)
    {
        if (iters > 1)
        {
            Console.WriteLine($"GPU {name,-5} {iters} iters (warmup {warmup})  min/avg/max: {t.bestMs,7:F2} /{t.avgMs,7:F2} /{t.worstMs,7:F2} ms   best: {Gflops(n, t.bestMs / 1000.0):F3} GFLOPS");
        }
        else
        {
            Console.WriteLine($"GPU {name,-5} elapsed: {t.bestMs:F2} ms ({Gflops(n, t.bestMs / 1000.0):F3} GFLOPS)");
        }
    }

    private static void MultiplyCpu(float[] a, float[] b, float[] c, int n)
    {
        Parallel.For(0, n, i => 
        {
            for (int k = 0; k < n; k++)
            {
                var aik = a[i * n + k];
                int rowB = k * n;
                int rowC = i * n;
                for (int j = 0; j < n; j++)
                {
                    c[rowC + j] += aik * b[rowB + j];
                }
            }
        });
    }

    private static float[] CreateRandomMatrix(int rows, int cols, Random rng)
    {
        var m = new float[rows * cols];
        for (int i = 0; i < m.Length; i++)
        {
            m[i] = (float)rng.NextDouble();
        }
        return m;
    }

    private static (float maxAbsErr, double sumGpu) Compare(float[] cpu, float[] gpu)
    {
        float maxAbsErr = 0.0f;
        double sumGpu = 0.0;
        for (int i = 0; i < cpu.Length; i++)
        {
            float diff = MathF.Abs(cpu[i] - gpu[i]);
            if (diff > maxAbsErr) maxAbsErr = diff;
            sumGpu += gpu[i];
        }
        return (maxAbsErr, sumGpu);
    }

    private static double Checksum(float[] m)
    {
        double s = 0.0;
        for (int i = 0; i < m.Length; i++) s += m[i];
        return s;
    }

    private static double Gflops(int n, double seconds)
    {
        return 2.0 * n * n * n / seconds / 1e9;
    }
}
