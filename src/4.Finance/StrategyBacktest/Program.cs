using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Hybridizer.Basic.Finance
{
    /// <summary>
    /// GPU-accelerated trading strategy backtester using Hybridizer.
    /// Performs an exhaustive parameter sweep of Moving Average crossover strategies
    /// across thousands of (shortMA, longMA) combinations simultaneously on the GPU.
    /// 
    /// Usage:
    ///   StrategyBacktest --file history.csv
    ///   StrategyBacktest --symbol BTC-USD
    ///   StrategyBacktest --symbol AAPL --days 3650
    /// </summary>
    class Program
    {
        // --- Parameter sweep bounds ---
        const int SHORT_MA_MIN = 2;
        const int SHORT_MA_MAX = 200;
        const int LONG_MA_MIN = 5;
        const int LONG_MA_MAX = 400;

        static readonly int ShortRange = SHORT_MA_MAX - SHORT_MA_MIN + 1;   // 199
        static readonly int LongRange = LONG_MA_MAX - LONG_MA_MIN + 1;     // 396
        static readonly int TotalCombinations = ShortRange * LongRange;     // 78,804

        const float INITIAL_EQUITY = 10000.0f;

        static void Main(string[] args)
        {
            // ── 1. Load price data ──────────────────────────────────────
            float[] prices = PriceDataLoader.LoadPrices(args, LONG_MA_MAX + 10);
            int N = prices.Length;

            Console.WriteLine($"╔══════════════════════════════════════════════════════════════╗");
            Console.WriteLine($"║   Strategy Backtest — Hybridizer GPU Parameter Sweep        ║");
            Console.WriteLine($"╠══════════════════════════════════════════════════════════════╣");
            Console.WriteLine($"║  Price data points : {N,10}                                 ║");
            Console.WriteLine($"║  Short MA range    : [{SHORT_MA_MIN}..{SHORT_MA_MAX}]                               ║");
            Console.WriteLine($"║  Long  MA range    : [{LONG_MA_MIN}..{LONG_MA_MAX}]                              ║");
            Console.WriteLine($"║  Total combinations: {TotalCombinations,10}                                 ║");
            Console.WriteLine($"╚══════════════════════════════════════════════════════════════╝");
            Console.WriteLine();

            float[] returns_cpu = new float[TotalCombinations];
            float[] returns_par = new float[TotalCombinations];
            float[] returns_gpu = new float[TotalCombinations];

            // ── 2. CPU Sequential ───────────────────────────────────────
            Console.Write("Running CPU sequential benchmark...    ");
            var sw = Stopwatch.StartNew();
            BacktestAllCpuSequential(prices, N, returns_cpu);
            sw.Stop();
            long cpuSeqMs = sw.ElapsedMilliseconds;
            Console.WriteLine($"{cpuSeqMs,8} ms");

            // ── 3. CPU Parallel ─────────────────────────────────────────
            Console.Write($"Running CPU parallel ({Environment.ProcessorCount} cores)...      ");
            sw.Restart();
            BacktestAllCpuParallel(prices, N, returns_par);
            sw.Stop();
            long cpuParMs = sw.ElapsedMilliseconds;
            Console.WriteLine($"{cpuParMs,8} ms");

            // ── 4. GPU Hybridizer ───────────────────────────────────────
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            Console.WriteLine($"GPU detected: {new string(prop.name).TrimEnd('\0')}");

            HybRunner runner = SatelliteLoader.Load()
                .SetDistrib(16 * prop.multiProcessorCount, 256);
            dynamic wrapper = runner.Wrap(new Program());

            // Warmup (includes JIT + memory transfers for both kernels)
            float[] returns_gpu_naive = new float[TotalCombinations];
            wrapper.BacktestKernel(prices, N, returns_gpu,
                SHORT_MA_MIN, ShortRange, LONG_MA_MIN, LongRange);
            wrapper.BacktestKernelNaive(prices, N, returns_gpu_naive,
                SHORT_MA_MIN, ShortRange, LONG_MA_MIN, LongRange);
            cuda.DeviceSynchronize();

            // ── 4a. GPU Naive (Parallel.For style → GPU) ────────────────
            Console.Write("Running GPU naive (from Parallel)...    ");
            sw.Restart();
            wrapper.BacktestKernelNaive(prices, N, returns_gpu_naive,
                SHORT_MA_MIN, ShortRange, LONG_MA_MIN, LongRange);
            cuda.DeviceSynchronize();
            sw.Stop();
            long gpuNaiveMs = sw.ElapsedMilliseconds;
            Console.WriteLine($"{gpuNaiveMs,8} ms");

            // ── 4b. GPU Native (hand-written kernel) ────────────────────
            Console.Write("Running GPU native kernel...            ");
            sw.Restart();
            wrapper.BacktestKernel(prices, N, returns_gpu,
                SHORT_MA_MIN, ShortRange, LONG_MA_MIN, LongRange);
            cuda.DeviceSynchronize();
            sw.Stop();
            long gpuMs = sw.ElapsedMilliseconds;
            Console.WriteLine($"{gpuMs,8} ms");

            // ── 5. Results ──────────────────────────────────────────────
            Console.WriteLine();
            Console.WriteLine("┌──────────────────────────────────────────────────────────┐");
            Console.WriteLine("│                PERFORMANCE COMPARISON                    │");
            Console.WriteLine("├──────────────────────┬──────────┬────────────────────────┤");
            Console.WriteLine("│ Method               │ Time(ms) │ Speedup vs seq. CPU    │");
            Console.WriteLine("├──────────────────────┼──────────┼────────────────────────┤");
            Console.WriteLine($"│ CPU sequential       │ {cpuSeqMs,8} │ {"(baseline)",22} │");
            Console.WriteLine($"│ CPU parallel ({Environment.ProcessorCount,2}c)   │ {cpuParMs,8} │ {(cpuParMs > 0 ? $"x{(double)cpuSeqMs / cpuParMs:F1}" : "N/A"),22} │");
            Console.WriteLine($"│ GPU naive (from CPU) │ {gpuNaiveMs,8} │ {(gpuNaiveMs > 0 ? $"x{(double)cpuSeqMs / gpuNaiveMs:F1}" : "N/A"),22} │");
            Console.WriteLine($"│ GPU native kernel    │ {gpuMs,8} │ {(gpuMs > 0 ? $"x{(double)cpuSeqMs / gpuMs:F1}" : "N/A"),22} │");
            Console.WriteLine("└──────────────────────┴──────────┴────────────────────────┘");

            // ── 6. Validate results ─────────────────────────────────────
            ValidateResults(returns_cpu, returns_gpu);

            // ── 7. Find best strategy ───────────────────────────────────
            FindBestStrategy(returns_gpu);

            // ── 8. Generate heatmap ─────────────────────────────────────
            string heatmapPath = "strategy_heatmap.png";
            HeatmapGenerator.GeneratePng(returns_gpu, heatmapPath,
                SHORT_MA_MIN, SHORT_MA_MAX, ShortRange,
                LONG_MA_MIN, LONG_MA_MAX, LongRange);
            Console.WriteLine($"\nHeatmap saved to: {heatmapPath}");

            // ── 9. Console heatmap ──────────────────────────────────────
            HeatmapGenerator.PrintConsoleHeatmap(returns_gpu,
                SHORT_MA_MIN, SHORT_MA_MAX, ShortRange,
                LONG_MA_MIN, LONG_MA_MAX, LongRange);

            Console.WriteLine("\nDONE");
        }

        // =====================================================================
        //  GPU KERNEL (NATIVE) — Fully inlined backtest logic
        //  Each thread evaluates one (shortMA, longMA) combination
        // =====================================================================
        [EntryPoint]
        public static void BacktestKernel(
            [In] float[] prices, int priceCount,
            [Out] float[] returns,
            int shortMaMin, int shortRange,
            int longMaMin, int longRange)
        {
            int totalCombinations = shortRange * longRange;
            for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
                 idx < totalCombinations;
                 idx += blockDim.x * gridDim.x)
            {
                int si = idx / longRange;
                int li = idx % longRange;
                int shortPeriod = shortMaMin + si;
                int longPeriod = longMaMin + li;

                if (shortPeriod >= longPeriod)
                {
                    returns[idx] = 0.0f;
                }
                else
                {
                    // ── Run single backtest for this (short, long) combo ──
                    float equity = 10000.0f;
                    float shares = 0.0f;
                    int inPosition = 0;
                    float prevShortMA = 0.0f;
                    float prevLongMA = 0.0f;
                    int hasPrev = 0;

                    for (int d = longPeriod - 1; d < priceCount; d++)
                    {
                        // Compute short MA (naive — intentionally O(shortPeriod) for compute intensity)
                        float shortSum = 0.0f;
                        for (int k = d - shortPeriod + 1; k <= d; k++)
                            shortSum += prices[k];
                        float shortMA = shortSum / shortPeriod;

                        // Compute long MA (naive — intentionally O(longPeriod) for compute intensity)
                        float longSum = 0.0f;
                        for (int k = d - longPeriod + 1; k <= d; k++)
                            longSum += prices[k];
                        float longMA = longSum / longPeriod;

                        if (hasPrev == 1)
                        {
                            // Golden cross → buy
                            if (prevShortMA <= prevLongMA && shortMA > longMA && inPosition == 0)
                            {
                                shares = equity / prices[d];
                                equity = 0.0f;
                                inPosition = 1;
                            }
                            // Death cross → sell
                            else if (prevShortMA >= prevLongMA && shortMA < longMA && inPosition == 1)
                            {
                                equity = shares * prices[d];
                                shares = 0.0f;
                                inPosition = 0;
                            }
                        }

                        prevShortMA = shortMA;
                        prevLongMA = longMA;
                        hasPrev = 1;
                    }

                    // Close position at end
                    if (inPosition == 1)
                    {
                        equity = shares * prices[priceCount - 1];
                    }

                    returns[idx] = (equity - 10000.0f) / 100.0f; // return in %
                }
            }
        }

        // =====================================================================
        //  GPU KERNEL (NAIVE) — Same structure as Parallel.For, calls [Kernel]
        //  Shows: minimal changes to port CPU code to GPU
        // =====================================================================
        [EntryPoint]
        public static void BacktestKernelNaive(
            [In] float[] prices, int priceCount,
            [Out] float[] returns,
            int shortMaMin, int shortRange,
            int longMaMin, int longRange)
        {
            int totalCombinations = shortRange * longRange;
            for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
                 idx < totalCombinations;
                 idx += blockDim.x * gridDim.x)
            {
                int si = idx / longRange;
                int li = idx % longRange;
                int shortPeriod = shortMaMin + si;
                int longPeriod = longMaMin + li;

                if (shortPeriod >= longPeriod)
                {
                    returns[idx] = 0.0f;
                }
                else
                {
                    // Calls the device function — same code as CPU version
                    returns[idx] = RunSingleBacktest(prices, priceCount, shortPeriod, longPeriod);
                }
            }
        }

        /// <summary>
        /// Device-callable backtest function — mirrors RunBacktestCpu exactly.
        /// Marked [Kernel] so it can be called from GPU entry points.
        /// </summary>
        [Kernel]
        public static float RunSingleBacktest(float[] prices, int priceCount, int shortPeriod, int longPeriod)
        {
            float equity = 10000.0f;
            float shares = 0.0f;
            int inPosition = 0;
            float prevShortMA = 0.0f;
            float prevLongMA = 0.0f;
            int hasPrev = 0;

            for (int d = longPeriod - 1; d < priceCount; d++)
            {
                float shortSum = 0.0f;
                for (int k = d - shortPeriod + 1; k <= d; k++)
                    shortSum += prices[k];
                float shortMA = shortSum / shortPeriod;

                float longSum = 0.0f;
                for (int k = d - longPeriod + 1; k <= d; k++)
                    longSum += prices[k];
                float longMA = longSum / longPeriod;

                if (hasPrev == 1)
                {
                    if (prevShortMA <= prevLongMA && shortMA > longMA && inPosition == 0)
                    {
                        shares = equity / prices[d];
                        equity = 0.0f;
                        inPosition = 1;
                    }
                    else if (prevShortMA >= prevLongMA && shortMA < longMA && inPosition == 1)
                    {
                        equity = shares * prices[d];
                        shares = 0.0f;
                        inPosition = 0;
                    }
                }

                prevShortMA = shortMA;
                prevLongMA = longMA;
                hasPrev = 1;
            }

            if (inPosition == 1)
                equity = shares * prices[priceCount - 1];

            return (equity - 10000.0f) / 100.0f;
        }

        // =====================================================================
        //  CPU Sequential — baseline for speedup measurement
        // =====================================================================
        static void BacktestAllCpuSequential(float[] prices, int priceCount, float[] returns)
        {
            for (int idx = 0; idx < TotalCombinations; idx++)
            {
                int si = idx / LongRange;
                int li = idx % LongRange;
                int shortPeriod = SHORT_MA_MIN + si;
                int longPeriod = LONG_MA_MIN + li;

                if (shortPeriod >= longPeriod)
                {
                    returns[idx] = 0.0f;
                    continue;
                }

                returns[idx] = RunBacktestCpu(prices, priceCount, shortPeriod, longPeriod);
            }
        }

        // =====================================================================
        //  CPU Parallel — shows GPU advantage even vs multi-core
        // =====================================================================
        static void BacktestAllCpuParallel(float[] prices, int priceCount, float[] returns)
        {
            Parallel.For(0, TotalCombinations, (idx) =>
            {
                int si = idx / LongRange;
                int li = idx % LongRange;
                int shortPeriod = SHORT_MA_MIN + si;
                int longPeriod = LONG_MA_MIN + li;

                if (shortPeriod >= longPeriod)
                {
                    returns[idx] = 0.0f;
                    return;
                }

                returns[idx] = RunBacktestCpu(prices, priceCount, shortPeriod, longPeriod);
            });
        }

        /// <summary>
        /// CPU version of a single backtest — same algorithm as the GPU kernel.
        /// Uses naive O(window) SMA computation for fair comparison.
        /// </summary>
        static float RunBacktestCpu(float[] prices, int priceCount, int shortPeriod, int longPeriod)
        {
            float equity = INITIAL_EQUITY;
            float shares = 0.0f;
            bool inPosition = false;
            float prevShortMA = 0.0f;
            float prevLongMA = 0.0f;
            bool hasPrev = false;

            for (int d = longPeriod - 1; d < priceCount; d++)
            {
                float shortSum = 0.0f;
                for (int k = d - shortPeriod + 1; k <= d; k++)
                    shortSum += prices[k];
                float shortMA = shortSum / shortPeriod;

                float longSum = 0.0f;
                for (int k = d - longPeriod + 1; k <= d; k++)
                    longSum += prices[k];
                float longMA = longSum / longPeriod;

                if (hasPrev)
                {
                    if (prevShortMA <= prevLongMA && shortMA > longMA && !inPosition)
                    {
                        shares = equity / prices[d];
                        equity = 0.0f;
                        inPosition = true;
                    }
                    else if (prevShortMA >= prevLongMA && shortMA < longMA && inPosition)
                    {
                        equity = shares * prices[d];
                        shares = 0.0f;
                        inPosition = false;
                    }
                }

                prevShortMA = shortMA;
                prevLongMA = longMA;
                hasPrev = true;
            }

            if (inPosition)
                equity = shares * prices[priceCount - 1];

            return (equity - INITIAL_EQUITY) / 100.0f;
        }

        // =====================================================================
        //  Validation & Analysis
        // =====================================================================
        static void ValidateResults(float[] cpu, float[] gpu)
        {
            float maxErr = 0, sumErr = 0;
            int count = 0;
            for (int i = 0; i < cpu.Length; i++)
            {
                float err = Math.Abs(cpu[i] - gpu[i]);
                maxErr = Math.Max(maxErr, err);
                sumErr += err;
                if (cpu[i] != 0) count++;
            }

            Console.WriteLine();
            Console.WriteLine($"Validation (CPU vs GPU): max error = {maxErr:G6}, avg error = {(count > 0 ? sumErr / count : 0):G6}");
            if (maxErr < 1.0f)
                Console.WriteLine("✓ Results match within acceptable tolerance.");
            else
                Console.WriteLine("⚠ Results differ — check floating-point precision (float32 GPU vs CPU).");
        }

        static void FindBestStrategy(float[] returns)
        {
            float bestReturn = float.MinValue;
            int bestShort = 0, bestLong = 0;
            float worstReturn = float.MaxValue;
            int worstShort = 0, worstLong = 0;
            int profitable = 0, total = 0;

            for (int si = 0; si < ShortRange; si++)
            {
                for (int li = 0; li < LongRange; li++)
                {
                    int shortP = SHORT_MA_MIN + si;
                    int longP = LONG_MA_MIN + li;
                    if (shortP >= longP) continue;

                    int idx = si * LongRange + li;
                    total++;
                    if (returns[idx] > 0) profitable++;

                    if (returns[idx] > bestReturn)
                    {
                        bestReturn = returns[idx];
                        bestShort = shortP;
                        bestLong = longP;
                    }
                    if (returns[idx] < worstReturn)
                    {
                        worstReturn = returns[idx];
                        worstShort = shortP;
                        worstLong = longP;
                    }
                }
            }

            Console.WriteLine();
            Console.WriteLine("┌────────────────────────────────────────────────────┐");
            Console.WriteLine("│              STRATEGY ANALYSIS                     │");
            Console.WriteLine("├────────────────────────────────────────────────────┤");
            Console.WriteLine($"│ Best:  MA({bestShort},{bestLong}) → {bestReturn:+0.00;-0.00}% return       ");
            Console.WriteLine($"│ Worst: MA({worstShort},{worstLong}) → {worstReturn:+0.00;-0.00}% return    ");
            Console.WriteLine($"│ Profitable: {profitable}/{total} ({100.0 * profitable / total:F1}%)               ");
            Console.WriteLine("└────────────────────────────────────────────────────┘");
        }
    }
}
