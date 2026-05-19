using System.Diagnostics;
using LlamaCsharp.Gguf;
using LlamaCsharp.Math;
using LlamaCsharp.Model;
using LlamaCsharp.Sampling;
using LlamaCsharp.Tokenizer;

namespace LlamaCsharp;

/// <summary>
/// LLama.CSharp — A pure C# LLM inference engine for educational purposes.
/// Loads GGUF models (LLaMA architecture) and generates text using all available CPUs.
/// </summary>
public static class Program
{
    private const string DefaultPrompt = "Once upon a time";
    private const int DefaultMaxTokens = 1280;
    private const float DefaultTemperature = 0.7f;
    private const float DefaultTopP = 0.9f;

    public static void Main(string[] args)
    {
        ConfigureRuntime();

        // ================================================================
        // Parse command line arguments
        // ================================================================
        if (args.Length < 1)
        {
            PrintUsage();
            return;
        }

        string modelPath = args[0];
        string prompt = DefaultPrompt;
        int maxTokens = DefaultMaxTokens;
        float temperature = DefaultTemperature;
        float topP = DefaultTopP;
        ComputeBackend backend = ComputeBackend.Managed;

        // Parse optional arguments
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--max-tokens" when i + 1 < args.Length:
                    maxTokens = int.Parse(args[++i]);
                    break;
                case "--temperature" when i + 1 < args.Length:
                    temperature = float.Parse(args[++i]);
                    break;
                case "--top-p" when i + 1 < args.Length:
                    topP = float.Parse(args[++i]);
                    break;
                case "--backend" when i + 1 < args.Length:
                    backend = ParseBackend(args[++i]);
                    break;
                default:
                    // If it's not a flag, treat it as the prompt
                    if (!args[i].StartsWith("--"))
                        prompt = args[i];
                    break;
            }
        }

        GpuBackend.Activate(backend);

        // Optional in-process kernel timing. Set LLAMA_KERNEL_PROFILE=1 to
        // enable. Forces cudaDeviceSynchronize after every kernel launch,
        // so inference is significantly slower under profiling, but the
        // per-kernel ranking is accurate enough to pick the next target.
        bool profileKernels = backend == ComputeBackend.Cuda
            && Environment.GetEnvironmentVariable("LLAMA_KERNEL_PROFILE") == "1";
        if (profileKernels)
            LlamaCsharp.Utils.CudaInvoke.EnableProfiling();

        // ================================================================
        // System info
        // ================================================================
        Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║          🦙 LLama.CSharp — Pure C# LLM Inference       ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"  CPU cores:      {Environment.ProcessorCount}");
        Console.WriteLine($"  .NET Runtime:   {Environment.Version}");
        Console.WriteLine($"  Model file:     {Path.GetFileName(modelPath)}");
        Console.WriteLine($"  Prompt:         \"{prompt}\"");
        Console.WriteLine($"  Max tokens:     {maxTokens}");
        Console.WriteLine($"  Temperature:    {temperature:F2}");
        Console.WriteLine($"  Top-p:          {topP:F2}");
        Console.WriteLine($"  Backend:        {backend}");
        Console.WriteLine();

        // ================================================================
        // Load GGUF model
        // ================================================================
        if (!File.Exists(modelPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"  ERROR: Model file not found: {modelPath}");
            Console.ResetColor();
            return;
        }

        var totalTimer = Stopwatch.StartNew();

        Console.WriteLine("── Loading Model ──────────────────────────────────────────");

        var loadTimer = Stopwatch.StartNew();
        var gguf = new GgufReader(modelPath);
        gguf.Load();

        // Extract model configuration
        var config = LlamaConfig.FromGguf(gguf);
        config.PrintSummary();
        Console.WriteLine();

        // Load and dequantize all weights
        Console.WriteLine("── Loading Weights ────────────────────────────────────────");
        var weights = LlamaWeights.FromGguf(gguf, config);

        loadTimer.Stop();
        long memoryMB = GC.GetTotalMemory(false) / (1024 * 1024);
        Console.WriteLine($"  Load time:      {loadTimer.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"  Memory used:    ~{memoryMB:N0} MB");
        Console.WriteLine();

        // ================================================================
        // Initialize tokenizer
        // ================================================================
        Console.WriteLine("── Tokenizer ──────────────────────────────────────────────");
        var tokenizer = new BpeTokenizer(gguf);
        Console.WriteLine();

        // ================================================================
        // Initialize transformer and sampler
        // ================================================================
        var transformer = new LlamaTransformer(config, weights);
        var sampler = new Sampler(temperature, topP);

        // GPU-greedy fast path: when the CUDA backend is active and the user
        // asked for deterministic decoding (temperature 0, top-p 1) we skip
        // the host-side argmax in Sampler.Sample and run a single
        // ArgmaxFullyResident on device, eliminating the 125 KB D→H copy of
        // _logitsResident per token. Stochastic CUDA, Managed and OMP stay on
        // the existing Forward + Sample path.
        bool gpuGreedy = backend == ComputeBackend.Cuda && temperature == 0f && topP == 1f;

        // ================================================================
        // Encode the prompt
        // ================================================================
        var promptTokens = tokenizer.Encode(prompt, addBos: true);
        Console.WriteLine($"  Prompt tokens:  [{string.Join(", ", promptTokens)}] ({promptTokens.Count} tokens)");
        Console.WriteLine();

        // ================================================================
        // Inference loop
        // ================================================================
        Console.WriteLine("── Generation ─────────────────────────────────────────────");
        Console.Write("  ");

        // Print the prompt
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write(prompt);
        Console.ForegroundColor = ConsoleColor.White;

        var genTimer = Stopwatch.StartNew();
        int generatedCount = 0;

        // Process prompt tokens first (prefill). On the GPU-greedy path the
        // non-last prefill tokens use ForwardPrefillNoLogits — same layer
        // loop, no trailing D→H of the 125 KB logits buffer (the result was
        // discarded anyway). The last prefill token uses ForwardArgmaxGpu to
        // sample the first generated id on device.
        int currentToken = promptTokens[0];
        for (int pos = 0; pos < promptTokens.Count; pos++)
        {
            currentToken = promptTokens[pos];
            bool isLast = pos == promptTokens.Count - 1;
            if (gpuGreedy)
            {
                if (isLast)
                    currentToken = transformer.ForwardArgmaxGpu(currentToken, pos);
                else
                    transformer.ForwardPrefillNoLogits(currentToken, pos);
            }
            else
            {
                float[] logits = transformer.Forward(currentToken, pos);
                if (isLast)
                    currentToken = sampler.Sample(logits, config.VocabSize);
            }
        }

        // GPU-greedy steady state: the first chosen token (above) is on the
        // host. Seed the device-side next-token slot once so the generation
        // loop can feed it through CopyRowToResidentByDeviceIdx with no more
        // host→device plumbing of the row index.
        if (gpuGreedy)
        {
            transformer.ResetDeferredTokenRing();
            transformer.SeedNextTokenIdFromHost(currentToken);
            // Iter 7.A.7.a: seed the device-side decode position once before
            // entering the deferred loop. From here on, each ForwardArgmaxDeferred
            // bumps it device-side, so there are no host int args in the
            // per-token forward (prep for graph capture in 7.A.7.b).
            transformer.SeedDecodePosition(promptTokens.Count);

            // Print the very first generated token (held on host) before
            // entering the deferred device-only loop.
            Console.Write(tokenizer.Decode(currentToken));
            generatedCount++;
        }

        if (gpuGreedy)
        {
            // Deferred-print path: every Forward stays entirely on the GPU
            // (zero D→H per token). A 200 ms wall-clock timer drains the
            // device ring buffer; EOS is detected at drain time so the
            // generator may overshoot EOS by at most one drain interval.
            // At 87 tok/s post-Iter-7.A this means ~17 tokens per drain; the
            // ring is sized at 1024 (~50× safety margin).
            const int DrainIntervalMs = 200;
            int[] drainBuf = new int[1024];
            int position = promptTokens.Count;
            var drainTimer = Stopwatch.StartNew();
            bool stop = false;

            // Helper: drain whatever the device has produced, print + check EOS.
            int Flush()
            {
                int n = transformer.DrainPendingTokens(drainBuf);
                for (int k = 0; k < n; k++)
                {
                    int id = drainBuf[k];
                    Console.Write(tokenizer.Decode(id));
                    generatedCount++;
                    if (id == tokenizer.EosTokenId)
                    {
                        stop = true;
                        break;
                    }
                }
                return n;
            }

            for (int i = 0; i < maxTokens && !stop; i++)
            {
                if (position >= config.ContextLength - 1)
                {
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.Write("  [Context length limit reached]");
                    Console.ResetColor();
                    break;
                }

                transformer.ForwardArgmaxDeferred();
                position++;  // host-side mirror for the context-length check; device side bumps itself

                if (drainTimer.ElapsedMilliseconds >= DrainIntervalMs)
                {
                    Flush();
                    drainTimer.Restart();
                }
            }

            // Final flush — pick up tokens generated since the last timer fire.
            Flush();
        }
        else
        {
            // Default (host-greedy / stochastic / Managed / OMP) path —
            // unchanged.
            int position = promptTokens.Count;
            for (int i = 0; i < maxTokens; i++)
            {
                // Print the generated token
                string tokenStr = tokenizer.Decode(currentToken);
                Console.Write(tokenStr);
                generatedCount++;

                // Stop on EOS
                if (currentToken == tokenizer.EosTokenId)
                    break;

                // Check context length
                if (position >= config.ContextLength - 1)
                {
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.Write("  [Context length limit reached]");
                    Console.ResetColor();
                    break;
                }

                float[] logits = transformer.Forward(currentToken, position);
                currentToken = sampler.Sample(logits, config.VocabSize);
                position++;
            }
        }

        genTimer.Stop();
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine();

        // ================================================================
        // Statistics
        // ================================================================
        Console.WriteLine("── Statistics ─────────────────────────────────────────────");
        double tokPerSec = generatedCount / genTimer.Elapsed.TotalSeconds;
        Console.WriteLine($"  Generated:      {generatedCount} tokens");
        Console.WriteLine($"  Gen time:       {genTimer.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"  Speed:          {tokPerSec:F2} tokens/sec");
        Console.WriteLine($"  Total time:     {totalTimer.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"  Memory:         ~{GC.GetTotalMemory(false) / (1024 * 1024):N0} MB");
        Console.WriteLine();

        if (profileKernels)
            LlamaCsharp.Utils.CudaInvoke.PrintProfile();
    }

    private static void PrintUsage()
    {
        Console.WriteLine("🦙 LLama.CSharp — Pure C# LLM Inference Engine");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  llama-csharp <model.gguf> [prompt] [options]");
        Console.WriteLine();
        Console.WriteLine("Arguments:");
        Console.WriteLine("  model.gguf          Path to a GGUF model file (LLaMA architecture)");
        Console.WriteLine("  prompt              Optional prompt text (default: \"Once upon a time\")");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --max-tokens N      Maximum number of tokens to generate (default: 128)");
        Console.WriteLine("  --temperature T     Sampling temperature (default: 0.7, 0 = greedy)");
        Console.WriteLine("  --top-p P           Top-p nucleus sampling (default: 0.9, 1.0 = disabled)");
        Console.WriteLine("  --backend NAME      Q8_0 matvec backend: managed (default), cuda, omp");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  llama-csharp tinyllama-1.1b-q8_0.gguf");
        Console.WriteLine("  llama-csharp model.gguf \"Il était une fois\" --max-tokens 256");
        Console.WriteLine("  llama-csharp model.gguf \"The meaning of life\" --temperature 0 --max-tokens 64");
        Console.WriteLine();
        Console.WriteLine("Supported quantizations: F32, F16, Q8_0, Q4_0");
    }

    private static ComputeBackend ParseBackend(string value) => value.ToLowerInvariant() switch
    {
        "managed" => ComputeBackend.Managed,
        "cuda" => ComputeBackend.Cuda,
        "omp" => ComputeBackend.Omp,
        _ => throw new ArgumentException($"Unknown --backend value '{value}' (expected: managed, cuda, omp)"),
    };

    private static void ConfigureRuntime()
    {
        // Warm the worker pool early so the hottest Parallel.For regions can
        // occupy all cores immediately instead of ramping up gradually.
        ThreadPool.GetMinThreads(out int minWorkers, out int minIoThreads);
        int targetWorkers = Environment.ProcessorCount;
        if (minWorkers < targetWorkers)
            ThreadPool.SetMinThreads(targetWorkers, minIoThreads);
    }
}
