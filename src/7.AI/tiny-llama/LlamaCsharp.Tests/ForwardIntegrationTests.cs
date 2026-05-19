using LlamaCsharp.Gguf;
using LlamaCsharp.Math;
using LlamaCsharp.Model;
using LlamaCsharp.Sampling;
using LlamaCsharp.Tokenizer;
using LlamaCsharp.Utils;
using Xunit;
using Xunit.Abstractions;

namespace LlamaCsharp.Tests;

/// <summary>
/// End-to-end forward-pass tests against the local tinyllama-1.1b-chat-v1.0.Q8_0.gguf.
/// Greedy decoding (temperature = 0) is fully deterministic — token IDs are pinned.
/// </summary>
[Trait("Category", "Integration")]
public class ForwardIntegrationTests
{
    private readonly ITestOutputHelper _output;

    public ForwardIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private const string ModelFileName = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    private const string Prompt = "Once upon a time";
    private const int TokensToGenerate = 20;

    /// <summary>
    /// Golden token IDs produced by greedy decoding from "Once upon a time" on TinyLlama-1.1B Q8_0
    /// with the pure-C# baseline implementation (captured 2026-05-12).
    /// Decodes to: ", there was a young woman named Lily. She lived in a small village, surrounded by fields"
    /// Any drift means a regression in the forward pass.
    /// </summary>
    private static readonly int[] GoldenGeneratedTokenIds =
    {
        29892, 727, 471, 263, 4123, 6114, 4257, 365, 2354, 29889,
        2296, 10600, 297, 263, 2319, 5720, 29892, 22047, 491, 4235,
    };

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "llama.csharp.sln")))
                return dir.FullName;
            dir = dir.Parent;
        }
        throw new InvalidOperationException("Could not locate repo root (no llama.csharp.sln found)");
    }

    private static string ModelPath() => Path.Combine(FindRepoRoot(), ModelFileName);

    public static bool ModelAvailable() => File.Exists(ModelPath());

    [SkippableFact]
    public void Greedy_Decode_Managed_ProducesStableTokenSequence()
        => RunGreedyAndAssertGolden(ComputeBackend.Managed);

    [SkippableFact]
    public void Greedy_Decode_Cuda_ProducesStableTokenSequence()
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built next to test assembly");
        RunGreedyAndAssertGolden(ComputeBackend.Cuda);
    }

    [SkippableFact]
    public void Greedy_Decode_Omp_ProducesStableTokenSequence()
    {
        Skip.IfNot(SatelliteLoader.OmpAvailable(), "OMP satellite not built next to test assembly");
        RunGreedyAndAssertGolden(ComputeBackend.Omp);
    }

    /// <summary>
    /// CUDA greedy decode through <see cref="LlamaTransformer.ForwardArgmaxDeferred"/>
    /// — the per-token path does zero D→H (Iter 3b). Tokens accumulate in a
    /// device ring buffer; a single
    /// <see cref="LlamaTransformer.DrainPendingTokens"/> call at the end
    /// fetches all 20 ids in one D→H. Asserts the same 20-token Lily
    /// sequence as the immediate-print path.
    /// </summary>
    [SkippableFact]
    public void Greedy_Decode_CudaGpuSampleDeferred_ProducesStableTokenSequence()
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built next to test assembly");
        Skip.IfNot(ModelAvailable(), $"Model file not found at {ModelPath()} — skipping integration test");

        GpuBackend.Activate(ComputeBackend.Cuda);
        try
        {
            var (config, transformer, tokenizer, _) = LoadGreedy();
            transformer.ResetDeferredTokenRing();

            var promptTokens = tokenizer.Encode(Prompt, addBos: true);

            int currentToken = promptTokens[0];
            for (int pos = 0; pos < promptTokens.Count; pos++)
            {
                currentToken = promptTokens[pos];
                if (pos == promptTokens.Count - 1)
                    currentToken = transformer.ForwardArgmaxGpu(currentToken, pos);
                else
                    transformer.Forward(currentToken, pos);
            }

            transformer.SeedNextTokenIdFromHost(currentToken);
            transformer.SeedDecodePosition(promptTokens.Count);

            // The very first generated token (the one we just sampled from
            // the prompt's last logits) is on the host; record it directly
            // and let the deferred loop produce the remaining 19.
            var generated = new List<int> { currentToken };
            for (int i = 1; i < TokensToGenerate; i++)
            {
                transformer.ForwardArgmaxDeferred();
            }

            // Final flush — drain whatever the ring accumulated.
            int[] drainBuf = new int[TokensToGenerate];
            int drained = transformer.DrainPendingTokens(drainBuf);
            for (int i = 0; i < drained; i++)
                generated.Add(drainBuf[i]);

            _output.WriteLine($"[CudaGpuSampleDeferred] drained={drained} generated token IDs ({generated.Count}):");
            _output.WriteLine($"  {{ {string.Join(", ", generated)} }}");
            Assert.Equal(GoldenGeneratedTokenIds, generated.ToArray());
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }
    }

    /// <summary>
    /// CUDA greedy decode through <see cref="LlamaTransformer.ForwardArgmaxGpuFromDevice"/>
    /// — the next-token id stays on the device between consecutive Forwards
    /// (Iter 3a). Asserts the same 20-token Lily sequence as every other
    /// greedy variant; any drift indicates a bug in
    /// <c>CopyRowToResidentByDeviceIdx</c> or the device-fed embedding lookup.
    /// </summary>
    [SkippableFact]
    public void Greedy_Decode_CudaGpuSampleDeviceFed_ProducesStableTokenSequence()
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built next to test assembly");
        Skip.IfNot(ModelAvailable(), $"Model file not found at {ModelPath()} — skipping integration test");

        GpuBackend.Activate(ComputeBackend.Cuda);
        try
        {
            var (config, transformer, tokenizer, _) = LoadGreedy();

            var promptTokens = tokenizer.Encode(Prompt, addBos: true);

            int currentToken = promptTokens[0];
            for (int pos = 0; pos < promptTokens.Count; pos++)
            {
                currentToken = promptTokens[pos];
                if (pos == promptTokens.Count - 1)
                    currentToken = transformer.ForwardArgmaxGpu(currentToken, pos);
                else
                    transformer.Forward(currentToken, pos);
            }

            transformer.SeedNextTokenIdFromHost(currentToken);

            var generated = new List<int>();
            int position = promptTokens.Count;
            for (int i = 0; i < TokensToGenerate; i++)
            {
                generated.Add(currentToken);
                if (currentToken == tokenizer.EosTokenId) break;
                currentToken = transformer.ForwardArgmaxGpuFromDevice(position);
                position++;
            }

            _output.WriteLine($"[CudaGpuSampleDeviceFed] generated token IDs ({generated.Count}):");
            _output.WriteLine($"  {{ {string.Join(", ", generated)} }}");
            Assert.Equal(GoldenGeneratedTokenIds, generated.ToArray());
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }
    }

    /// <summary>
    /// CUDA greedy decode driven by <see cref="LlamaTransformer.ForwardArgmaxGpu"/>
    /// — bypasses the host <see cref="Sampler.Sample"/> entirely, picks the
    /// next token via the on-device <c>ArgmaxFullyResident</c> dispatcher and
    /// reads back only the 4-byte int. Asserts the same 20-token id sequence
    /// as the host-greedy path; any tie-break drift or argmax-kernel bug
    /// breaks this test before it reaches Lily.
    /// </summary>
    [SkippableFact]
    public void Greedy_Decode_CudaGpuSample_ProducesStableTokenSequence()
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built next to test assembly");
        Skip.IfNot(ModelAvailable(), $"Model file not found at {ModelPath()} — skipping integration test");

        GpuBackend.Activate(ComputeBackend.Cuda);
        try
        {
            var (config, transformer, tokenizer, _) = LoadGreedy();

            var promptTokens = tokenizer.Encode(Prompt, addBos: true);
            _output.WriteLine($"[CudaGpuSample] prompt tokens: [{string.Join(", ", promptTokens)}]");

            int currentToken = promptTokens[0];
            for (int pos = 0; pos < promptTokens.Count; pos++)
            {
                currentToken = promptTokens[pos];
                if (pos == promptTokens.Count - 1)
                {
                    currentToken = transformer.ForwardArgmaxGpu(currentToken, pos);
                }
                else
                {
                    transformer.Forward(currentToken, pos);
                }
            }

            var generated = new List<int>();
            int position = promptTokens.Count;
            for (int i = 0; i < TokensToGenerate; i++)
            {
                generated.Add(currentToken);
                if (currentToken == tokenizer.EosTokenId) break;
                currentToken = transformer.ForwardArgmaxGpu(currentToken, position);
                position++;
            }

            _output.WriteLine($"[CudaGpuSample] generated token IDs ({generated.Count}):");
            _output.WriteLine($"  {{ {string.Join(", ", generated)} }}");
            Assert.Equal(GoldenGeneratedTokenIds, generated.ToArray());
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }
    }

    private void RunGreedyAndAssertGolden(ComputeBackend backend)
    {
        Skip.IfNot(ModelAvailable(), $"Model file not found at {ModelPath()} — skipping integration test");

        GpuBackend.Activate(backend);
        try
        {
            var (config, transformer, tokenizer, sampler) = LoadGreedy();

            var promptTokens = tokenizer.Encode(Prompt, addBos: true);
            _output.WriteLine($"[{backend}] prompt tokens: [{string.Join(", ", promptTokens)}]");

            int currentToken = promptTokens[0];
            for (int pos = 0; pos < promptTokens.Count; pos++)
            {
                currentToken = promptTokens[pos];
                float[] logits = transformer.Forward(currentToken, pos);
                if (pos == promptTokens.Count - 1)
                    currentToken = sampler.Sample(logits, config.VocabSize);
            }

            var generated = new List<int>();
            int position = promptTokens.Count;
            for (int i = 0; i < TokensToGenerate; i++)
            {
                generated.Add(currentToken);
                if (currentToken == tokenizer.EosTokenId) break;
                float[] logits = transformer.Forward(currentToken, position);
                currentToken = sampler.Sample(logits, config.VocabSize);
                position++;
            }

            _output.WriteLine($"[{backend}] generated token IDs ({generated.Count}):");
            _output.WriteLine($"  {{ {string.Join(", ", generated)} }}");
            _output.WriteLine($"[{backend}] decoded: \"{string.Concat(generated.Select(tokenizer.Decode))}\"");

            if (GoldenGeneratedTokenIds.Length == 0)
            {
                Assert.Fail(
                    "GoldenGeneratedTokenIds is empty — first-run capture. " +
                    $"Copy the generated sequence into the test source:\n  {{ {string.Join(", ", generated)} }}");
            }

            Assert.Equal(GoldenGeneratedTokenIds, generated.ToArray());
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }
    }

    [SkippableFact]
    public void Forward_FirstTokenLogits_AreFinite()
    {
        Skip.IfNot(ModelAvailable(), $"Model file not found at {ModelPath()} — skipping integration test");

        var (config, transformer, tokenizer, _) = LoadGreedy();
        var promptTokens = tokenizer.Encode(Prompt, addBos: true);

        float[] logits = transformer.Forward(promptTokens[0], 0);

        Assert.Equal(config.VocabSize, logits.Length);
        int nanCount = 0, infCount = 0, nonZeroCount = 0;
        float maxAbs = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (float.IsNaN(logits[i])) nanCount++;
            else if (float.IsInfinity(logits[i])) infCount++;
            else
            {
                if (logits[i] != 0f) nonZeroCount++;
                float a = MathF.Abs(logits[i]);
                if (a > maxAbs) maxAbs = a;
            }
        }
        _output.WriteLine($"logits: nonZero={nonZeroCount}/{logits.Length}, NaN={nanCount}, Inf={infCount}, maxAbs={maxAbs}");

        Assert.Equal(0, nanCount);
        Assert.Equal(0, infCount);
        Assert.True(nonZeroCount > logits.Length / 2, $"only {nonZeroCount}/{logits.Length} logits non-zero");
    }

    private static (LlamaConfig, LlamaTransformer, BpeTokenizer, Sampler) LoadGreedy()
    {
        var gguf = new GgufReader(ModelPath());
        gguf.Load();
        var config = LlamaConfig.FromGguf(gguf);
        var weights = LlamaWeights.FromGguf(gguf, config);
        var tokenizer = new BpeTokenizer(gguf);
        var transformer = new LlamaTransformer(config, weights);
        var sampler = new Sampler(temperature: 0f, topP: 1.0f, seed: 0);
        return (config, transformer, tokenizer, sampler);
    }
}
