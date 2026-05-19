using LlamaCsharp.Math;

namespace LlamaCsharp.Sampling;

/// <summary>
/// Token sampling strategies for text generation.
/// Supports greedy (argmax), temperature scaling, and top-p (nucleus) sampling.
/// </summary>
public class Sampler
{
    private readonly Random _rng;
    private readonly float _temperature;
    private readonly float _topP;
    private TokenProb[] _topPBuffer = Array.Empty<TokenProb>();

    private readonly record struct TokenProb(int Idx, float Prob);

    private sealed class TokenProbComparer : IComparer<TokenProb>
    {
        public static readonly TokenProbComparer Instance = new();

        public int Compare(TokenProb x, TokenProb y) => y.Prob.CompareTo(x.Prob);
    }

    /// <summary>
    /// Create a sampler with the given parameters.
    /// </summary>
    /// <param name="temperature">Temperature for softmax. 0 = greedy, 1 = default, >1 = more random.</param>
    /// <param name="topP">Top-p (nucleus) sampling threshold. 1.0 = disabled.</param>
    /// <param name="seed">Random seed for reproducibility. -1 = random.</param>
    public Sampler(float temperature = 0.7f, float topP = 0.9f, int seed = -1)
    {
        _temperature = temperature;
        _topP = topP;
        _rng = seed >= 0 ? new Random(seed) : new Random();
    }

    /// <summary>
    /// Sample the next token from the logits distribution.
    /// Mutates the logits buffer in place to avoid per-token allocations.
    /// </summary>
    public int Sample(float[] logits, int vocabSize)
    {
        if (_temperature < 1e-6f)
            return ParallelMath.Argmax(logits, vocabSize);

        float invTemp = 1.0f / _temperature;
        for (int i = 0; i < vocabSize; i++)
            logits[i] *= invTemp;

        ParallelMath.Softmax(logits, vocabSize);

        if (_topP < 1.0f)
            return SampleTopP(logits, vocabSize);

        return SampleCategorical(logits, vocabSize);
    }

    /// <summary>
    /// Sample from the distribution using top-p (nucleus) filtering.
    /// Only considers the smallest set of tokens whose cumulative probability >= topP.
    /// </summary>
    private int SampleTopP(float[] probs, int vocabSize)
    {
        TokenProb[] indexed = GetTopPBuffer(vocabSize);
        for (int i = 0; i < vocabSize; i++)
            indexed[i] = new TokenProb(i, probs[i]);

        Array.Sort(indexed, 0, vocabSize, TokenProbComparer.Instance);

        float cumulative = 0f;
        int cutoff = vocabSize;
        for (int i = 0; i < vocabSize; i++)
        {
            cumulative += indexed[i].Prob;
            if (cumulative >= _topP)
            {
                cutoff = i + 1;
                break;
            }
        }

        float sum = 0f;
        for (int i = 0; i < cutoff; i++)
            sum += indexed[i].Prob;

        float r = (float)_rng.NextDouble() * sum;
        float cdf = 0f;
        for (int i = 0; i < cutoff; i++)
        {
            cdf += indexed[i].Prob;
            if (cdf >= r)
                return indexed[i].Idx;
        }

        return indexed[0].Idx;
    }

    /// <summary>
    /// Simple categorical sampling from a probability distribution.
    /// </summary>
    private int SampleCategorical(float[] probs, int vocabSize)
    {
        float r = (float)_rng.NextDouble();
        float cdf = 0f;
        for (int i = 0; i < vocabSize; i++)
        {
            cdf += probs[i];
            if (cdf >= r)
                return i;
        }

        return vocabSize - 1;
    }

    private TokenProb[] GetTopPBuffer(int vocabSize)
    {
        if (_topPBuffer.Length < vocabSize)
            _topPBuffer = new TokenProb[vocabSize];

        return _topPBuffer;
    }
}
