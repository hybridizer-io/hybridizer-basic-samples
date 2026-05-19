using System.Text;
using LlamaCsharp.Gguf;

namespace LlamaCsharp.Tokenizer;

/// <summary>
/// BPE (Byte Pair Encoding) tokenizer compatible with LLaMA/SentencePiece models.
/// Reads vocabulary, scores, and merge rules from GGUF metadata.
/// </summary>
public class BpeTokenizer
{
    private readonly string[] _vocab;
    private readonly float[] _scores;
    private readonly int[] _tokenTypes;
    private readonly Dictionary<string, int> _tokenToId;

    public int BosTokenId { get; }
    public int EosTokenId { get; }
    public int UnknownTokenId { get; }
    public int VocabSize => _vocab.Length;

    // Token type constants (from GGUF spec)
    private const int TOKEN_TYPE_NORMAL = 1;
    private const int TOKEN_TYPE_UNKNOWN = 2;
    private const int TOKEN_TYPE_CONTROL = 3;
    private const int TOKEN_TYPE_USER_DEFINED = 4;
    private const int TOKEN_TYPE_UNUSED = 5;
    private const int TOKEN_TYPE_BYTE = 6;

    public BpeTokenizer(GgufReader reader)
    {
        // Load vocabulary tokens
        var tokensMeta = reader.GetMetadata("tokenizer.ggml.tokens")
            ?? throw new InvalidDataException("Missing tokenizer.ggml.tokens");
        _vocab = tokensMeta.GetStringArray();

        // Load scores (priorities for merging)
        var scoresMeta = reader.GetMetadata("tokenizer.ggml.scores");
        _scores = scoresMeta != null
            ? scoresMeta.GetFloat32Array()
            : Enumerable.Repeat(0f, _vocab.Length).ToArray();

        // Load token types
        var typesMeta = reader.GetMetadata("tokenizer.ggml.token_type");
        _tokenTypes = typesMeta != null
            ? typesMeta.GetInt32Array()
            : Enumerable.Repeat(TOKEN_TYPE_NORMAL, _vocab.Length).ToArray();

        // Special tokens
        var bosMeta = reader.GetMetadata("tokenizer.ggml.bos_token_id");
        BosTokenId = bosMeta != null ? (int)bosMeta.GetUInt32() : 1;

        var eosMeta = reader.GetMetadata("tokenizer.ggml.eos_token_id");
        EosTokenId = eosMeta != null ? (int)eosMeta.GetUInt32() : 2;

        var unkMeta = reader.GetMetadata("tokenizer.ggml.unknown_token_id");
        UnknownTokenId = unkMeta != null ? (int)unkMeta.GetUInt32() : 0;

        // Build reverse lookup (token string -> id)
        _tokenToId = new Dictionary<string, int>(_vocab.Length);
        for (int i = 0; i < _vocab.Length; i++)
        {
            // Only add first occurrence (some vocabs have duplicates)
            _tokenToId.TryAdd(_vocab[i], i);
        }

        Console.WriteLine($"  Tokenizer: {_vocab.Length:N0} tokens | BOS={BosTokenId} | EOS={EosTokenId}");
    }

    /// <summary>
    /// Encode a text string into a sequence of token IDs.
    /// Uses SentencePiece-style BPE:
    ///   1. Convert text to initial character/byte tokens
    ///   2. Iteratively merge the best pair (highest score)
    /// </summary>
    public List<int> Encode(string text, bool addBos = true)
    {
        var tokens = new List<int>();
        if (addBos)
            tokens.Add(BosTokenId);

        if (string.IsNullOrEmpty(text))
            return tokens;

        // SentencePiece prepends a space before the text
        text = " " + text;

        // Step 1: Initialize with individual UTF-8 characters or byte fallbacks
        // First try to find each character as a token (with the SentencePiece ▁ prefix for spaces)
        var charTokens = new List<int>();

        int i = 0;
        while (i < text.Length)
        {
            // Try to find the longest matching token starting at position i
            // For initial encoding, we try single characters
            string ch;
            if (text[i] == ' ')
            {
                ch = "▁"; // SentencePiece space marker
            }
            else
            {
                ch = text[i].ToString();
            }

            if (_tokenToId.TryGetValue(ch, out int charId))
            {
                charTokens.Add(charId);
                i++;
            }
            else
            {
                // Byte fallback: encode as <0xHH> tokens
                byte[] bytes = Encoding.UTF8.GetBytes(text[i].ToString());
                foreach (byte b in bytes)
                {
                    string byteToken = $"<0x{b:X2}>";
                    if (_tokenToId.TryGetValue(byteToken, out int byteId))
                    {
                        charTokens.Add(byteId);
                    }
                    else
                    {
                        charTokens.Add(UnknownTokenId);
                    }
                }
                i++;
            }
        }

        // Step 2: Iteratively merge the best adjacent pair
        // The "best" pair is the one whose merged token has the highest score
        while (charTokens.Count >= 2)
        {
            float bestScore = float.NegativeInfinity;
            int bestIdx = -1;
            int bestTokenId = -1;

            // Find the best merge
            for (int j = 0; j < charTokens.Count - 1; j++)
            {
                string merged = _vocab[charTokens[j]] + _vocab[charTokens[j + 1]];
                if (_tokenToId.TryGetValue(merged, out int mergedId))
                {
                    float score = _scores[mergedId];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestIdx = j;
                        bestTokenId = mergedId;
                    }
                }
            }

            // If no merge is possible, we're done
            if (bestIdx == -1)
                break;

            // Apply the merge
            charTokens[bestIdx] = bestTokenId;
            charTokens.RemoveAt(bestIdx + 1);
        }

        tokens.AddRange(charTokens);
        return tokens;
    }

    /// <summary>
    /// Decode a single token ID to its string representation.
    /// Handles the SentencePiece space marker (▁ → space) and byte fallback tokens.
    /// </summary>
    public string Decode(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _vocab.Length)
            return "";

        string token = _vocab[tokenId];

        // Control tokens (BOS, EOS, etc.) produce no text
        if (_tokenTypes[tokenId] == TOKEN_TYPE_CONTROL)
            return "";

        // Handle byte fallback tokens: <0xHH>
        if (token.StartsWith("<0x") && token.EndsWith(">") && token.Length == 6)
        {
            if (byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte b))
            {
                return ((char)b).ToString();
            }
        }

        // Replace SentencePiece space marker with actual space
        token = token.Replace("▁", " ");

        return token;
    }

    /// <summary>
    /// Decode a sequence of token IDs to text.
    /// </summary>
    public string Decode(IEnumerable<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (int id in tokenIds)
            sb.Append(Decode(id));
        return sb.ToString();
    }
}
