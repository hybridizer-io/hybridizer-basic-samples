using System.Globalization;
using System.Text.Json;

namespace Hybridizer.Basic.Finance
{
    /// <summary>
    /// Loads price data from a local CSV file or Yahoo Finance API.
    /// </summary>
    static class PriceDataLoader
    {
        /// <summary>
        /// Parses command-line arguments and loads price data accordingly.
        /// </summary>
        public static float[] LoadPrices(string[] args, int minDataPoints)
        {
            string? filePath = null;
            string symbol = "BTC-USD";
            int days = 3650;

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--file" && i + 1 < args.Length)
                    filePath = args[++i];
                else if (args[i] == "--symbol" && i + 1 < args.Length)
                    symbol = args[++i];
                else if (args[i] == "--days" && i + 1 < args.Length)
                    days = int.Parse(args[++i]);
            }

            if (filePath != null)
            {
                Console.WriteLine($"Loading prices from file: {filePath}");
                return LoadFromCsv(filePath, minDataPoints);
            }
            else
            {
                Console.WriteLine($"Downloading {symbol} data ({days} days)...");
                return DownloadPrices(symbol, days, minDataPoints);
            }
        }

        static float[] LoadFromCsv(string path, int minDataPoints)
        {
            var lines = File.ReadAllLines(path);
            if (lines.Length < 2)
                throw new Exception("CSV file is empty or has no data rows.");

            // Find "Close" column in header
            var header = lines[0].Split(',');
            int closeIdx = Array.FindIndex(header, h =>
                h.Trim().Equals("Close", StringComparison.OrdinalIgnoreCase));
            if (closeIdx < 0)
                closeIdx = Array.FindIndex(header, h =>
                    h.Trim().Equals("Adj Close", StringComparison.OrdinalIgnoreCase));
            if (closeIdx < 0)
                closeIdx = 1; // fallback: second column

            var prices = new List<float>();
            for (int i = 1; i < lines.Length; i++)
            {
                var parts = lines[i].Split(',');
                if (parts.Length > closeIdx &&
                    float.TryParse(parts[closeIdx], NumberStyles.Float, CultureInfo.InvariantCulture, out float val)
                    && val > 0)
                {
                    prices.Add(val);
                }
            }

            if (prices.Count < minDataPoints)
                throw new Exception($"Not enough data points ({prices.Count}). Need at least {minDataPoints}.");

            return prices.ToArray();
        }

        static float[] DownloadPrices(string symbol, int days, int minDataPoints)
        {
            try
            {
                long now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                long start = now - (long)days * 86400;
                string url = $"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start}&period2={now}&interval=1d";

                using var client = new HttpClient();
                client.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0");
                var response = client.GetStringAsync(url).Result;

                using var doc = JsonDocument.Parse(response);
                var closes = doc.RootElement
                    .GetProperty("chart")
                    .GetProperty("result")[0]
                    .GetProperty("indicators")
                    .GetProperty("quote")[0]
                    .GetProperty("close");

                var prices = new List<float>();
                foreach (var el in closes.EnumerateArray())
                {
                    if (el.ValueKind == JsonValueKind.Number)
                        prices.Add((float)el.GetDouble());
                    else
                        prices.Add(prices.Count > 0 ? prices[^1] : 0); // fill nulls with last known
                }

                // Remove leading zeros
                while (prices.Count > 0 && prices[0] == 0)
                    prices.RemoveAt(0);

                if (prices.Count < minDataPoints)
                    throw new Exception($"Only {prices.Count} data points retrieved.");

                Console.WriteLine($"Downloaded {prices.Count} daily prices for {symbol}");
                return prices.ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to download data: {ex.Message}");
                Console.WriteLine();
                Console.WriteLine("Please download a CSV file manually and use --file option:");
                Console.WriteLine("  1. Go to https://finance.yahoo.com/quote/BTC-USD/history/");
                Console.WriteLine("  2. Set time period and click Download");
                Console.WriteLine($"  3. Run: StrategyBacktest --file downloaded.csv");
                Console.WriteLine();
                Console.WriteLine("Falling back to synthetic price data for demo purposes...");
                return GenerateSyntheticPrices(2500);
            }
        }

        /// <summary>
        /// Generates realistic-looking synthetic price data using geometric Brownian motion.
        /// Used as fallback when real data cannot be downloaded.
        /// </summary>
        static float[] GenerateSyntheticPrices(int count)
        {
            var prices = new float[count];
            var rand = new Random(42);
            float price = 100.0f;
            float drift = 0.0003f;   // slight upward bias
            float vol = 0.02f;       // 2% daily volatility

            for (int i = 0; i < count; i++)
            {
                prices[i] = price;
                // Box-Muller for normal distribution
                float u1 = (float)rand.NextDouble();
                float u2 = (float)rand.NextDouble();
                float z = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                price *= (float)Math.Exp(drift + vol * z);
                if (price < 1.0f) price = 1.0f;
            }

            Console.WriteLine($"Generated {count} synthetic price data points (GBM model)");
            return prices;
        }
    }
}
