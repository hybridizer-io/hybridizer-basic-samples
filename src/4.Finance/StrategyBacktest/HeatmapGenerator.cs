using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Png;

namespace Hybridizer.Basic.Finance
{
    /// <summary>
    /// Generates heatmap visualizations of backtest results (PNG file + ANSI console output).
    /// </summary>
    static class HeatmapGenerator
    {
        /// <summary>
        /// Generates a PNG heatmap image with a diverging red–black–green colormap and color bar.
        /// </summary>
        public static void GeneratePng(
            float[] returns, string path,
            int shortMaMin, int shortMaMax, int shortRange,
            int longMaMin, int longMaMax, int longRange)
        {
            int width = longRange;
            int height = shortRange;
            int barWidth = 30;
            int margin = 5;
            int totalWidth = width + margin + barWidth;

            // Find min/max for valid combinations
            float minRet = float.MaxValue, maxRet = float.MinValue;
            for (int si = 0; si < shortRange; si++)
            {
                for (int li = 0; li < longRange; li++)
                {
                    if (shortMaMin + si >= longMaMin + li) continue;
                    int idx = si * longRange + li;
                    minRet = Math.Min(minRet, returns[idx]);
                    maxRet = Math.Max(maxRet, returns[idx]);
                }
            }

            // Symmetric scale around 0
            float absMax = Math.Max(Math.Abs(minRet), Math.Abs(maxRet));
            if (absMax < 1.0f) absMax = 1.0f;

            var image = new Image<Rgba32>(totalWidth, height);

            for (int si = 0; si < shortRange; si++)
            {
                for (int li = 0; li < longRange; li++)
                {
                    int shortP = shortMaMin + si;
                    int longP = longMaMin + li;
                    int idx = si * longRange + li;
                    int y = shortRange - 1 - si; // flip Y so lower short periods are at bottom

                    Rgba32 color;
                    if (shortP >= longP)
                    {
                        color = new Rgba32(30, 30, 30); // dark gray for invalid combos
                    }
                    else
                    {
                        color = ReturnToColor(returns[idx], absMax);
                    }

                    image[li, y] = color;
                }
            }

            // Draw color bar
            for (int y = 0; y < height; y++)
            {
                float t = 1.0f - (float)y / (height - 1); // top = +max, bottom = -max
                float val = (2.0f * t - 1.0f) * absMax;
                Rgba32 barColor = ReturnToColor(val, absMax);

                for (int x = width + margin; x < totalWidth; x++)
                    image[x, y] = barColor;
            }

            image.Save(path, new PngEncoder());
        }

        /// <summary>
        /// Prints a downsampled ANSI true-color heatmap to the console.
        /// </summary>
        public static void PrintConsoleHeatmap(
            float[] returns,
            int shortMaMin, int shortMaMax, int shortRange,
            int longMaMin, int longMaMax, int longRange)
        {
            Console.WriteLine();
            Console.WriteLine("Strategy Heatmap (Short MA ↑, Long MA →)");
            Console.WriteLine("  Red = loss, Green = profit");
            Console.WriteLine();

            // Downsample to fit console
            int cols = 70;
            int rows = 25;

            float minRet = float.MaxValue, maxRet = float.MinValue;
            for (int si = 0; si < shortRange; si++)
                for (int li = 0; li < longRange; li++)
                {
                    if (shortMaMin + si >= longMaMin + li) continue;
                    int idx = si * longRange + li;
                    minRet = Math.Min(minRet, returns[idx]);
                    maxRet = Math.Max(maxRet, returns[idx]);
                }

            float absMax = Math.Max(Math.Abs(minRet), Math.Abs(maxRet));
            if (absMax < 1.0f) absMax = 1.0f;

            // Y axis label
            Console.Write($" {shortMaMax,3} ");
            for (int c = 0; c < cols; c++) Console.Write("─");
            Console.WriteLine();

            for (int row = rows - 1; row >= 0; row--)
            {
                int siStart = row * shortRange / rows;
                int siEnd = (row + 1) * shortRange / rows;
                int shortLabel = shortMaMin + (siStart + siEnd) / 2;

                if (row == rows - 1 || row == rows / 2 || row == 0)
                    Console.Write($" {shortLabel,3} │");
                else
                    Console.Write($"     │");

                for (int col = 0; col < cols; col++)
                {
                    int liStart = col * longRange / cols;
                    int liEnd = (col + 1) * longRange / cols;

                    // Average return in this block
                    float sum = 0; int cnt = 0;
                    for (int si = siStart; si < siEnd; si++)
                        for (int li = liStart; li < liEnd; li++)
                        {
                            if (shortMaMin + si >= longMaMin + li) continue;
                            int idx = si * longRange + li;
                            sum += returns[idx];
                            cnt++;
                        }

                    if (cnt == 0)
                    {
                        Console.Write("\x1B[48;2;30;30;30m \x1B[0m");
                    }
                    else
                    {
                        float avg = sum / cnt;
                        var (cr, cg, cb) = ReturnToRgb(avg, absMax);
                        Console.Write($"\x1B[48;2;{cr};{cg};{cb}m \x1B[0m");
                    }
                }
                Console.WriteLine();
            }

            Console.Write($"   {shortMaMin,1} ");
            for (int c = 0; c < cols; c++) Console.Write("─");
            Console.WriteLine();
            Console.Write("      ");
            Console.Write($"{longMaMin}");
            for (int c = 0; c < cols - 8; c++) Console.Write(" ");
            Console.Write($"{longMaMax}");
            Console.WriteLine();
            Console.WriteLine("      └── Long MA period ──────────────────────────────────────────┘");
        }

        static Rgba32 ReturnToColor(float val, float absMax)
        {
            float t = val / absMax;
            t = Math.Clamp(t, -1.0f, 1.0f);

            byte r, g, b;
            if (t < 0)
            {
                float s = -t;
                r = (byte)(40 + 215 * s);
                g = (byte)(40 * (1 - s));
                b = (byte)(60 * (1 - s));
            }
            else
            {
                float s = t;
                r = (byte)(40 * (1 - s));
                g = (byte)(40 + 215 * s);
                b = (byte)(60 * (1 - s));
            }

            return new Rgba32(r, g, b);
        }

        static (byte r, byte g, byte b) ReturnToRgb(float val, float absMax)
        {
            float t = Math.Clamp(val / absMax, -1.0f, 1.0f);
            byte r, g, b;
            if (t < 0)
            {
                float s = -t;
                r = (byte)(40 + 215 * s);
                g = (byte)(40 * (1 - s));
                b = (byte)(60 * (1 - s));
            }
            else
            {
                float s = t;
                r = (byte)(40 * (1 - s));
                g = (byte)(40 + 215 * s);
                b = (byte)(60 * (1 - s));
            }
            return (r, g, b);
        }
    }
}
