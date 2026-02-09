using System.Diagnostics;
using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MonteCarloHeatEquation
{
    class Program
    {
        static void Main(string[] args)
        {
            // adjust these numbers depending on your graphics card, this is quite compute intensive^^
            const int N = 128;
            const int iterCount = 512;

            var problem = new SquareProblem<SimpleWalker, SimpleBoundaryCondition>(N, iterCount);
            // example of another instanciation
            // var problem = new TetrisProblem<SimpleWalker, TetrisBoundaryCondition>(N, iterCount);

            cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);

            HybRunner runner = SatelliteLoader.Load().SetDistrib(16 * prop.multiProcessorCount, 128);
            var solver = new MonteCarloHeatSolver(problem);
            dynamic wrapped = runner.Wrap(solver);
            
            TraceExec(solver.Solve, "C#");
            TraceExec(() => wrapped.Solve(), "CUDA");

            problem.RefreshHost();
            problem.SaveImage("result.png", GetColor);
        }

        private static void TraceExec(Action solve, string label)
        {
            Stopwatch watch = new();
            watch.Start();
            solve();
            watch.Stop();
            Console.WriteLine($"{label} time : {watch.ElapsedMilliseconds} ms");
        }

        /// <summary>
        /// from white (warm) to black (cold) following rainbow colors
        /// </summary>
        static Color GetColor(float temperature)
        {
            int map = (int)Math.Floor(temperature * 8.0F);
            if (temperature <= 0.0F)
                return Color.Black;
            if (temperature >= 1.0F)
                return Color.White;
            float t = 8.0F * temperature - (float)Math.Floor(temperature * 8.0F);
            Color[] colors = [Color.Black, Color.Red, Color.Orange, Color.Yellow, Color.Green, Color.Blue, Color.Indigo, Color.Violet, Color.White];
            return Interpolate(colors[map], colors[map + 1], t);
        }

        static Color Interpolate(Rgba32 a, Rgba32 b, float t)
        {
            return Color.FromRgb((byte)((1.0F - t) * a.R + t * b.R), (byte)((1.0F - t) * a.G + t * b.G), (byte)((1.0F - t) * a.B + t * b.B));
        }
    }
}