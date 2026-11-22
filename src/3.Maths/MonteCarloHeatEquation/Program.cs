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

            gpu.GetDeviceProperties(out gpuDeviceProp prop, 0);

            HybRunner runner = SatelliteLoader.Load().SetDistrib(16 * prop.multiProcessorCount, 128);
            var solver = new MonteCarloHeatSolver(problem);
            dynamic wrapped = runner.Wrap(solver);
            
            //TraceExec(solver.Solve, "C#");
            TraceExec(() => wrapped.Solve(), "GPU");

            problem.RefreshHost();
            problem.SaveImage("result.png", GetColor);

            // The walks estimate the discrete harmonic function of the boundary temperatures, each point from
            // iterCount walks: an error of sqrt(u (1 - u) / iterCount) at most 0.022 for 512 walks
            float[,] exact = SolveLaplace(N, (x, y) => default(SimpleBoundaryCondition).Temperature(x, y));
            double sum = 0.0, max = 0.0;
            for (int y = 1; y < N; ++y)
            {
                for (int x = 1; x < N; ++x)
                {
                    double error = Math.Abs(problem.Temperature(x, y) - exact[x, y]);
                    sum += error;
                    max = Math.Max(max, error);
                }
            }
            double mean = sum / ((N - 1) * (N - 1));
            Console.WriteLine($"difference to the harmonic function: mean {mean:G4}, largest {max:G4}");
            if (!(mean < 0.03 && max < 0.2))
            {
                Console.Error.WriteLine("ERROR : the walks do not estimate the temperature");
                Environment.Exit(6); // abort
            }
        }

        // u = the average of its four neighbours inside the square of side N, the temperature of the boundary on it
        // (at the points (x / N, y / N)), by successive over-relaxation
        static float[,] SolveLaplace(int N, Func<float, float, float> boundary)
        {
            var u = new float[N + 1, N + 1];
            float h = 1.0F / N;
            for (int k = 0; k <= N; ++k)
            {
                u[0, k] = boundary(0.0F, k * h);
                u[N, k] = boundary(N * h, k * h);
                u[k, 0] = boundary(k * h, 0.0F);
                u[k, N] = boundary(k * h, N * h);
            }
            double omega = 2.0 / (1.0 + Math.Sin(Math.PI / N));
            for (int sweep = 0; sweep < 10 * N; ++sweep)
            {
                double change = 0.0;
                for (int x = 1; x < N; ++x)
                {
                    for (int y = 1; y < N; ++y)
                    {
                        double next = (u[x - 1, y] + u[x + 1, y] + u[x, y - 1] + u[x, y + 1]) / 4.0;
                        double delta = omega * (next - u[x, y]);
                        u[x, y] += (float)delta;
                        change = Math.Max(change, Math.Abs(delta));
                    }
                }
                if (change < 1.0E-6)
                    break;
            }
            return u;
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