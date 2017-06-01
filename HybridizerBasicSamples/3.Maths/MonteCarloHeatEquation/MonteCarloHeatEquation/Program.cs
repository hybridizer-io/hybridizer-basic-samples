using Hybridizer.Runtime.CUDAImports;
using System;
using System.Drawing;

namespace MonteCarloHeatEquation
{
    class Program
    {
        public static float clamp(float x, float min, float max)
        {
            return x > min ? ((x < max) ? x : max) : min;
        }

        static Color Interpolate(Color a, Color b, float t)
        {
            return Color.FromArgb((int) (t * a.R + (1.0F - t) * b.R), (int) (t * a.G + (1.0F - t) * b.G), (int) (t * a.B + (1.0F - t) * b.B));
        }

        static Color GetColor(float temperature)
        {
            int map = (int)Math.Floor(temperature * 8.0F);
            if (temperature <= 0.0F)
                return Color.Black;
            if (temperature >= 1.0F)
                return Color.White;
            float t = 8.0F * temperature - (float) Math.Floor(temperature * 8.0F);
            Color[] colors = new Color[9] { Color.Black, Color.Red, Color.Orange, Color.Yellow, Color.Green, Color.Blue, Color.Indigo, Color.Violet, Color.White };
            return Interpolate(colors[map], colors[map + 1], 1.0F-t);
        }

        static void Main(string[] args)
        {
            const int N = 256;
            const int iterCount = 100;
            
            SquareProblem<SimpleWalker> problem = new SquareProblem<SimpleWalker>(N, ambientTemperature, iterCount);
            problem.RefreshDevice();

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);

            HybRunner runner = HybRunner.Cuda("MonteCarloHeatEquation_CUDA.dll").SetDistrib(16 * prop.multiProcessorCount, 128);
            var solver = new MonteCarloHeatSolver(problem);
            //solver.Solve();
            dynamic wrapped = runner.Wrap(solver);
            wrapped.Solve();

            problem.RefreshHost();
            problem.SaveImage("square.bmp", GetColor);
        }

        private static float ambientTemperature(float x, float y)
        {
            if ((x == 1.0F && y >= 0.5F) || (x == 0.0F && y <= 0.5F))
                return 1.0F;
            return 0.0F;
        }
    }
}
