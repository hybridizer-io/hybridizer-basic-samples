using Hybridizer.Runtime.CUDAImports;
using System;
using System.Diagnostics;
using System.Drawing;

namespace MonteCarloHeatEquation
{
    class Program
    {
        static void Main(string[] args)
        {
            const int N = 128;
            const int iterCount = 1000;
            
            var problem = new SquareProblem<SimpleWalker, SimpleBoundaryCondition>(N, iterCount);
            // example of another instanciation
            // var problem = new TetrisProblem<SimpleWalker, TetrisBoundaryCondition>(N, iterCount);

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);

            HybRunner runner = HybRunner.Cuda("MonteCarloHeatEquation_CUDA.dll").SetDistrib(16 * prop.multiProcessorCount, 128);
            var solver = new MonteCarloHeatSolver(problem);
            dynamic wrapped = runner.Wrap(solver);
            //solver.Solve();   // C# version
            wrapped.Solve();    // generated version

            problem.RefreshHost();
            problem.SaveImage("result.bmp", GetColor);
			try { Process.Start("result.bmp");} catch {} // catch exception for non interactives machines
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
            Color[] colors = new Color[9] { Color.Black, Color.Red, Color.Orange, Color.Yellow, Color.Green, Color.Blue, Color.Indigo, Color.Violet, Color.White };
            return Interpolate(colors[map], colors[map + 1], t);
        }

        static Color Interpolate(Color a, Color b, float t)
        {
            return Color.FromArgb((int)((1.0F - t) * a.R + t * b.R), (int)((1.0F - t) * a.G + t * b.G), (int)((1.0F - t) * a.B + t * b.B));
        }
    }
}
