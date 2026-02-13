using Hybridizer.Runtime.CUDAImports;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Hybridizer.Basic.Utilities;
using SixLabors.ImageSharp.Formats.Png;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace Mandelbrot
{
    class Program
    {
        const int maxiter = 32;
        const int N = 4096;
        const float fromX = -2.0f;
        const float fromY = -2.0f;
        const float size = 4.0f;
        const float h = size / N;

        [Kernel]
        public static int IterCount(float cx, float cy)
        {
            int result = 0;
            float x = 0.0f;
            float y = 0.0f;
            float xx = 0.0f, yy = 0.0f;
            while (xx + yy <= 4.0f && result < maxiter)
            {
                xx = x * x;
                yy = y * y;
                float xtmp = xx - yy + cx;
                y = 2.0f * x * y + cy;
                x = xtmp;
                result++;
            }

            return result;
        }

        [EntryPoint]
        public static void Run(IntResidentArray light, int lineFrom, int lineTo)
        {
            for (int line = lineFrom + threadIdx.y + blockDim.y * blockIdx.y; line < lineTo; line += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < N; j += blockDim.x * gridDim.x)
                {
                    float x = fromX + line * h;
                    float y = fromY + j * h;
                    light[line * N + j] = IterCount(x, y);
                }
            }
        }

        private static dynamic wrapper;

        public static void ComputeImage(IntResidentArray light, bool accelerate = true)
        {
            if (accelerate)
            {
                wrapper.Run(light, 0, N);
            }
            else
            {
                Parallel.For(0, N, (line) =>
                {
                    Run(light, line, line + 1);
                });
            }
        }

        static void Main()
        {
            const int redo = 20;

            IntResidentArray light_net = new(N*N);
            IntResidentArray light_cuda = new(N*N);

            #region c#
            Stopwatch w = new();
            w.Start();
            for (int i = 0; i < redo; ++i)
            {
                ComputeImage(light_net, false);
            }
            w.Stop();
            Console.WriteLine($"elapsed time per image (C#) : {w.ElapsedMilliseconds/redo} ms");
            #endregion c#

            HybRunner runner = SatelliteLoader.Load().SetDistrib(32, 32, 16, 16, 1, 0);
            wrapper = runner.Wrap(new Program());
            // profile with nsight to get performance
            w.Reset();
            w.Start();
            #region cuda
            for (int i = 0; i < redo; ++i)
            {
                ComputeImage(light_cuda, true);
                light_cuda.RefreshHost(); // included for fair comparison
            }
            w.Stop();
            Console.WriteLine($"elapsed time per image (CUDA) : {w.ElapsedMilliseconds/redo} ms");
            #endregion

            #region save to image
            Color[] colors = new Color[maxiter + 1];

            for (int k = 0; k < maxiter; ++k)
            {
                byte red = (byte)  (127.0F * k / maxiter);
                byte green = (byte)(200.0F * k / maxiter);
                byte blue = (byte) (90.0F * k / maxiter);
                colors[k] = Color.FromRgb(red, green, blue);
            }
            colors[maxiter] = Color.Black;

            var image = new Image<Rgba32>(N, N);
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    int index = i * N + j;
                    image[i, j] = colors[light_cuda[index]];
                }
            }

            image.Save("mandelbrot.png", new PngEncoder());
            #endregion
        }
    }
}
