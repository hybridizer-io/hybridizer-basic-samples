using Hybridizer.Runtime.CUDAImports;
using System.Drawing;
using System.Diagnostics;
using System;
using System.Threading.Tasks;

namespace Mandelbrot
{
    class Program
    {
        const int maxiter = 256;
        const int N = 1024;
        const float fromX = -2.0f;
        const float fromY = -2.0f;
        const float size = 4.0f;
        const float h = size / (float)N;

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

        [EntryPoint("run")]
        public static void Run(int[] light, int lineFrom, int lineTo)
        {
            for (int line = blockIdx.x + lineFrom; line < lineTo; line += gridDim.x)
            {
                for (int j = threadIdx.x; j < N; j += blockDim.x)
                {
                    float x = fromX + line * h;
                    float y = fromY + j * h;
                    light[line * N + j] = IterCount(x, y);
                }
            }
        }

        private static dynamic wrapper;

        public static void ComputeImage(int[] light, bool accelerate = true)
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

        static void Main(string[] args)
        {
            const int redo = 20;

            int[] light_net = new int[N * N];
            int[] light_cuda = new int[N * N];

            #region c#
            for (int i = 0; i < redo; ++i)
            {
                ComputeImage(light_net, false);
            }
            #endregion c#

            HybRunner runner = HybRunner.Cuda("Mandelbrot_CUDA.dll").SetDistrib(N, 256);
            wrapper = runner.Wrap(new Program());

            #region cuda
            for (int i = 0; i < redo; ++i)
            {
                ComputeImage(light_cuda, true);
            }
            #endregion

            #region save to image
            Color[] colors = new Color[maxiter + 1];
            Random rand = new Random(42);

            for (int k = 0; k < maxiter; ++k)
            {
                colors[k] = Color.FromArgb(rand.Next(256), rand.Next(256), rand.Next(256));
            }
            colors[maxiter] = Color.Black;

            Bitmap image = new Bitmap(N, N);
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    int index = i * N + j;
                    image.SetPixel(i, j, colors[light_cuda[index]]);
                }
            }

            image.Save("mandelbrot.png", System.Drawing.Imaging.ImageFormat.Png);
            #endregion

            Process.Start("mandelbrot.png");
        }
    }
}

