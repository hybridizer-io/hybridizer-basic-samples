using Hybridizer.Runtime.CUDAImports;
using System.Drawing;
using System.Diagnostics;
using System;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

namespace Hybridizer.Basic.Maths
{
    class Program
    {
        const int maxiter = 4096;
        const int N = 2048;
        const float fromX = -1.5f;
        const float fromY = -1.5f;
        const float size = 3.0f;
        const float h = size / (float)N;
        const float tol = 0.0000001f;

        [Kernel]
        public static void IterCount(ref int2 result, float cx, float cy)
        {
            int itercount = 0;
            int root = 0;
            float x = cx;
            float y = cy;
            float xx = 0.0f, xy = 0.0f, yy = 0.0f, xxy = 0.0f, xyy = 0.0f, xxx = 0.0f, yyy = 0.0f, yyyy = 0.0f, xxxx = 0.0f, xxxxx = 0.0f;
            while (itercount < maxiter)
            {
                xy = x * y;
                xx = x * x;
                yy = y * y;
                xyy = x * yy;
                xxy = xx * y;
                xxx = xx * x;
                yyy = yy * y;
                xxxx = xx * xx;
                yyyy = yy * yy;
                xxxxx = xxx * xx;

                float invdenum = 1.0f / (3.0f * xxxx + 6.0f * xx * yy + 3.0f * yyyy);

                float numreal = 2.0f * xxxxx + 4.0f * xxx * yy + xx + 2.0f * x * yyyy - yy;
                float numim = 2.0f * xxxx * y + 4.0f * xx * yyy - 2.0f * x * y + 2.0f * yyy * yy;

                x = numreal * invdenum;
                y = numim * invdenum;
                itercount++;

                root = RootFind(x, y);
                if (root > 0)
                {
                    break;
                }
            }

            result.x = root;
            result.y = itercount;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("sqrtf")]
        private static float sqrtf(float a)
        {
            return (float)Math.Sqrt(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("fabsf")]
        private static float fabsf(float a)
        {
            return (float)Math.Abs(a);
        }


        const float sqrtRoot = 0.86602540378443864676372317075294f; //(float)sqrtf(3.0f / 4.0f);

        [Kernel]
        public static int RootFind(float x, float y)
        {
            if(fabsf(x - 1.0f) < tol && fabsf(y) < tol)
            {
                return 1;
            }
            else if (fabsf(x + 0.5f) < tol && fabsf(y - sqrtRoot) < tol)
            {
                return 2;
            }
            else if (fabsf(x + 0.5f) < tol && fabsf(y + sqrtRoot) < tol)
            {
                return 3;
            }

            return 0;
        }

        [EntryPoint("run")]
        public static void Run(int2[] results, int lineFrom, int lineTo)
        {
            for (int i = lineFrom + threadIdx.y + blockIdx.y * blockDim.y; i < lineTo; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < N; j += blockDim.x * gridDim.x)
                 {
                    float x = fromX + i * h;
                    float y = fromY + j * h;
                    IterCount(ref results[i * N + j], x, y);
                }
            }
        }

        private static dynamic wrapper;

        public static void ComputeImage(int2[] results, bool accelerate = true)
        {
            if (accelerate)
            {
                wrapper.Run(results, 0, N);
            }
            else
            {
                Parallel.For(0, N, (line) =>
                {
                    Run(results, line, line + 1);
                });
            }
        }

        static int ComputeLight(int iter)
        {
            return System.Math.Min(iter*16,255);
        }

        static void Main(string[] args)
        {
            const int redo = 1;
            int2[] result_net = new int2[N * N];
            int2[] result_cuda = new int2[N * N];

            #region c#

            Stopwatch watch = new Stopwatch();
            watch.Start();

            for (int i = 0; i < redo; ++i)
            {
                ComputeImage(result_net, false);
            }

            watch.Stop();
            double firstWatchResult = 1.0E-6 * ((double)(N * N) * (double)redo / (1.0E-3 * watch.ElapsedMilliseconds));

            #endregion c#

            HybRunner runner = HybRunner.Cuda("Newton_CUDA.dll").SetDistrib(32, 32, 16, 16, 1, 0);
            wrapper = runner.Wrap(new Program());

            #region cuda
            
            for (int i = 0; i < redo; ++i)
            {
                if(i == 1) // skip first call to skip cubin link phase
                    watch.Restart();

                ComputeImage(result_cuda, true);
            }
            watch.Stop();

            Console.WriteLine("C#   MPixels/s :  {0}", firstWatchResult);
            Console.WriteLine("CUDA MPixels/s :  {0}", 1.0E-6 * ((double)(N * N) * (double)redo / (1.0E-3 * watch.ElapsedMilliseconds)));
            Console.WriteLine("without memcpy :  {0}", 1.0E-6 * ((double)(N * N) / (1.0E-3 * runner.LastKernelDuration.ElapsedMilliseconds)));

            #endregion

            #region save to image

            Bitmap image = new Bitmap(N, N);

            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {

                    int index = i * N + j;
                    int root = result_cuda[index].x;
                    int light = ComputeLight(result_cuda[index].y);

                    switch (root)
                    {
                        case 0:
                            image.SetPixel(i, j, Color.Black);
                            break;
                        case 1:
                            image.SetPixel(i, j, Color.FromArgb(light, 0 ,0));
                            break;
                        case 2:
                            image.SetPixel(i, j, Color.FromArgb(0, 0, light));
                            break;
                        case 3:
                            image.SetPixel(i, j, Color.FromArgb(0, light, 0));
                            break;
                        default:
                            throw new ApplicationException();
                    }
                }
            }

            image.Save("newton.png", System.Drawing.Imaging.ImageFormat.Png);
            #endregion

            Process.Start("newton.png");
        }
    }
}

