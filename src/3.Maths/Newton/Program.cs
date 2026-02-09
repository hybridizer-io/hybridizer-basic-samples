using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Newton
{
    class Program
    {
        const int maxiter = 4096;
        const int N = 4096;
        const float fromX = -1.5f;
        const float fromY = -1.5f;
        const float size = 3.0f;
        const float h = size / N;
        const float tol = 0.0000001f;

        [Kernel]
        public static int2 IterCount(float cx, float cy)
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

            return new int2 { x = root, y = itercount };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("sqrtf")]
        private static float Sqrtf(float a)
        {
            return (float)Math.Sqrt(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("fabsf")]
        private static float Fabsf(float a)
        {
            return (float)Math.Abs(a);
        }


        const float sqrtRoot = 0.86602540378443864676372317075294f; //(float)sqrtf(3.0f / 4.0f);

        [Kernel]
        public static int RootFind(float x, float y)
        {
            if(Fabsf(x - 1.0f) < tol && Fabsf(y) < tol)
            {
                return 1;
            }
            else if (Fabsf(x + 0.5f) < tol && Fabsf(y - sqrtRoot) < tol)
            {
                return 2;
            }
            else if (Fabsf(x + 0.5f) < tol && Fabsf(y + sqrtRoot) < tol)
            {
                return 3;
            }

            return 0;
        }

        [EntryPoint("run")]
        public static void Run(ResidentArrayGeneric<int2> results)
        {
            Parallel2D.For(0, N, 0, N, (i, j) => {
                float x = fromX + i * h;
                float y = fromY + j * h;
                results[i * N + j] = IterCount(x, y);
            });
        }

        private static dynamic wrapper;

        public static void ComputeImage(ResidentArrayGeneric<int2> results, bool accelerate = true)
        {
            if (accelerate)
            {
                wrapper.Run(results);
            }
            else
            {
                Run(results);
            }
        }

        static byte ComputeLight(int iter)
        {
            return (byte) Math.Min(iter*16, 255);
        }

        static void Main()
        {
            const int redo = 4;
            ResidentArrayGeneric<int2> result_net = new(N * N);
            ResidentArrayGeneric<int2> result_cuda = new(N * N);

            #region c#
            Stopwatch stopwatch = new();
            stopwatch.Start();
            for (int i = 0; i < redo; ++i)
            {
                ComputeImage(result_net, false);
            }
            stopwatch.Stop();
            Console.WriteLine($"C# time per image : {stopwatch.ElapsedMilliseconds / redo} ms");
            
            #endregion c#

            HybRunner runner = SatelliteLoader.Load().SetDistrib(32, 32, 16, 16, 1, 0);
            wrapper = runner.Wrap(new Program());

            #region cuda
            
            stopwatch.Reset();
            stopwatch.Start();
            for (int i = 0; i < redo; ++i)
            {
                ComputeImage(result_cuda, true);
            }
            stopwatch.Stop();
            Console.WriteLine($"CUDA time per image : {stopwatch.ElapsedMilliseconds / redo} ms");

            #endregion

            #region save to image

            var image = new Image<Argb32>(N, N);
            result_cuda.RefreshHost();
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    int index = i * N + j;
                    int root = result_cuda[index].x;
                    byte light = ComputeLight(result_cuda[index].y);

                    image[i, j] = root switch
                    {
                        0 => (Argb32)Color.Black,
                        1 => (Argb32)Color.FromRgb(light, 0, 0),
                        2 => (Argb32)Color.FromRgb(0, 0, light),
                        3 => (Argb32)Color.FromRgb(0, light, 0),
                        _ => throw new ApplicationException(),
                    };
                }
            }

            image.Save("newton.png", new PngEncoder());
            #endregion
        }
    }
}
