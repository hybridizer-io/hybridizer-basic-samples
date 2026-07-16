using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Hybridizer.Basic.Utilities;

namespace Malloc
{
    /// <summary>
    /// Toy example showing usage of thread-local alloc/free
    /// No physical meaning at all
    /// </summary>
    unsafe class Program
    {
        [Kernel]
        public static double apply(double[] stencil, double[] src, int i)
        {
            double res = 0.0;
            for(int k = -4; k <= 4; ++k)
            {
                res += stencil[k+4] * src[i + k];
            }

            return res;
        }
        
        [EntryPoint]
        public static void test([Out] double[] dest, [In] double[] src, int N)
        {
            double[] stencil = new double[9];
            for(int i = 0; i < 9; ++i) { stencil[i] = i - 4.0; }

            for (int k = 4 + threadIdx.x + blockIdx.x * blockDim.x; k < N - 4; k += blockDim.x * gridDim.x)
            {
                dest[k] = apply(stencil, src, k);
            }
        }

        static void Main(string[] args)
        {
            const int N = 1024 * 1024 * 32;
            double[] src = new double[N];
            double[] dst = new double[N];
            Random rand = new();
            for (int i = 0; i < N; ++i)
            {
                src[i] = rand.NextDouble();
                dst[i] = src[i];
            }

            cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);

            HybRunner runner = SatelliteLoader.Load().SetDistrib(prop.multiProcessorCount, 512);
            dynamic wrapper = runner.Wrap(new Program());

            var sw = System.Diagnostics.Stopwatch.StartNew();
            wrapper.test(dst, src, N);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
            sw.Stop();

            Console.WriteLine($"GPU time for {N:N0} elements : {sw.ElapsedMilliseconds} ms");

            Console.WriteLine("Result Sample (from i = 4 to 13) :");
            for (int i = 4; i < 14; ++i)
                Console.Write($"{dst[i]:F4}, ");
            Console.WriteLine();
        }
    }
}
