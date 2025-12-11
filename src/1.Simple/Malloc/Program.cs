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
        public static void test(double[] dest, double[] src, int N)
        {
            double[] stencil =
            [
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ];
            for (int k = 4 + threadIdx.x + blockIdx.x * blockDim.x; k < N - 4; k += blockDim.x * gridDim.x)
            {
                dest[k] = apply(stencil, src, k);
            }
        }

        static void Main(string[] args)
        {
            const int N = 1024*1024*32;
            double[] src = new double[N];
            double[] dst = new double[N];
            Random rand = new Random();
            for(int i = 0; i < N; ++i)
            {
                src[i] = rand.NextDouble();
                dst[i] = src[i];
            }
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);

            HybRunner runner = SatelliteLoader.Load().SetDistrib(prop.multiProcessorCount, 512);
            dynamic wrapper = runner.Wrap(new Program());
            wrapper.test(dst, src, N);
        }
    }
}