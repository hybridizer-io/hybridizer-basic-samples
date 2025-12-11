using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Hybridizer.Basic.Utilities;

namespace Recursion
{
    /**
     * This sample demonstrates simple recursion in a kernel function
     */
    class Program
    {
        [Kernel]
        public static int Fact(int N)
        {
            // stack allocations, such as new StackArray or stackalloc are forbidden in recursive functions
            if (N <= 1)
                return 1;
            return N * Fact(N - 1);
        }

        [EntryPoint]
        public static void Run(int N, int[] a, int[] b)
        {
            // calling a recursive function from a Parallel.For is not yet supported
            // work distribution must be explicit
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; i < N; i += blockDim.x * gridDim.x)
            {
                a[i] = Fact(b[i]);
            }
        }

        static void Main(string[] args)
        {
            const int N = 1024 * 1024 * 32;
            int[] a = new int[N];
            int[] b = new int[N];
            for(int i = 0; i < N; ++i)
            {
                a[i] = 0;
                b[i] = i % 11;
            }

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            //if .SetDistrib is not used, the default is .SetDistrib(prop.multiProcessorCount * 16, 128)
            HybRunner runner = SatelliteLoader.Load();

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());

            wrapped.Run(N, a, b);
            cuda.ERROR_CHECK(cuda.GetLastError());
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());

            for (int i = 0; i < N; ++i)
            {
                if(a[i] != Fact(b[i]))
                {
                    Console.Error.WriteLine("Error at {0} : {1} != {2}", i, a[i], Fact(b[i]));
                    Environment.Exit(6); // abort
                }
            }

            Console.Out.WriteLine("OK");
        }
    }
}